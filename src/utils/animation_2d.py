from __future__ import annotations

from typing import Literal
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pyglet
import pyglet.shapes
import pyglet.sprite
import pyglet.image
import pyglet.gl
from PIL import Image

from integrators.base import Simulator

_W, _H = 840, 840
_MARGIN = 65
_PLOT_L, _PLOT_R = _MARGIN, _W - _MARGIN
_PLOT_B, _PLOT_T = _MARGIN, _H - 70

_STEP_INTERVAL = 0.03
_PARTICLE_RADIUS = 3
_PARTICLE_COLOR = (255, 165, 50, 190)
_GRAY = (160, 160, 160, 255)
_DIM = (55, 55, 55, 255)
_TITLE_C = (200, 200, 200, 255)
_FS, _FM, _FL = 9, 11, 14


def _build_density_texture(
    density,
    scale: float,
    bins: int,
    device: torch.device,
) -> pyglet.image.Texture:
    xs = torch.linspace(-scale, scale, bins, device=device)
    ys = torch.linspace(-scale, scale, bins, device=device)
    # indexing="xy": gx[i,j]=xs[j], gy[i,j]=ys[i]  -> row i = y=ys[i]
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    with torch.no_grad():
        log_p = density.log_density(pts).reshape(bins, bins).float().cpu().numpy()

    vmin = -15.0
    clipped = np.clip(log_p, vmin, None)
    norm = (clipped - vmin) / max(float(clipped.max()) - vmin, 1e-6)

    r = ((1.0 - norm * 0.55) * 255).astype(np.uint8)
    g = ((1.0 - norm * 0.35) * 255).astype(np.uint8)
    b = np.full_like(r, 255)
    a = (norm * 200 + 30).astype(np.uint8)

    rgba = np.stack([r, g, b, a], axis=-1)
    # Row 0 = y=ys[0]=-scale. Pyglet's default positive pitch reads data
    # bottom-to-top, so row 0 lands at the image bottom — correct.
    img = pyglet.image.ImageData(bins, bins, "RGBA", rgba.tobytes())
    return img.get_texture()


@dataclass
class SimulationData2D:
    trajs: np.ndarray  # (n_samples, n_steps, 2)
    ts: np.ndarray  # (n_steps,)
    title: str = ""
    density: Optional[object] = None
    scale: float = 15.0
    density_bins: int = 150
    device: Optional[torch.device] = None

    @classmethod
    def from_sde(
        cls,
        sde,
        simulator: Simulator,
        x0: torch.Tensor,
        ts: torch.Tensor,
        title: str = "",
        density=None,
        scale: float = 15.0,
        density_bins: int = 150,
        device: Optional[torch.device] = None,
    ) -> SimulationData2D:
        with torch.no_grad():
            trajs = simulator.simulate_with_trajectory(x0, ts)
        return cls(
            trajs=trajs.detach().cpu().numpy(),
            ts=ts.detach().cpu().numpy(),
            title=title,
            density=density,
            scale=scale,
            density_bins=density_bins,
            device=device,
        )


class SDEVisualizer2D:
    """Pyglet animator for 2D diffusion. All particles advance simultaneously,
    showing how the sample cloud evolves under the SDE in real time.

    The target density is pre-rendered once as a texture so per-frame
    cost is just updating n_samples (x, y) attributes — no
    recomputation.
    """

    _bg: Optional[pyglet.sprite.Sprite] = None

    def __init__(
        self,
        sim: SimulationData2D,
        step_interval: float = _STEP_INTERVAL,
        steps_per_frame: int = 2,
        gif_path: Optional[str] = None,
        gif_capture_every: int = 3,
        gif_fps: int = 30,
        gif_scale: float = 0.5,
    ):
        self.sim = sim
        self.frame = 0
        self.n_steps = sim.trajs.shape[1]
        self.steps_per_frame = steps_per_frame

        self._gif_path = Path(gif_path) if gif_path else None
        self._gif_capture_every = gif_capture_every
        self._gif_fps = gif_fps
        self._gif_scale = gif_scale
        self._gif_frames: list[Image.Image] = []
        self._draw_count = 0

        self.window = pyglet.window.Window(
            width=_W, height=_H, caption="SDE Visualizer 2D"
        )
        pyglet.gl.glClearColor(0.08, 0.08, 0.1, 1.0)

        self.axis_batch = pyglet.graphics.Batch()
        self.particle_batch = pyglet.graphics.Batch()
        self._shapes: list = []
        self._labels: list = []

        if sim.density is not None:
            tex = _build_density_texture(
                sim.density,
                sim.scale,
                sim.density_bins,
                sim.device or torch.device("cpu"),
            )
            self._bg = pyglet.sprite.Sprite(tex, x=_PLOT_L, y=_PLOT_B)
            plot_w = _PLOT_R - _PLOT_L
            plot_h = _PLOT_T - _PLOT_B

            tex_w, tex_h = tex.width, tex.height

            # uniform scaling (preserve aspect ratio)
            scale = min(plot_w / tex_w, plot_h / tex_h)

            # centered placement
            new_w = tex_w * scale
            new_h = tex_h * scale
            offset_x = _PLOT_L + (plot_w - new_w) / 2
            offset_y = _PLOT_B + (plot_h - new_h) / 2

            self._bg = pyglet.sprite.Sprite(tex)
            self._bg.scale = scale
            self._bg.x = offset_x
            self._bg.y = offset_y

            # store for particles
            self._scale = scale
            self._offset_x = offset_x
            self._offset_y = offset_y
            self._new_w = new_w
            self._new_h = new_h
        else:
            self._bg = None

        self._build_axes()

        x0 = sim.trajs[:, 0, :]
        self._circles = [
            pyglet.shapes.Circle(
                *self._to_screen(x0[i, 0], x0[i, 1]),
                radius=_PARTICLE_RADIUS,
                color=_PARTICLE_COLOR,
                batch=self.particle_batch,
            )
            for i in range(sim.trajs.shape[0])
        ]

        self._time_label = pyglet.text.Label(
            f"t = {sim.ts[0]:.2f}",
            x=(_PLOT_L + _PLOT_R) // 2,
            y=_H - 20,
            anchor_x="center",
            anchor_y="top",
            font_size=_FM,
            color=_TITLE_C,
        )

        pyglet.clock.schedule_interval(self._step, step_interval)
        self.window.event(self.on_draw)

    def _to_screen(self, x, y):
        s = self.sim.scale
        # normalized 0..1
        nx = (x + s) / (2 * s)
        ny = (y + s) / (2 * s)
        # map into the actual texture rectangle
        sx = self._offset_x + nx * self._new_w
        sy = self._offset_y + ny * self._new_h
        return sx, sy

    def _build_axes(self):
        scale = self.sim.scale

        def line(p0, p1, color):
            self._shapes.append(
                pyglet.shapes.Line(*p0, *p1, color=color, batch=self.axis_batch)
            )

        def label(
            text,
            x,
            y,
            anchor_x: Literal["center", "left", "right"] = "center",
            anchor_y: Literal["center", "bottom", "top", "baseline"] = "center",
            size=_FM,
            color=_GRAY,
        ):
            self._labels.append(
                pyglet.text.Label(
                    text,
                    x=x,
                    y=y,
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                    font_size=size,
                    color=color,
                    batch=self.axis_batch,
                )
            )

        for p0, p1 in [
            ((_PLOT_L, _PLOT_B), (_PLOT_R, _PLOT_B)),
            ((_PLOT_R, _PLOT_B), (_PLOT_R, _PLOT_T)),
            ((_PLOT_R, _PLOT_T), (_PLOT_L, _PLOT_T)),
            ((_PLOT_L, _PLOT_T), (_PLOT_L, _PLOT_B)),
        ]:
            line(p0, p1, _GRAY)

        step = max(1, int(np.ceil(scale)) // 5)
        for v in range(-int(np.ceil(scale)), int(np.ceil(scale)) + 1, step):
            sx, _ = self._to_screen(v, 0)
            _, sy = self._to_screen(0, v)
            line((_PLOT_L, sy), (_PLOT_R, sy), _DIM)
            line((sx, _PLOT_B), (sx, _PLOT_T), _DIM)
            label(f"{v}", sx, _PLOT_B - 12, anchor_y="top", size=_FS)
            label(f"{v}", _PLOT_L - 10, sy, anchor_x="right", size=_FS)

        label("x", (_PLOT_L + _PLOT_R) // 2, 14, anchor_y="bottom")
        label("y", 14, (_PLOT_B + _PLOT_T) // 2, anchor_x="left")

        if self.sim.title:
            label(
                self.sim.title,
                (_PLOT_L + _PLOT_R) // 2,
                _H - 40,
                size=_FL,
                color=_TITLE_C,
                anchor_y="top",
            )

    def _step(self, dt):
        next_frame = min(self.frame + self.steps_per_frame, self.n_steps - 1)
        if next_frame == self.frame:
            pyglet.clock.unschedule(self._step)
            if self._gif_path:
                self._finalize_gif()
            return
        self.frame = next_frame
        xt = self.sim.trajs[:, self.frame, :]
        for i, c in enumerate(self._circles):
            c.x, c.y = self._to_screen(float(xt[i, 0]), float(xt[i, 1]))
        self._time_label.text = f"t = {self.sim.ts[self.frame]:.2f}"

    def on_draw(self):
        self.window.clear()
        if self._bg is not None:
            self._bg.draw()
        self.particle_batch.draw()
        self.axis_batch.draw()
        self._time_label.draw()
        if self._gif_path and self._draw_count % self._gif_capture_every == 0:
            self._capture_frame()
        self._draw_count += 1

    def _capture_frame(self):
        # pyglet reads the back buffer bottom-to-top; flip vertically so the
        # image is upright when handed to Pillow.
        buf = pyglet.image.get_buffer_manager().get_color_buffer()
        raw = buf.get_image_data()
        arr = np.frombuffer(raw.get_data("RGB", raw.width * 3), dtype=np.uint8)
        arr = arr.reshape(raw.height, raw.width, 3)[::-1]
        img = Image.fromarray(arr, "RGB")
        if self._gif_scale != 1.0:
            w = int(img.width * self._gif_scale)
            h = int(img.height * self._gif_scale)
            img = img.resize((w, h))
        self._gif_frames.append(img)

    def _finalize_gif(self):
        if not self._gif_frames:
            return
        frame_duration_ms = int(1000 / self._gif_fps)
        print(f"Saving {len(self._gif_frames)} frames to {self._gif_path} ...")
        assert self._gif_path is not None
        self._gif_frames[0].save(
            self._gif_path,
            save_all=True,
            append_images=self._gif_frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
        print(f"Saved {self._gif_path}")

    def run(self):
        pyglet.app.run()
