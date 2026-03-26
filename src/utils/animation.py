from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Literal

import colorsys
import numpy as np
import torch
import pyglet
import pyglet.shapes

from integrators.base import Simulator


PANEL_MAIN_W = 370
PANEL_HIST_W = 95
HIST_PAD = 20
PANEL_GAP = 50
MARGIN_X = 70
H = 680
PLOT_B = 60
PLOT_T = H - 75
TRAJ_INTERVAL = 0.04
N_BINS = 30

_GRAY = (160, 160, 160, 255)
_DIM = (55, 55, 55, 255)
_TITLE = (200, 200, 200, 255)
_HIST_FILL = (210, 120, 50, 190)
_PDF_LINE = (240, 70, 70, 230)
_FS, _FM, _FL = 9, 11, 13


def _traj_color(i: int, n: int, alpha: int = 150) -> tuple[int, int, int, int]:
    h = (i / max(n - 1, 1)) * 0.78
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255), alpha


@dataclass
class PanelLayout:
    main_l: int
    main_r: int
    hist_l: int
    hist_r: int

    def to_screen(self, t, x, t_min, t_max, x_min, x_max) -> tuple[float, float]:
        sx = self.main_l + (t - t_min) / (t_max - t_min) * (self.main_r - self.main_l)
        sy = PLOT_B + (x - x_min) / (x_max - x_min) * (PLOT_T - PLOT_B)
        return sx, sy

    def x_to_sy(self, x_val, x_min, x_max) -> float:
        return PLOT_B + (x_val - x_min) / (x_max - x_min) * (PLOT_T - PLOT_B)


def _make_layouts(n: int) -> list[PanelLayout]:
    panel_w = PANEL_MAIN_W + HIST_PAD + PANEL_HIST_W
    return [
        PanelLayout(
            main_l=(left := MARGIN_X + i * (panel_w + PANEL_GAP)),
            main_r=left + PANEL_MAIN_W,
            hist_l=left + PANEL_MAIN_W + HIST_PAD,
            hist_r=left + PANEL_MAIN_W + HIST_PAD + PANEL_HIST_W,
        )
        for i in range(n)
    ]


@dataclass
class SimulationData:
    """Pre-computed trajectory data for any SDE. Decoupled from visualization."""

    trajs: np.ndarray  # (n_traj, n_steps)
    ts: np.ndarray  # (n_steps,)
    title: str = ""
    theoretical_pdf: Optional[Callable[[np.ndarray], np.ndarray]] = None

    @classmethod
    def from_sde(
        cls,
        sde,
        simulator: Simulator,
        x0: torch.Tensor,
        ts: torch.Tensor,
        title: str = "",
        theoretical_pdf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> SimulationData:
        with torch.no_grad():
            trajs_tensor = simulator.simulate_with_trajectory(x0, ts)
        return cls(
            trajs=trajs_tensor[:, :, 0].detach().cpu().numpy(),
            ts=ts.detach().cpu().numpy(),
            title=title,
            theoretical_pdf=theoretical_pdf,
        )


class SDEVisualizer:
    """
    Pyglet-based animator for one or more SDE simulations rendered side by side.

    Trajectories are added to all panels in lock-step: one per panel per clock
    tick. This lets the user compare how different processes build up their
    terminal distributions at the same pace.

    When a `theoretical_pdf` is attached to a SimulationData, the histogram
    density scale is anchored to its peak so bars and curve stay on the same
    scale throughout the animation. Without it, the histogram auto-scales to
    the current empirical max, which re-normalizes as more trajectories arrive.
    """

    def __init__(
        self,
        simulations: list[SimulationData],
        traj_interval: float = TRAJ_INTERVAL,
        n_bins: int = N_BINS,
    ):
        self.sims = simulations
        self.n = len(simulations)
        self.n_bins = n_bins
        self.layouts = _make_layouts(self.n)

        panel_w = PANEL_MAIN_W + HIST_PAD + PANEL_HIST_W
        W = 2 * MARGIN_X + self.n * panel_w + (self.n - 1) * PANEL_GAP

        self.window = pyglet.window.Window(width=W, height=H, caption="SDE Visualizer")
        pyglet.gl.glClearColor(0.08, 0.08, 0.1, 1.0)

        self.traj_batch = pyglet.graphics.Batch()
        self.hist_batch = pyglet.graphics.Batch()
        self.axis_batch = pyglet.graphics.Batch()

        self._shapes: list = []
        self._labels: list = []

        self.shown = [0] * self.n
        self.terminal_vals: list[list[float]] = [[] for _ in range(self.n)]
        self.traj_segments: list[list] = [[] for _ in range(self.n)]
        self.bin_rects: list[list] = [[] for _ in range(self.n)]
        self.bin_edges: list[np.ndarray] = []
        self.density_scales: list[float] = []

        self.bounds: list[tuple[float, float, float, float]] = []
        for sim in simulations:
            span = sim.trajs.max() - sim.trajs.min()
            margin = max(0.5, span * 0.05)
            self.bounds.append(
                (
                    float(sim.ts.min()),
                    float(sim.ts.max()),
                    float(sim.trajs.min()) - margin,
                    float(sim.trajs.max()) + margin,
                )
            )

        for i in range(self.n):
            t_min, t_max, x_min, x_max = self.bounds[i]
            self._build_panel_axes(i, self.layouts[i], t_min, t_max, x_min, x_max)
            self._init_histogram(i, self.layouts[i], x_min, x_max)
            if simulations[i].theoretical_pdf is not None:
                self._build_pdf_curve(i, self.layouts[i], x_min, x_max)

        pyglet.clock.schedule_interval(self._step, traj_interval)
        self.window.event(self.on_draw)

    def _line(self, p0, p1, color):
        self._shapes.append(
            pyglet.shapes.Line(*p0, *p1, color=color, batch=self.axis_batch)
        )

    def _label(
        self,
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

    def _box(self, l, r):
        for p0, p1 in [
            ((l, PLOT_B), (r, PLOT_B)),
            ((r, PLOT_B), (r, PLOT_T)),
            ((r, PLOT_T), (l, PLOT_T)),
            ((l, PLOT_T), (l, PLOT_B)),
        ]:
            self._line(p0, p1, _GRAY)

    def _build_panel_axes(self, idx, layout, t_min, t_max, x_min, x_max):
        self._box(layout.main_l, layout.main_r)
        self._box(layout.hist_l, layout.hist_r)

        for t_val in np.arange(np.ceil(t_min), np.floor(t_max) + 1, 1.0):
            sx = layout.main_l + (t_val - t_min) / (t_max - t_min) * (
                layout.main_r - layout.main_l
            )
            self._line((sx, PLOT_B), (sx, PLOT_T), _DIM)
            self._label(f"{int(t_val)}", sx, PLOT_B - 12, anchor_y="top", size=_FS)

        x_step = max(1.0, round((x_max - x_min) / 8))
        for x_val in np.arange(np.ceil(x_min / x_step) * x_step, x_max, x_step):
            sy = layout.x_to_sy(x_val, x_min, x_max)
            self._line((layout.main_l, sy), (layout.main_r, sy), _DIM)
            self._line((layout.hist_l, sy), (layout.hist_r, sy), _DIM)
            self._label(
                f"{x_val:.0f}", layout.main_l - 10, sy, anchor_x="right", size=_FS
            )

        mx = (layout.main_l + layout.main_r) // 2
        self._label("time (t)", mx, 14, anchor_y="bottom")
        self._label("Xt", layout.main_l - 35, (PLOT_B + PLOT_T) // 2)
        self._label(
            "density", (layout.hist_l + layout.hist_r) // 2, 14, anchor_y="bottom"
        )

        if title := self.sims[idx].title:
            cx = (layout.main_l + layout.hist_r) // 2
            self._label(title, cx, H - 18, anchor_y="top", size=_FL, color=_TITLE)

    def _init_histogram(self, idx, layout, x_min, x_max):
        bin_edges = np.linspace(x_min, x_max, self.n_bins + 1)
        self.bin_edges.append(bin_edges)

        hist_w = layout.hist_r - layout.hist_l
        if (pdf := self.sims[idx].theoretical_pdf) is not None:
            x_sample = np.linspace(x_min, x_max, 500)
            self.density_scales.append(hist_w * 0.88 / float(pdf(x_sample).max()))
        else:
            self.density_scales.append(hist_w * 0.88)

        for i in range(self.n_bins):
            sy_bot = layout.x_to_sy(bin_edges[i], x_min, x_max)
            sy_top = layout.x_to_sy(bin_edges[i + 1], x_min, x_max)
            self.bin_rects[idx].append(
                pyglet.shapes.Rectangle(
                    x=layout.hist_l,
                    y=sy_bot,
                    width=1,
                    height=max(1, sy_top - sy_bot),
                    color=_HIST_FILL,
                    batch=self.hist_batch,
                )
            )

    def _build_pdf_curve(self, idx, layout, x_min, x_max):
        x_curve = np.linspace(x_min, x_max, 300)
        pdf_vals = self.sims[idx].theoretical_pdf(x_curve)
        scale = self.density_scales[idx]
        pts = [
            (layout.hist_l + p * scale, layout.x_to_sy(xv, x_min, x_max))
            for xv, p in zip(x_curve, pdf_vals)
        ]
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            self._shapes.append(
                pyglet.shapes.Line(
                    ax, ay, bx, by, color=_PDF_LINE, batch=self.axis_batch
                )
            )

    def _update_hist(self, idx):
        if not self.terminal_vals[idx]:
            return
        layout = self.layouts[idx]
        _, _, x_min, x_max = self.bounds[idx]
        bin_edges = self.bin_edges[idx]
        bin_width = bin_edges[1] - bin_edges[0]

        counts, _ = np.histogram(self.terminal_vals[idx], bins=bin_edges)
        densities = counts / (len(self.terminal_vals[idx]) * bin_width)

        if self.sims[idx].theoretical_pdf is None:
            if (max_d := densities.max()) > 0:
                self.density_scales[idx] = (layout.hist_r - layout.hist_l) * 0.8 / max_d

        scale = self.density_scales[idx]
        hist_w = layout.hist_r - layout.hist_l
        for j, rect in enumerate(self.bin_rects[idx]):
            rect.width = max(1.0, min(densities[j] * scale, hist_w))

    def _step(self, dt):
        if all(self.shown[i] >= self.sims[i].trajs.shape[0] for i in range(self.n)):
            pyglet.clock.unschedule(self._step)
            return

        for i, sim in enumerate(self.sims):
            if self.shown[i] >= sim.trajs.shape[0]:
                continue
            j = self.shown[i]
            color = _traj_color(j, sim.trajs.shape[0])
            t_min, t_max, x_min, x_max = self.bounds[i]
            layout = self.layouts[i]

            pts = [
                layout.to_screen(sim.ts[k], sim.trajs[j, k], t_min, t_max, x_min, x_max)
                for k in range(len(sim.ts))
            ]
            for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                self.traj_segments[i].append(
                    pyglet.shapes.Line(
                        ax, ay, bx, by, color=color, batch=self.traj_batch
                    )
                )

            self.terminal_vals[i].append(float(sim.trajs[j, -1]))
            self._update_hist(i)
            self.shown[i] += 1

    def on_draw(self):
        self.window.clear()
        self.traj_batch.draw()
        self.hist_batch.draw()
        self.axis_batch.draw()

    def run(self):
        pyglet.app.run()
