import colorsys
import numpy as np
import torch
import pyglet
import pyglet.shapes
from PIL import Image

from processes.brownian import BrownianMotion
from simulator.euler_maruyama import EulerMaruyamaSimulator

W, H = 1350, 660
MAIN_L, MAIN_R = 70, 820
HIST_L, HIST_R = 880, 1300
PLOT_B, PLOT_T = 55, H - 65

TRAJ_INTERVAL = 0.05
N_BINS = 30
GIF_PATH = "brownian_motion.gif"
GIF_SCALE = 0.5
GIF_DURATION = int(TRAJ_INTERVAL * 1000)


def main_to_screen(t, x, t_min, t_max, x_min, x_max):
    sx = MAIN_L + (t - t_min) / (t_max - t_min) * (MAIN_R - MAIN_L)
    sy = PLOT_B + (x - x_min) / (x_max - x_min) * (PLOT_T - PLOT_B)
    return sx, sy


def x_to_sy(x_val, x_min, x_max):
    return PLOT_B + (x_val - x_min) / (x_max - x_min) * (PLOT_T - PLOT_B)


def traj_color(i, n, alpha=150):
    h = (i / max(n - 1, 1)) * 0.78
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255), alpha


def build_main_axes(t_min, t_max, x_min, x_max, batch):
    gray = (160, 160, 160, 255)
    dim = (55, 55, 55, 255)
    shapes, labels = [], []

    def line(p0, p1, color):
        shapes.append(pyglet.shapes.Line(*p0, *p1, color=color, batch=batch))

    for p0, p1 in [
        ((MAIN_L, PLOT_B), (MAIN_R, PLOT_B)),
        ((MAIN_R, PLOT_B), (MAIN_R, PLOT_T)),
        ((MAIN_R, PLOT_T), (MAIN_L, PLOT_T)),
        ((MAIN_L, PLOT_T), (MAIN_L, PLOT_B)),
    ]:
        line(p0, p1, gray)

    for t_val in np.arange(np.ceil(t_min), np.floor(t_max) + 1, 1.0):
        sx = MAIN_L + (t_val - t_min) / (t_max - t_min) * (MAIN_R - MAIN_L)
        line((sx, PLOT_B), (sx, PLOT_T), dim)
        labels.append(
            pyglet.text.Label(
                f"{int(t_val)}",
                x=sx,
                y=PLOT_B - 12,
                anchor_x="center",
                anchor_y="top",
                font_size=9,
                color=gray,
                batch=batch,
            )
        )

    x_tick_step = max(1.0, round((x_max - x_min) / 8))
    for x_val in np.arange(
        np.ceil(x_min / x_tick_step) * x_tick_step, x_max, x_tick_step
    ):
        sy = x_to_sy(x_val, x_min, x_max)
        line((MAIN_L, sy), (MAIN_R, sy), dim)
        labels.append(
            pyglet.text.Label(
                f"{x_val:.0f}",
                x=MAIN_L - 10,
                y=sy,
                anchor_x="right",
                anchor_y="center",
                font_size=9,
                color=gray,
                batch=batch,
            )
        )

    labels += [
        pyglet.text.Label(
            "time (t)",
            x=(MAIN_L + MAIN_R) // 2,
            y=14,
            anchor_x="center",
            anchor_y="bottom",
            font_size=11,
            color=gray,
            batch=batch,
        ),
        pyglet.text.Label(
            "Xt",
            x=14,
            y=(PLOT_B + PLOT_T) // 2,
            anchor_x="center",
            anchor_y="center",
            font_size=11,
            color=gray,
            batch=batch,
        ),
        pyglet.text.Label(
            "Trajectories",
            x=(MAIN_L + MAIN_R) // 2,
            y=H - 18,
            anchor_x="center",
            anchor_y="top",
            font_size=13,
            color=(200, 200, 200, 255),
            batch=batch,
        ),
    ]
    return shapes, labels


def build_hist_axes(x_min, x_max, batch):
    gray = (160, 160, 160, 255)
    dim = (55, 55, 55, 255)
    shapes, labels = [], []

    def line(p0, p1, color):
        shapes.append(pyglet.shapes.Line(*p0, *p1, color=color, batch=batch))

    for p0, p1 in [
        ((HIST_L, PLOT_B), (HIST_R, PLOT_B)),
        ((HIST_R, PLOT_B), (HIST_R, PLOT_T)),
        ((HIST_R, PLOT_T), (HIST_L, PLOT_T)),
        ((HIST_L, PLOT_T), (HIST_L, PLOT_B)),
    ]:
        line(p0, p1, gray)

    x_tick_step = max(1.0, round((x_max - x_min) / 8))
    for x_val in np.arange(
        np.ceil(x_min / x_tick_step) * x_tick_step, x_max, x_tick_step
    ):
        sy = x_to_sy(x_val, x_min, x_max)
        line((HIST_L, sy), (HIST_R, sy), dim)

    labels += [
        pyglet.text.Label(
            "density",
            x=(HIST_L + HIST_R) // 2,
            y=14,
            anchor_x="center",
            anchor_y="bottom",
            font_size=11,
            color=gray,
            batch=batch,
        ),
        pyglet.text.Label(
            "Terminal Distribution X_T",
            x=(HIST_L + HIST_R) // 2,
            y=H - 18,
            anchor_x="center",
            anchor_y="top",
            font_size=13,
            color=(200, 200, 200, 255),
            batch=batch,
        ),
    ]
    return shapes, labels


class BrownianSimulation:
    def __init__(self, sigma, n_traj, T, steps):
        self.sigma = sigma
        self.n_traj = n_traj
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bm = BrownianMotion(sigma)
        sim = EulerMaruyamaSimulator(sde=bm)
        x0 = torch.zeros(n_traj, 1).to(device)
        ts = torch.linspace(0, T, steps).to(device)

        trajs = sim.simulate_with_trajectory(x0, ts)
        self.trajs = trajs[:, :, 0].cpu().numpy()
        self.ts = ts.cpu().numpy()

        self._calculate_bounds()
        self._calculate_theoretical_pdf()

    def _calculate_bounds(self):
        margin = 0.5
        self.x_min = self.trajs.min() - margin
        self.x_max = self.trajs.max() + margin
        self.t_min = float(self.ts.min())
        self.t_max = float(self.ts.max())

    def _calculate_theoretical_pdf(self):
        self.std = self.sigma * np.sqrt(self.ts[-1])
        self.max_pdf = 1 / (self.std * np.sqrt(2 * np.pi))
        self.density_scale = (HIST_R - HIST_L) * 0.88 / self.max_pdf


class BMVisualizer:
    def __init__(self, sim: BrownianSimulation):
        self.sim = sim
        self.window = pyglet.window.Window(
            width=W, height=H, caption=f"Brownian Motion  sigma = {sim.sigma}"
        )
        pyglet.gl.glClearColor(0.08, 0.08, 0.1, 1.0)

        self.traj_batch = pyglet.graphics.Batch()
        self.hist_batch = pyglet.graphics.Batch()
        self.axis_batch = pyglet.graphics.Batch()

        self.axis_shapes, self.axis_labels = build_main_axes(
            sim.t_min, sim.t_max, sim.x_min, sim.x_max, self.axis_batch
        )
        hist_shapes, hist_labels = build_hist_axes(
            sim.x_min, sim.x_max, self.axis_batch
        )
        self.axis_shapes += hist_shapes
        self.axis_labels += hist_labels

        self._build_histogram_bins()
        self.gauss_lines = self._build_theoretical_pdf()

        self.shown = 0
        self.terminal_vals = []
        self.traj_segments = []
        self.gif_frames = []
        self.capture_next = False

        pyglet.clock.schedule_interval(self.add_next_trajectory, TRAJ_INTERVAL)
        self.window.event(self.on_draw)

    def _build_histogram_bins(self):
        self.bin_edges = np.linspace(self.sim.x_min, self.sim.x_max, N_BINS + 1)
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]
        self.bin_rects = []
        for i in range(N_BINS):
            sy_bot = x_to_sy(self.bin_edges[i], self.sim.x_min, self.sim.x_max)
            sy_top = x_to_sy(self.bin_edges[i + 1], self.sim.x_min, self.sim.x_max)
            self.bin_rects.append(
                pyglet.shapes.Rectangle(
                    x=HIST_L,
                    y=sy_bot,
                    width=1,
                    height=max(1, sy_top - sy_bot),
                    color=(210, 120, 50, 190),
                    batch=self.hist_batch,
                )
            )

    def _build_theoretical_pdf(self):
        x_curve = np.linspace(self.sim.x_min, self.sim.x_max, 300)
        pdf = np.exp(-0.5 * (x_curve / self.sim.std) ** 2) / (
            self.sim.std * np.sqrt(2 * np.pi)
        )
        pts = [
            (
                HIST_L + p * self.sim.density_scale,
                x_to_sy(xv, self.sim.x_min, self.sim.x_max),
            )
            for xv, p in zip(x_curve, pdf)
        ]
        return [
            pyglet.shapes.Line(
                *pts[i], *pts[i + 1], color=(240, 70, 70, 230), batch=self.axis_batch
            )
            for i in range(len(pts) - 1)
        ]

    def _update_hist(self):
        counts, _ = np.histogram(self.terminal_vals, bins=self.bin_edges)
        densities = counts / (len(self.terminal_vals) * self.bin_width)
        for i, rect in enumerate(self.bin_rects):
            rect.width = max(
                1.0, min(densities[i] * self.sim.density_scale, HIST_R - HIST_L)
            )

    def add_next_trajectory(self, dt):
        if self.shown >= self.sim.n_traj:
            pyglet.clock.unschedule(self.add_next_trajectory)
            return
        i = self.shown
        color = traj_color(i, self.sim.n_traj)
        pts = [
            main_to_screen(
                self.sim.ts[j],
                self.sim.trajs[i, j],
                self.sim.t_min,
                self.sim.t_max,
                self.sim.x_min,
                self.sim.x_max,
            )
            for j in range(len(self.sim.ts))
        ]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            self.traj_segments.append(
                pyglet.shapes.Line(x0, y0, x1, y1, color=color, batch=self.traj_batch)
            )
        self.terminal_vals.append(float(self.sim.trajs[i, -1]))
        self._update_hist()
        self.shown += 1
        self.capture_next = True

    def _capture_frame(self):
        buf = pyglet.image.get_buffer_manager().get_color_buffer()
        raw = buf.get_image_data().get_data("RGB", W * 3)

        img = Image.frombytes("RGB", (W, H), raw).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )
        if GIF_SCALE != 1.0:
            img = img.resize((int(W * GIF_SCALE), int(H * GIF_SCALE)))
        return img

    def _save_gif(self):
        print(f"Encoding {len(self.gif_frames)} frames → {GIF_PATH} …")
        self.gif_frames[0].save(
            GIF_PATH,
            save_all=True,
            append_images=self.gif_frames[1:],
            duration=GIF_DURATION,
            loop=0,
            optimize=False,
        )
        print(f"Saved {GIF_PATH}")

    def on_draw(self):
        self.window.clear()
        self.traj_batch.draw()
        self.hist_batch.draw()
        self.axis_batch.draw()

        # if self.capture_next:
        #     self.gif_frames.append(self._capture_frame())
        #     self.capture_next = False
        #     if self.shown >= self.sim.n_traj:
        #         self._save_gif()

    def run(self):
        pyglet.app.run()


def main():
    sim = BrownianSimulation(sigma=1, n_traj=500, T=5.0, steps=300)
    vis = BMVisualizer(sim)
    vis.run()


if __name__ == "__main__":
    main()
