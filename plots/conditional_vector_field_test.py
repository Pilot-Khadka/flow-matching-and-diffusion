from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from distributions.gaussian_conditional_prob_path import (
    GaussianConditionalProbabilityPath,
)
from distributions.gaussian import GaussianMixture, Gaussian
from distributions.base import LinearAlpha, SquareRootBeta


from distributions.base import Density
from equations.conditional_vector_field_ode import ConditionalVectorFieldODE
from integrators.euler import EulerSimulator
from utils import record_every


@dataclass(frozen=True)
class Config:
    scale: float = 15.0
    target_scale: float = 10.0
    target_std: float = 1.0
    n_modes: int = 5
    n_samples: int = 1000
    n_timesteps: int = 1000
    n_marginals: int = 3
    n_trajectories: int = 15
    density_bins: int = 200
    legend_size: int = 24
    markerscale: float = 1.8


def imshow_density(
    density: Density,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    bins: int,
    ax: Optional[Axes] = None,
    x_offset: float = 0.0,
    device=torch.device,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    # pyrefly: ignore [no-matching-overload]
    x = torch.linspace(x_min, x_max, bins).to(device) + x_offset
    # pyrefly: ignore [no-matching-overload]
    y = torch.linspace(y_min, y_max, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    # pyrefly: ignore [bad-assignment]
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.imshow(
        # pyrefly: ignore [bad-argument-type, missing-attribute]
        density.cpu(), extent=[x_min, x_max, y_min, y_max], origin="lower", **kwargs
    )


def build_distributions(cfg: Config, device: torch.device):
    p_simple = Gaussian.isotropic(dim=2, std=1.0).to(device)
    p_data = GaussianMixture.symmetric_2D(
        nmodes=cfg.n_modes, std=cfg.target_std, scale=cfg.target_scale
    ).to(device)
    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=LinearAlpha(),
        beta=SquareRootBeta(),
    ).to(device)
    return p_simple, p_data, path


def configure_ax(ax: Axes, title: str, bounds: list[float]):
    ax.set_xlim(*bounds)
    ax.set_ylim(*bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=20)


def plot_background(ax: Axes, p_simple, p_data, cfg: Config, device: torch.device):
    bounds = [-cfg.scale, cfg.scale]
    x_bounds = bounds
    y_bounds = bounds

    shared = dict(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bins=cfg.density_bins,
        ax=ax,
        vmin=-10,
        alpha=0.25,
        device=device,
    )

    # pyrefly: ignore [bad-argument-type]
    imshow_density(density=p_simple, cmap=plt.get_cmap("Reds"), **shared)
    # pyrefly: ignore [bad-argument-type]
    imshow_density(density=p_data, cmap=plt.get_cmap("Blues"), **shared)


def plot_z(ax: Axes, z: torch.Tensor):
    ax.scatter(
        z[:, 0].cpu(),
        z[:, 1].cpu(),
        marker="*",
        color="red",
        s=200,
        label="z",
        zorder=20,
    )


def plot_ground_truth_path(
    ax: Axes,
    path,
    z: torch.Tensor,
    ts_every_n: torch.Tensor,
    cfg: Config,
):
    for t_val in ts_every_n:
        tt = t_val.view(1, 1).expand(cfg.n_samples, 1)
        zz = z.expand(cfg.n_samples, -1)
        samples = path.sample_conditional_path(zz, tt)
        print("samples:", samples.shape)
        ax.scatter(
            samples[:, 0].detach().cpu(),
            samples[:, 1].detach().cpu(),
            marker="o",
            alpha=0.5,
            label=f"t={t_val.item():.2f}",
        )


def plot_ode_samples(
    ax: Axes,
    xts_every_n: torch.Tensor,
    ts_every_n: torch.Tensor,
):

    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].item()
        samples = xts_every_n[:, plot_idx]
        print("ode samples:", samples.shape)
        ax.scatter(
            samples[:, 0].detach().cpu(),
            samples[:, 1].detach().cpu(),
            marker="o",
            alpha=0.5,
            label=f"t={tt:.2f}",
        )


def plot_trajectories(ax: Axes, xts: torch.Tensor, n_trajectories: int):
    for i in range(n_trajectories):
        ax.plot(
            xts[i, :, 0].detach().cpu(),
            xts[i, :, 1].detach().cpu(),
            alpha=0.5,
            color="black",
        )


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_simple, p_data, path = build_distributions(cfg, device)

    torch.cuda.manual_seed(1)
    z = path.sample_conditioning_variable(1).to(device)  # (1, dim)

    x0 = path.p_simple.sample(cfg.n_samples)
    ts = (
        torch.linspace(0.0, 1.0, cfg.n_timesteps)
        .view(1, -1, 1)
        .expand(cfg.n_samples, -1, 1)
        .to(device)
    )  # (n_samples, n_timesteps, 1)

    xts = EulerSimulator(ConditionalVectorFieldODE(path, z)).simulate_with_trajectory(
        x0, ts
    )
    print("xts shape:", xts.shape)

    every_n = record_every(
        num_timesteps=cfg.n_timesteps,
        record_every=cfg.n_timesteps // cfg.n_marginals,
    )
    xts_every_n = xts[:, every_n, :]  # (n_samples, n_marginals, dim)
    ts_every_n = ts[0, every_n, 0]  # (n_marginals,)

    bounds = [-cfg.scale, cfg.scale]
    legend_kwargs = dict(
        prop={"size": cfg.legend_size}, loc="upper right", markerscale=cfg.markerscale
    )
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))

    ax = axes[0]
    configure_ax(ax, "Ground-Truth Conditional Probability Path", bounds)
    plot_background(ax, p_simple, p_data, cfg, device)
    plot_ground_truth_path(ax, path, z, ts_every_n, cfg)
    plot_z(ax, z)
    ax.legend(**legend_kwargs)

    ax = axes[1]
    configure_ax(ax, "Samples from Conditional ODE", bounds)
    plot_background(ax, p_simple, p_data, cfg, device)
    plot_ode_samples(ax, xts_every_n, ts_every_n)
    plot_z(ax, z)
    ax.legend(**legend_kwargs)

    ax = axes[2]
    configure_ax(ax, "Trajectories of Conditional ODE", bounds)
    plot_background(ax, p_simple, p_data, cfg, device)
    plot_trajectories(ax, xts, cfg.n_trajectories)
    plot_z(ax, z)
    ax.legend(**legend_kwargs)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
