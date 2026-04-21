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
from equations.learned_vector_field_ode import LearnedVectorFieldODE
from integrators.euler import EulerSimulator
from vector_field.trainer import ConditionalFlowMatchingTrainer
from vector_field.mlp import MLPVectorField
from utils import record_every


@dataclass(frozen=True)
class BaseConfig:
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
    train_epochs: int = 5000
    train_lr: float = 1e-3
    train_batch_size: int = 1000


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
    ax.imshow(
        # pyrefly: ignore [bad-argument-type, missing-attribute]
        density.cpu(),
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        **kwargs,
    )


def build_distributions(cfg: BaseConfig, device: torch.device):
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


def build_time_grid(cfg: BaseConfig, device: torch.device) -> torch.Tensor:
    return (
        torch.linspace(0.0, 1.0, cfg.n_timesteps)
        .view(1, -1, 1)
        .expand(cfg.n_samples, -1, 1)
        .to(device)
    )


def configure_ax(ax: Axes, title: str, bounds: list[float]):
    ax.set_xlim(*bounds)
    ax.set_ylim(*bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=20)


def plot_background(ax: Axes, p_simple, p_data, cfg: BaseConfig, device: torch.device):
    bounds = [-cfg.scale, cfg.scale]
    shared = dict(
        x_bounds=bounds,
        y_bounds=bounds,
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


def plot_ground_truth_conditional_path(
    ax: Axes,
    path,
    z: torch.Tensor,
    ts_every_n: torch.Tensor,
    cfg: BaseConfig,
):
    for t_val in ts_every_n:
        tt = t_val.view(1, 1).expand(cfg.n_samples, 1)
        zz = z.expand(cfg.n_samples, -1)
        samples = path.sample_conditional_path(zz, tt)
        ax.scatter(
            samples[:, 0].detach().cpu(),
            samples[:, 1].detach().cpu(),
            marker="o",
            alpha=0.5,
            label=f"t={t_val.item():.2f}",
        )


def plot_ground_truth_marginal_path(
    ax: Axes,
    path,
    ts_every_n: torch.Tensor,
    cfg: BaseConfig,
):
    for t_val in ts_every_n:
        tt = t_val.view(1, 1).expand(cfg.n_samples, 1)
        samples = path.sample_marginal_path(tt)
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


def simulate(ode, x0: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    return EulerSimulator(ode).simulate_with_trajectory(x0, ts)


def extract_marginals(
    xts: torch.Tensor, ts: torch.Tensor, cfg: BaseConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    every_n = record_every(
        num_timesteps=cfg.n_timesteps,
        record_every=cfg.n_timesteps // cfg.n_marginals,
    )
    return xts[:, every_n, :], ts[0, every_n, 0]


def main():
    cfg = BaseConfig()
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
    )

    conditional_xts = simulate(ConditionalVectorFieldODE(path, z), x0, ts)
    conditional_xts_every_n, ts_every_n = extract_marginals(conditional_xts, ts, cfg)

    flow_model = MLPVectorField(dim=2, hiddens=[64, 64, 64, 64])
    ConditionalFlowMatchingTrainer(path, flow_model).train(
        num_epochs=cfg.train_epochs,
        device=device,
        lr=cfg.train_lr,
        batch_size=cfg.train_batch_size,
    )

    learned_xts = simulate(LearnedVectorFieldODE(flow_model), x0, ts)
    learned_xts_every_n, _ = extract_marginals(learned_xts, ts, cfg)

    bounds = [-cfg.scale, cfg.scale]
    legend_kwargs = dict(
        prop={"size": cfg.legend_size}, loc="upper right", markerscale=cfg.markerscale
    )

    fig, axes = plt.subplots(2, 3, figsize=(36, 24))

    row0_titles = [
        "Ground-Truth Conditional Probability Path",
        "Samples from Conditional ODE",
        "Trajectories of Conditional ODE",
    ]
    row1_titles = [
        "Ground-Truth Marginal Probability Path",
        "Samples from Learned Marginal ODE",
        "Trajectories of Learned Marginal ODE",
    ]

    for ax, title in zip(axes[0], row0_titles):
        configure_ax(ax, title, bounds)
        plot_background(ax, p_simple, p_data, cfg, device)

    for ax, title in zip(axes[1], row1_titles):
        configure_ax(ax, title, bounds)
        plot_background(ax, p_simple, p_data, cfg, device)

    plot_ground_truth_conditional_path(axes[0][0], path, z, ts_every_n, cfg)
    plot_z(axes[0][0], z)
    axes[0][0].legend(**legend_kwargs)

    plot_ode_samples(axes[0][1], conditional_xts_every_n, ts_every_n)
    plot_z(axes[0][1], z)
    axes[0][1].legend(**legend_kwargs)

    plot_trajectories(axes[0][2], conditional_xts, cfg.n_trajectories)
    plot_z(axes[0][2], z)
    axes[0][2].legend(**legend_kwargs)

    plot_ground_truth_marginal_path(axes[1][0], path, ts_every_n, cfg)
    axes[1][0].legend(**legend_kwargs)

    plot_ode_samples(axes[1][1], learned_xts_every_n, ts_every_n)
    axes[1][1].legend(**legend_kwargs)

    plot_trajectories(axes[1][2], learned_xts, cfg.n_trajectories)
    axes[1][2].legend(**legend_kwargs)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
