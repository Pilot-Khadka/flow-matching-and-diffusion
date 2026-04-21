from dataclasses import dataclass

import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from equations.conditional_vector_field_ode import ConditionalVectorFieldODE
from plot_utils import (
    BaseConfig,
    build_distributions,
    build_time_grid,
    simulate,
    extract_marginals,
    configure_ax,
    plot_background,
    plot_z,
    plot_ode_samples,
    plot_trajectories,
)


@dataclass(frozen=True)
class Config(BaseConfig):
    pass


def plot_ground_truth_conditional_path(
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
        ax.scatter(
            samples[:, 0].detach().cpu(),
            samples[:, 1].detach().cpu(),
            marker="o",
            alpha=0.5,
            label=f"t={t_val.item():.2f}",
        )


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_simple, p_data, path = build_distributions(cfg, device)

    torch.cuda.manual_seed(1)
    z = path.sample_conditioning_variable(1).to(device)

    x0 = path.p_simple.sample(cfg.n_samples)
    ts = build_time_grid(cfg, device)

    xts = simulate(ConditionalVectorFieldODE(path, z), x0, ts)
    xts_every_n, ts_every_n = extract_marginals(xts, ts, cfg)

    bounds = [-cfg.scale, cfg.scale]
    legend_kwargs = dict(
        prop={"size": cfg.legend_size}, loc="upper right", markerscale=cfg.markerscale
    )

    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    titles = [
        "Ground-Truth Conditional Probability Path",
        "Samples from Conditional ODE",
        "Trajectories of Conditional ODE",
    ]
    for ax, title in zip(axes, titles):
        configure_ax(ax, title, bounds)
        plot_background(ax, p_simple, p_data, cfg, device)

    plot_ground_truth_conditional_path(axes[0], path, z, ts_every_n, cfg)
    plot_z(axes[0], z)
    axes[0].legend(**legend_kwargs)

    plot_ode_samples(axes[1], xts_every_n, ts_every_n)
    plot_z(axes[1], z)
    axes[1].legend(**legend_kwargs)

    plot_trajectories(axes[2], xts, cfg.n_trajectories)
    plot_z(axes[2], z)
    axes[2].legend(**legend_kwargs)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
