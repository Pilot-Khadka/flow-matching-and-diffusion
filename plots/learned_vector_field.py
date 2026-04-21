from dataclasses import dataclass

import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from equations.learned_vector_field_ode import LearnedVectorFieldODE
from vector_field.trainer import ConditionalFlowMatchingTrainer
from vector_field.mlp import MLPVectorField
from plot_utils import (
    BaseConfig,
    build_distributions,
    build_time_grid,
    simulate,
    extract_marginals,
    configure_ax,
    plot_background,
    plot_ode_samples,
    plot_trajectories,
)


@dataclass(frozen=True)
class Config(BaseConfig):
    train_epochs: int = 5000
    train_lr: float = 1e-3
    train_batch_size: int = 1000


def plot_ground_truth_marginal_path(
    ax: Axes,
    path,
    ts_every_n: torch.Tensor,
    cfg: Config,
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


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_simple, p_data, path = build_distributions(cfg, device)

    flow_model = MLPVectorField(dim=2, hiddens=[64, 64, 64, 64])
    ConditionalFlowMatchingTrainer(path, flow_model).train(
        num_epochs=cfg.train_epochs,
        device=device,
        lr=cfg.train_lr,
        batch_size=cfg.train_batch_size,
    )

    x0 = path.p_simple.sample(cfg.n_samples)
    ts = build_time_grid(cfg, device)

    xts = simulate(LearnedVectorFieldODE(flow_model), x0, ts)
    xts_every_n, ts_every_n = extract_marginals(xts, ts, cfg)

    bounds = [-cfg.scale, cfg.scale]
    legend_kwargs = dict(
        prop={"size": cfg.legend_size}, loc="upper right", markerscale=cfg.markerscale
    )

    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    titles = [
        "Ground-Truth Marginal Probability Path",
        "Samples from Learned Marginal ODE",
        "Trajectories of Learned Marginal ODE",
    ]
    for ax, title in zip(axes, titles):
        configure_ax(ax, title, bounds)
        plot_background(ax, p_simple, p_data, cfg, device)

    plot_ground_truth_marginal_path(axes[0], path, ts_every_n, cfg)
    axes[0].legend(**legend_kwargs)

    plot_ode_samples(axes[1], xts_every_n, ts_every_n)
    axes[1].legend(**legend_kwargs)

    plot_trajectories(axes[2], xts, cfg.n_trajectories)
    axes[2].legend(**legend_kwargs)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
