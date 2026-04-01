from dataclasses import dataclass

import torch
import matplotlib.pyplot as plt

from distributions.gaussian import GaussianMixture, Gaussian
from distributions.gaussian_conditional_prob_path import (
    GaussianConditionalProbabilityPath,
)
from distributions.base import LinearAlpha, SquareRootBeta
from utils.plot import imshow_density


@dataclass(frozen=True)
class Config:
    scale: float = 15.0
    target_scale: float = 10.0
    target_std: float = 1.0
    n_modes: int = 5
    n_timesteps: int = 7
    n_samples: int = 1000
    density_bins: int = 200


def build_distributions(cfg: Config, device: torch.device):
    p_simple = Gaussian.isotropic(dim=2, std=1.0).to(device)
    p_data = GaussianMixture.symmetric_2D(
        nmodes=cfg.n_modes, std=cfg.target_std, scale=cfg.target_scale
    ).to(device)
    path = GaussianConditionalProbabilityPath(
        p_data=p_data, alpha=LinearAlpha(), beta=SquareRootBeta()
    ).to(device)
    return p_simple, p_data, path


def plot_marginals(p_simple, p_data, scale: float, device: torch.device):
    shared = dict(bins=200, vmin=-10, alpha=0.25, device=device, scale=scale)
    imshow_density(density=p_simple, cmap=plt.get_cmap("Reds"), **shared)
    imshow_density(density=p_data, cmap=plt.get_cmap("Blues"), **shared)


def plot_conditional_path(path, z: torch.Tensor, ts: torch.Tensor, n_samples: int):
    plt.scatter(z[:, 0].cpu(), z[:, 1].cpu(), marker="*", color="red", s=75, label="z")

    for t in ts:
        zz = z.expand(n_samples, -1)
        tt = t.expand(n_samples, 1)
        samples = path.sample_conditional_path(zz, tt)
        plt.scatter(
            samples[:, 0].cpu(),
            samples[:, 1].cpu(),
            alpha=0.25,
            s=8,
            label=f"t={t.item():.1f}",
        )


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_simple, p_data, path = build_distributions(cfg, device)

    z = path.sample_conditioning_variable(1).to(device)
    ts = torch.linspace(0.0, 1.0, cfg.n_timesteps, device=device).unsqueeze(-1)

    bounds = [-cfg.scale, cfg.scale]
    _, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=bounds, ylim=bounds, title="Gaussian Conditional Probability Path")
    ax.set_xticks([])
    ax.set_yticks([])

    plot_marginals(p_simple, p_data, cfg.scale, device)
    plot_conditional_path(path, z, ts, cfg.n_samples)

    plt.legend(prop={"size": 18}, markerscale=3)
    plt.show()


if __name__ == "__main__":
    main()
