import torch
import matplotlib.pyplot as plt

from distributions.gaussian import GaussianMixture, Gaussian
from distributions.gaussian_conditional_prob_path import (
    GaussianConditionalProbabilityPath,
)
from distributions.base import LinearAlpha, SquareRootBeta
from utils.plot import imshow_density

PARAMS = {
    "scale": 15.0,
    "target_scale": 10.0,
    "target_std": 1.0,
}


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    p_simple = Gaussian.isotropic(dim=2, std=1.0).to(device)
    p_data = GaussianMixture.symmetric_2D(
        nmodes=5, std=PARAMS["target_std"], scale=PARAMS["target_scale"]
    ).to(device)
    # Construct conditional probability path
    path = GaussianConditionalProbabilityPath(
        p_data=GaussianMixture.symmetric_2D(
            nmodes=5, std=PARAMS["target_std"], scale=PARAMS["target_scale"]
        ).to(device),
        alpha=LinearAlpha(),
        beta=SquareRootBeta(),
    ).to(device)

    scale = PARAMS["scale"]
    x_bounds = [-scale, scale]
    y_bounds = [-scale, scale]

    plt.figure(figsize=(10, 10))
    plt.xlim(*x_bounds)
    plt.ylim(*y_bounds)
    plt.title("Gaussian Conditional Probability Path")

    # Plot source and target
    imshow_density(
        density=p_simple,
        bins=200,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Reds"),
        device=device,
        scale=scale,
    )
    imshow_density(
        density=p_data,
        bins=200,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Blues"),
        device=device,
        scale=scale,
    )

    # Sample conditioning variable z
    z = path.sample_conditioning_variable(1)  # (1,2)
    ts = torch.linspace(0.0, 1.0, 7).to(device)

    # Plot z
    plt.scatter(z[:, 0].cpu(), z[:, 1].cpu(), marker="*", color="red", s=75, label="z")
    plt.xticks([])
    plt.yticks([])

    # Plot conditional probability path at each intermediate t
    num_samples = 1000
    for t in ts:
        zz = z.expand(num_samples, 2)
        tt = t.unsqueeze(0).expand(num_samples, 1)  # (samples, 1)
        samples = path.sample_conditional_path(zz, tt)  # (samples, 2)
        plt.scatter(
            samples[:, 0].cpu(),
            samples[:, 1].cpu(),
            alpha=0.25,
            s=8,
            label=f"t={t.item():.1f}",
        )

    plt.legend(prop={"size": 18}, markerscale=3)
    plt.show()


if __name__ == "__main__":
    main()
