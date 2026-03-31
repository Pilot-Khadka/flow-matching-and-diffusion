import torch
import seaborn as sns
import matplotlib.pyplot as plt


from distributions.base import Density, Sampleable
from integrators.base import Simulator
from utils.plot import imshow_density
from distributions.gaussian import Gaussian, GaussianMixture
from processes.langevin import LangevinSDE
from integrators.euler_maruyama import EulerMaruyamaSimulator


def every_nth_index(num_timesteps: int, n: int) -> torch.Tensor:
    """Compute the indices to record in the trajectory given a record_every.

    parameter.
    """
    if n == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, n),
            torch.tensor([num_timesteps - 1]),
        ]
    )


def graph_dynamics(
    num_samples: int,
    source_distribution: Sampleable,
    simulator: Simulator,
    density: Density,
    timesteps: torch.Tensor,
    plot_every: int,
    bins: int,
    scale: float,
    device: torch.device,
):
    """Plot the evolution of samples from source under the simulation scheme.

    given by simulator (itself a discretization of an ODE or SDE).

    num_samples: the number of samples to simulate
    source_distribution: distribution from which we draw initial samples at t=0
    simulator: the discertized simulation scheme used to simulate the dynamics
    density: the target density
    timesteps: the timesteps used by the simulator
    plot_every: number of timesteps between consecutive plots
    bins: number of bins for imshow
    scale: scale for imshow
    """
    # Simulate
    x0 = source_distribution.sample(num_samples)
    xts = simulator.simulate_with_trajectory(x0, timesteps)
    indices_to_plot = every_nth_index(len(timesteps), plot_every)
    plot_timesteps = timesteps[indices_to_plot]
    plot_xts = xts[:, indices_to_plot]

    # Graph
    fig, axes = plt.subplots(
        2, len(plot_timesteps), figsize=(8 * len(plot_timesteps), 16)
    )
    axes = axes.reshape((2, len(plot_timesteps)))
    for t_idx in range(len(plot_timesteps)):
        t = plot_timesteps[t_idx].item()
        xt = plot_xts[:, t_idx]
        # Scatter axes
        scatter_ax = axes[0, t_idx]
        imshow_density(
            density=density,
            bins=bins,
            scale=scale,
            device=device,
            ax=scatter_ax,
            vmin=-15,
            alpha=0.25,
            cmap=plt.get_cmap("Blues"),
        )
        scatter_ax.scatter(
            xt[:, 0].cpu(),
            xt[:, 1].cpu(),
            marker="x",
            color="black",
            alpha=0.75,
            s=15,
        )
        scatter_ax.set_title(f"Samples at t={t:.1f}", fontsize=15)
        scatter_ax.set_xticks([])
        scatter_ax.set_yticks([])

        # Kdeplot axes
        kdeplot_ax = axes[1, t_idx]
        imshow_density(
            density=density,
            bins=bins,
            scale=scale,
            ax=kdeplot_ax,
            vmin=-15,
            alpha=0.5,
            cmap=plt.get_cmap("Blues"),
            device=device,
        )
        sns.kdeplot(
            x=xt[:, 0].cpu().numpy(),
            y=xt[:, 1].cpu().numpy(),
            alpha=0.5,
            ax=kdeplot_ax,
            color="grey",
        )
        kdeplot_ax.set_title(f"Density of Samples at t={t:.1f}", fontsize=15)
        kdeplot_ax.set_xticks([])
        kdeplot_ax.set_yticks([])
        kdeplot_ax.set_xlabel("")
        kdeplot_ax.set_ylabel("")

    plt.show()


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    target = GaussianMixture.random_2D(nmodes=5, std=0.75, scale=15.0, seed=3.0).to(
        device
    )
    sde = LangevinSDE(sigma=0.6, density=target)
    simulator = EulerMaruyamaSimulator(sde)

    # Graph the results!
    graph_dynamics(
        num_samples=1000,
        source_distribution=Gaussian(mean=torch.zeros(2), cov=20 * torch.eye(2)).to(
            device
        ),
        simulator=simulator,
        density=target,
        timesteps=torch.linspace(0, 5.0, 1000).to(device),
        plot_every=334,
        bins=200,
        scale=15,
        device=device,
    )


if __name__ == "__main__":
    main()
