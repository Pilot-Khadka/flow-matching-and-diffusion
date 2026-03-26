import torch

from distributions.gaussian import Gaussian, GaussianMixture
from processes.langevin import LangevinSDE
from integrators.euler_maruyama import EulerMaruyamaSimulator
from utils.animation_2d import SimulationData2D, SDEVisualizer2D


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    target = GaussianMixture.random_2D(nmodes=5, std=0.75, scale=15.0, seed=3.0).to(
        device
    )
    sde = LangevinSDE(sigma=0.6, density=target)
    simulator = EulerMaruyamaSimulator(sde)

    x0 = Gaussian(mean=torch.zeros(2), cov=20 * torch.eye(2)).to(device).sample(1000)
    ts = torch.linspace(0, 5.0, 500).to(device)

    sim_data = SimulationData2D.from_sde(
        sde=sde,
        simulator=simulator,
        x0=x0,
        ts=ts,
        title="Langevin Dynamics, Gaussian Mixture",
        density=target,
        scale=15.0,
        density_bins=150,
        device=device,
    )

    # set gif_path to None to skip saving
    SDEVisualizer2D(
        sim_data,
        steps_per_frame=2,
        gif_path="langevin.gif",  # set to None to skip saving
        gif_capture_every=3,
        gif_fps=30,
        gif_scale=0.5,
    ).run()

    # SDEVisualizer2D(
    #     sim_data,
    #     steps_per_frame=2,
    # ).run()


if __name__ == "__main__":
    main()
