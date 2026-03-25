import numpy as np
import torch

from utils.viz_sde import SimulationData, SDEVisualizer
from processes.ornstein_uhlenbeck import OUProcess
from integrators.euler_maruyama import EulerMaruyamaSimulator


def ou_stationary_pdf(theta: float, sigma: float):
    std = sigma / np.sqrt(2 * theta)
    return lambda x: np.exp(-0.5 * (x / std) ** 2) / (std * np.sqrt(2 * np.pi))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        (0.25, 0.0),
        (0.25, 0.5),
        (0.25, 2.0),
    ]
    T, n_traj = 10.0, 500

    simulations = []
    for theta, sigma in configs:
        ou = OUProcess(theta, sigma)
        sim = EulerMaruyamaSimulator(sde=ou)
        x0 = torch.linspace(-10.0, 10.0, n_traj).view(-1, 1).to(device)
        ts = torch.linspace(0.0, T, 1000).to(device)

        pdf = ou_stationary_pdf(theta, sigma) if theta > 0 and sigma > 0 else None
        simulations.append(
            SimulationData.from_sde(
                ou,
                sim,
                x0,
                ts,
                title=f"OU Process  θ={theta},  σ={sigma}",
                theoretical_pdf=pdf,
            )
        )

    SDEVisualizer(simulations).run()


if __name__ == "__main__":
    main()
