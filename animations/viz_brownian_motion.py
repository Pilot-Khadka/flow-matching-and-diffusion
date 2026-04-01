import numpy as np
import torch

# pyrefly: ignore [missing-import]
from utils.viz_sde import SimulationData, SDEVisualizer
from processes.brownian import BrownianMotion
from integrators.euler_maruyama import EulerMaruyamaSimulator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma, T = 1.0, 5.0

    bm = BrownianMotion(sigma)
    sim = EulerMaruyamaSimulator(sde=bm)
    x0 = torch.zeros(500, 1).to(device)
    ts = torch.linspace(0.0, T, 300).to(device)

    std = sigma * np.sqrt(T)
    sim_data = SimulationData.from_sde(
        bm,
        sim,
        x0,
        ts,
        title=f"Brownian Motion  σ={sigma}",
        theoretical_pdf=lambda x: (
            np.exp(-0.5 * (x / std) ** 2) / (std * np.sqrt(2 * np.pi))
        ),
    )

    SDEVisualizer([sim_data]).run()


if __name__ == "__main__":
    main()
