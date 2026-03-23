from typing import Optional

import torch
import seaborn as sns
import matplotlib.pyplot as plt

from processes.brownian import BrownianMotion
from simulator.euler_maruyama import EulerMaruyamaSimulator
from utils.plot import plot_trajectories_1d


def main():
    sigma = 1.0
    n_traj = 500
    brownian_motion = BrownianMotion(sigma)
    simulator = EulerMaruyamaSimulator(sde=brownian_motion)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    x0 = torch.zeros(n_traj, 1).to(device)
    ts = torch.linspace(0.0, 5.0, 500).to(device)

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.set_title(
        r"Trajectories of Brownian Motion with $\sigma=$" + str(sigma), fontsize=18
    )
    ax.set_xlabel(r"time ($t$)", fontsize=18)
    ax.set_ylabel(r"$x_t$", fontsize=18)
    plot_trajectories_1d(x0, simulator, ts, ax, show_hist=True)
    plt.show()


if __name__ == "__main__":
    main()
