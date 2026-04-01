import torch


from .base import Simulator
from equations.ode import ODE


class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    # pyrefly: ignore [bad-override]
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        drift = self.ode.drift_coefficient(xt, t)
        return xt + dt * drift
