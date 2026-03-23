import torch


from .base import Simulator
from diff_eqn.ode import ODE


class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        drift = self.ode.drift_coefficient(xt, t)
        return xt + dt * drift
