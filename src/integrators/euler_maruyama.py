import torch

from .base import Simulator
from equations.sde import SDE


class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        drift = self.sde.drift_coefficient(xt=xt, t=t)
        diffusion = self.sde.diffusion_coefficient(xt=xt, t=t)
        noise = torch.randn_like(xt)

        return xt + dt * drift + diffusion * torch.sqrt(dt) * noise
