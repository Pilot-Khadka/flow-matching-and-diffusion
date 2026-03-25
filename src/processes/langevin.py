import torch

from equations.sde import SDE
from distributions.base import Density


class LangevinSDE(SDE):
    def __init__(self, sigma: float, density: Density):
        self.sigma = sigma
        self.density = density

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        description:
            Returns the drift coefficient of the ODE.
        inputs:
            xt: state at time t, shape (bs, dim)
            t: time, shape ()
        outputs:
            drift: shape (bs, dim)
        """
        score = self.density.score(xt)
        return 0.5 * (self.sigma**2) * score

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        description:
            Returns the diffusion coefficient of the ODE.
        inputs:
            xt: state at time t, shape (bs, dim)
            t: time, shape ()
        outputs:
            diffusion: shape (bs, dim)
        """
        return torch.full_like(xt, self.sigma)
