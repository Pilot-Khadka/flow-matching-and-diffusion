import torch

from equations.sde import SDE


class OUProcess(SDE):
    def __init__(self, theta: float, sigma: float):
        self.theta = theta
        self.sigma = sigma

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
        return -self.theta * xt

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
