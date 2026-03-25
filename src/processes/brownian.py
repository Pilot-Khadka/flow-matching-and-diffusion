import torch

from equations.sde import SDE


class BrownianMotion(SDE):
    """
    Brownian motion is recovered (by definition) by setting $u_t = 0$ and $\sigma_t = \sigma$,
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        description:
            Returns the drift coefficient of the ODE.
        outputs:
            drift: shape (bs, dim)
        """
        return torch.zeros_like(xt)

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
