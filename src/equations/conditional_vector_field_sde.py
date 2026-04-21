import torch

from .sde import SDE
from distributions.base import ConditionalProbabilityPath


class ConditionalVectorFieldSDE(SDE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor, sigma: float):
        """
        inputs:
            path: the ConditionalProbabilityPath object to which this vector field corresponds
            z: the conditioning variable, (1, ...)
        """
        super().__init__()
        self.path = path
        self.z = z
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        outputs the conditional vector field u_t(x|z)
        inputs:
            x: state at time t, shape (bs, dim)
            t: time, shape (bs,.)
        outputs:
            u_t(x|z): shape (batch_size, dim)
        """
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(
            x, z, t
        ) + 0.5 * self.sigma**2 * self.path.conditional_score(x, z, t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        inputs:
            x: state at time t, shape (bs, dim)
            t: time, shape (bs,.)
        outputs:
            u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma
