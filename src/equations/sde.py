from abc import ABC, abstractmethod

from torch import Tensor


class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        """
        Returns the drift coefficient of the SDE.

        inputs:
            xt: state at time t, shape (batch_size, dim)
            t: time, shape ()
        outputs:
            drift_coefficient: shape (batch_size, dim)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        """
        Returns the diffusion coefficient of the SDE.

        inputs:
            xt: state at time t, shape (batch_size, dim)
            t: time, shape ()
        outputs:
            diffusion_coefficient: shape (batch_size, dim)
        """
        pass
