from abc import ABC, abstractmethod

from torch import Tensor


class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        """
        description:
            returns drift coeff of ODE
        inputs:
            xt: state at time t, shape (bs, dim)
            t: time, shape ()
        outputs:
            drift_coefficient: shape (bs, dim)
        """
        pass
