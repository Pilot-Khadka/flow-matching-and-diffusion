from abc import ABC, abstractmethod

import torch
from torch.func import vmap, jacrev


class Density(ABC):
    """
    Distribution with tractable density
    """

    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        description:
            Returns the log density at x.
        inputs:
             x: shape (batch_size, dim)
        outputs:
            log_density: shape (batch_size, 1)
        """
        pass

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        description:
            Returns the score dx log density(x)
        inputs:
            x: (batch_size, dim)
        outputs:
            score: (batch_size, dim)
        """
        x = x.unsqueeze(1)  # (batch_size, 1, ...)
        score = vmap(jacrev(self.log_density))(x)  # (batch_size, 1, 1, 1, ...)
        return score.squeeze((1, 2, 3))  # (batch_size, ...)


class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        description:
            Returns the log density at x.
        inputs:
            num_samples: the desired number of samples
        outputs:
            samples: shape (batch_size, dim)
        """
        pass
