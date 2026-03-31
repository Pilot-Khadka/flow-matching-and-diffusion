from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.func import vmap, jacrev


class Density(ABC):
    """Distribution with tractable density."""

    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:

            outputs the log density at x.
        inputs:
             x: shape (batch_size, dim)
        outputs:
            log_density: shape (batch_size, 1)
        """
        pass

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:

            outputs the score dx log density(x)
        inputs:
            x: (batch_size, dim)
        outputs:
            score: (batch_size, dim)
        """
        x = x.unsqueeze(1)  # (batch_size, 1, ...)
        score = vmap(jacrev(self.log_density))(x)  # (batch_size, 1, 1, 1, ...)
        return score.squeeze((1, 2, 3))  # (batch_size, ...)


class Sampleable(ABC):
    """Distribution which can be sampled from."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Returns:

            - Dimensionality of the distribution
        """
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Description:

            outputs the log density at x.
        inputs:
            num_samples: the desired number of samples
        outputs:
            samples: shape (batch_size, dim)
        """
        pass


class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(self(torch.zeros(1, 1)), torch.zeros(1, 1))
        # Check alpha_1 = 1
        assert torch.allclose(self(torch.ones(1, 1)), torch.ones(1, 1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t.

        Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        inputs:
            t: time (num_samples, 1)
        outputs:
            alpha_t (num_samples, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.

        inputs:
            t: time (num_samples, 1)
        outputs:
            d/dt alpha_t (num_samples, 1)
        """
        t = t.unsqueeze(1)  # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t)  # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)


class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(self(torch.zeros(1, 1)), torch.ones(1, 1))
        # Check beta_1 = 0
        assert torch.allclose(self(torch.ones(1, 1)), torch.zeros(1, 1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t.

        Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        inputs:
            t: time (num_samples, 1)
        outputs:
            beta_t (num_samples, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.

        inputs:
            t: time (num_samples, 1)
        outputs:
            d/dt beta_t (num_samples, 1)
        """
        t = t.unsqueeze(1)  # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t)  # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)


class ConditionalProbabilityPath(nn.Module, ABC):
    """Abstract base class for conditional probability paths."""

    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Description:

            Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        inputs:
            t: time (num_samples, 1)
        outputs:
            x: samples from p_t(x), (num_samples, dim)
        """
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z = self.sample_conditioning_variable(num_samples)  # (num_samples, dim)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t)  # (num_samples, dim)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Description:

            Samples the conditioning variable z
        inputs:
            num_samples: the number of samples
        outputs:
            z: samples from p(z), (num_samples, dim)
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Description:

            Samples from the conditional distribution p_t(x|z)
        inputs:
            z: conditioning variable (num_samples, dim)
            t: time (num_samples, 1)
        outputs:
            x: samples from p_t(x|z), (num_samples, dim)
        """
        pass

    @abstractmethod
    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Description:

            Evaluates the conditional vector field u_t(x|z)
        inputs:
            x: position variable (num_samples, dim)
            z: conditioning variable (num_samples, dim)
            t: time (num_samples, 1)
        outputs:
            conditional_vector_field: conditional vector field (num_samples, dim)
        """
        pass

    @abstractmethod
    def conditional_score(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Description:

            Evaluates the conditional score of p_t(x|z)
        inputs:
            x: position variable (num_samples, dim)
            z: conditioning variable (num_samples, dim)
            t: time (num_samples, 1)
        outputs:
            conditional_score: conditional score (num_samples, dim)
        """
        pass


class LinearAlpha(Alpha):
    """Implements alpha_t = t."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Inputs:

            t: time (num_samples, 1)
        outputs:
            alpha_t (num_samples, 1)
        """
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.

        inputs:
            t: time (num_samples, 1)
        outputs:
            d/dt alpha_t (num_samples, 1)
        """
        return torch.ones_like(t)


class SquareRootBeta(Beta):
    """Implements beta_t = rt(1-t)."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Inputs:

            t: time (num_samples, 1)
        outputs:
            beta_t (num_samples, 1)
        """
        return torch.sqrt(1 - t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.

        inputs:
            t: time (num_samples, 1)
        outputs:
            d/dt alpha_t (num_samples, 1)
        """
        return -0.5 / (torch.sqrt(1 - t) + 1e-4)
