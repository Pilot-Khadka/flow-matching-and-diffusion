import torch
import numpy as np
import torch.distributions as D

from .base import Sampleable, Density


class Gaussian(torch.nn.Module, Sampleable, Density):
    """Two-dimensional Gaussian.

    Is a Density and a Sampleable. Wrapper around
    torch.distributions.MultivariateNormal
    """

    def __init__(self, mean, cov):
        """
        Mean: shape (2,).

        cov: shape (2,2)
        """
        super().__init__()
        self.mean: torch.Tensor
        self.cov: torch.Tensor
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std**2
        return cls(mean, cov)


class GaussianMixture(torch.nn.Module, Sampleable, Density):
    """Two-dimensional Gaussian mixture model, and is a Density and a.

    Sampleable.

    Wrapper around torch.distributions.MixtureSameFamily.
    """

    def __init__(
        self,
        means: torch.Tensor,  # nmodes x data_dim
        covs: torch.Tensor,  # nmodes x data_dim x data_dim
        weights: torch.Tensor,  # nmodes
    ):
        """
        Means: shape (nmodes, 2).

        covs: shape (nmodes, 2, 2)
        weights: shape (nmodes, 1)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.means: torch.Tensor
        self.covs: torch.Tensor
        self.weights: torch.Tensor
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=self.means,
                covariance_matrix=self.covs,
                validate_args=False,
            ),
            validate_args=False,
        )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_2D(
        cls,
        nmodes: int,
        std: float,
        scale: float = 10.0,
        seed=0.0,
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std**2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls,
        nmodes: int,
        std: float,
        scale: float = 10.0,
    ) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std**2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)
