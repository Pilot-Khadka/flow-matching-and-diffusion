import torch


from .gaussian import Gaussian
from .base import Alpha, Beta, Sampleable, ConditionalProbabilityPath


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, alpha: Alpha, beta: Beta):
        p_simple = Gaussian.isotropic(p_data.dim, 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Description:

            Samples the conditioning variable z ~ p_data(x)
        inputs:
            num_samples: the number of samples
        outputs:
            z: samples from p(z), (num_samples, dim)
        """
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Description:

            Samples from the conditional distribution p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        inputs:
            z: conditioning variable (num_samples, dim)
            t: time (num_samples, 1)
        outputs:
            x: samples from p_t(x|z), (num_samples, dim)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        eps = torch.randn_like(z)
        return alpha_t * z + beta_t * eps

    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Description:

            Evaluates the conditional vector field u_t(x|z)
            Note: Only defined on t in [0,1)
        inputs:
            x: position variable (num_samples, dim)
            z: conditioning variable (num_samples, dim)
            t: time (num_samples, 1)
        outputs:
            conditional_vector_field: conditional vector field (num_samples, dim)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        alpha_dt = self.alpha.dt(t)
        beta_dt = self.beta.dt(t)
        mean = alpha_t * z
        return alpha_dt * z + (beta_dt / beta_t) * (x - mean)

    def conditional_score(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Description:

            Evaluates the conditional score of p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
            Note: Only defined on t in [0,1)
        inputs:
            x: position variable (num_samples, dim)
            z: conditioning variable (num_samples, dim)
            t: time (num_samples, 1)
        outputs:
            conditional_score: conditional score (num_samples, dim)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        mean = alpha_t * z

        return -(x - mean) / (beta_t**2)
