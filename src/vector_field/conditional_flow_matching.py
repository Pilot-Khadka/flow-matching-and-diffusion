import torch

from .trainer import Trainer
from .mlp import MLPVectorField
from distributions.gaussian_conditional_prob_path import ConditionalProbabilityPath


class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model: MLPVectorField,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size, 1).to(z.device)
        x = self.path.sample_conditional_path(z=z, t=t)
        u_theta = self.model(x, t)
        u_ref = self.path.conditional_vector_field(x=x, z=z, t=t)
        loss = torch.mean((u_theta - u_ref) ** 2)
        return loss
