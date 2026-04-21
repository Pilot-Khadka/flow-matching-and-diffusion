from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from .mlp import MLPVectorField
from distributions.gaussian_conditional_prob_path import ConditionalProbabilityPath


class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs
    ) -> torch.Tensor:
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f"Epoch {idx}, loss: {loss.item()}")

        self.model.eval()


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
