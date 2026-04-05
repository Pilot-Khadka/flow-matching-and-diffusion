from dataclasses import dataclass
import torch

from vector_field.conditional_flow_matching import (
    ConditionalFlowMatchingTrainer,
    MLPVectorField,
)
from distributions.gaussian_conditional_prob_path import (
    GaussianConditionalProbabilityPath,
)
from distributions.gaussian import GaussianMixture
from distributions.base import LinearAlpha, SquareRootBeta


@dataclass(frozen=True)
class Config:
    scale: float = 15.0
    target_scale: float = 10.0
    target_std: float = 1.0


def main():
    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = GaussianConditionalProbabilityPath(
        p_data=GaussianMixture.symmetric_2D(
            nmodes=5, std=config.target_std, scale=config.target_scale
        ).to(device),
        alpha=LinearAlpha(),
        beta=SquareRootBeta(),
    ).to(device)

    # Construct learnable vector field
    flow_model = MLPVectorField(dim=2, hiddens=[64, 64, 64, 64])

    # Construct trainer
    trainer = ConditionalFlowMatchingTrainer(path, flow_model)
    losses = trainer.train(num_epochs=5000, device=device, lr=1e-3, batch_size=1000)
    # tui = TrainingTUI(trainer, num_epochs=5000, device=device, lr=1e-3, batch_size=1000)
    # tui.run()


if __name__ == "__main__":
    main()
