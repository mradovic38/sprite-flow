from abc import ABC, abstractmethod

import torch
from torch import nn

from helpers import model_size_b, MiB
from sampling.conditional_probability_path import GaussianConditionalProbabilityPath


class Trainer(ABC):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        # TODO: add AdamW and maybe some other optimizer variations
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> None:
        # Print model size
        size_b = model_size_b(self.model)
        print(f'Model size: {size_b / MiB:.4f} MiB')

        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Loop
        # TODO: add validation
        for idx, epoch in enumerate(range(num_epochs)):
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            print(f'Epoch {epoch}, loss {loss.item():.4f}')

        # Final eval
        self.model.eval()


class UnguidedTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: nn.Module) -> None:
        super().__init__(model)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Sample from p_data
        z = self.path.p_data.sample(batch_size)
        device = z.device

        # Sample t and x
        t = torch.rand(batch_size, 1, 1, 1, device=device)
        x = self.path.sample_conditional_path(z, t)

        # Regress and output loss
        ut_theta = self.model(x, t)  # (batch_size, 3, 128, 128)
        ut_ref = self.path.conditional_vector_field(x, z, t)  # (batch_size, 3, 128, 128)
        return torch.mean(torch.sum(torch.square(ut_theta - ut_ref), dim=-1))