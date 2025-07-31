from abc import ABC, abstractmethod

import torch
from torch import nn
from tqdm import tqdm

from helpers import model_size_b, MiB
from sampling.conditional_probability_path import GaussianConditionalProbabilityPath


class Trainer(ABC):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def get_validation_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        # TODO: add AdamW and maybe some other optimizer variations
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, validate_every: int = 1, **kwargs) -> None:
        # Print model size
        size_b = model_size_b(self.model)
        print(f'Model size: {size_b / MiB:.4f} MiB')

        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Loop
        pbar = tqdm(enumerate(range(num_epochs)), total=num_epochs)
        for idx, epoch in pbar:
            opt.zero_grad()
            train_loss = self.get_train_loss(**kwargs)
            train_loss.backward()
            opt.step()

            log = {
                "train_loss": f"{train_loss.item():.4f}"
            }

            if validate_every > 0 and (epoch + 1) % validate_every == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self.get_validation_loss(**kwargs)
                    log["val_loss"] = f"{val_loss.item():.4f}"
                self.model.train()

            pbar.set_postfix(log)

        # Final eval
        self.model.eval()


class UnguidedTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: nn.Module) -> None:
        super().__init__(model)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        return self._compute_loss(batch_size, mode='train')

    def get_validation_loss(self, batch_size: int) -> torch.Tensor:
        return self._compute_loss(batch_size, mode='val')

    def _compute_loss(self, batch_size: int, mode: str = 'train') -> torch.Tensor:
        # Sample from p_data
        z = self.path.p_data.sample(batch_size, mode=mode)
        device = z.device

        # Sample t and x
        t = torch.rand(batch_size, 1, 1, 1, device=device)
        x = self.path.sample_conditional_path(z, t)

        # Regress and output loss
        ut_theta = self.model(x, t)  # (batch_size, 4, 128, 128)
        ut_ref = self.path.conditional_vector_field(x, z, t)  # (batch_size, 4, 128, 128)
        return torch.mean(torch.sum(torch.square(ut_theta - ut_ref), dim=-1))