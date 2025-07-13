from abc import ABC, abstractmethod
import torch
from torch import nn

from helpers import model_size_b, MiB

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

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
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