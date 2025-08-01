from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os

from helpers import model_size_b, MiB
from sampling.conditional_probability_path import GaussianConditionalProbabilityPath
from lr_scheduling import CosineWarmupScheduler


class Trainer(ABC):
    def __init__(self, model: nn.Module, checkpoint_path: str = "model_checkpoint.pt") -> None:
        super().__init__()
        self.model = model
        self.checkpoint_path = checkpoint_path

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def get_validation_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
            self,
            num_epochs: int,
            device: torch.device,
            lr: float = 1e-3,
            validate_every: int = 1,
            resume: bool = False,
            lr_warmup_steps_frac: float = 0.1,
            **kwargs
    ) -> None:
        """
        Trains the model and saves the model checkpoints.
        :param num_epochs: total number of training epochs
        :param device: device to train on
        :param lr: learning rate
        :param validate_every: validation frequency
        :param resume: whether to resume training or to start over, overwriting the checkpoint file
        :param lr_warmup_steps_frac: learning rate warmup steps - fraction of the total training steps
        """
        # Print model size
        size_b = model_size_b(self.model)
        print(f'Model size: {size_b / MiB:.4f} MiB')

        self.model.to(device)
        opt = self.get_optimizer(lr)
        warmup_steps = lr_warmup_steps_frac * num_epochs
        scheduler = CosineWarmupScheduler(optimizer=opt, num_warmup_steps=warmup_steps, num_training_steps=num_epochs)

        start_epoch = 0
        best_val_loss = float("inf")

        # Optionally load from checkpoints
        if resume and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state"])
            opt.load_state_dict(checkpoint["optimizer_state"])
            best_val_loss = checkpoint["best_val_loss"]
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from checkpoint at epoch {start_epoch}")

        # Start training
        self.model.train()
        pbar = tqdm(enumerate(range(start_epoch, num_epochs)), total=num_epochs-start_epoch)

        for _, epoch in pbar:
            opt.zero_grad()
            train_loss = self.get_train_loss(**kwargs)
            train_loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            log = {"train_loss": f"{train_loss.item():.4f}"}

            if validate_every > 0 and (epoch + 1) % validate_every == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self.get_validation_loss(**kwargs)
                    val_loss_value = val_loss.item()
                    log["val_loss"] = f"{val_loss_value:.4f}"

                # Save if best
                if val_loss_value < best_val_loss:
                    best_val_loss = val_loss_value
                    torch.save({
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "best_val_loss": best_val_loss,
                    }, self.checkpoint_path)
                    log["checkpoint"] = "saved"

                self.model.train()

            pbar.set_postfix(log)

        # Final eval
        self.model.eval()


class UnguidedTrainer(Trainer):
    def __init__(
            self,
            path: GaussianConditionalProbabilityPath,
            model: nn.Module,
            experiment_name: str = 'unet',
            num_images_for_valid: int = 5,
    ) -> None:
        super().__init__(model, os.path.join(experiment_name, 'checkpoint.pt'))
        self.path = path
        self.experiment_name = experiment_name
        self.num_images_for_valid = num_images_for_valid

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        return self._compute_loss(batch_size, mode='train')

    def get_validation_loss(self, batch_size: int) -> torch.Tensor:
        return self._compute_loss(batch_size, mode='val')

    def save_validation_images(self, images: torch.Tensor) -> None:
        raise NotImplementedError() # TODO: implement

    def _compute_loss(self, batch_size: int, mode: str = 'train') -> torch.Tensor:
        # Sample from p_data
        z = self.path.p_data.sample(batch_size, mode=mode)
        device = z.device

        # Sample t and x
        t = torch.rand(batch_size, 1, 1, 1, device=device)
        x = self.path.sample_conditional_path(z, t)

        # Regress and output loss
        ut_theta = self.model(x, t)  # (batch_size, 4, 128, 128)
        if mode == 'val' and ut_theta.shape[0] <= self.num_images_for_valid:
            self.save_validation_images(ut_theta[:self.num_images_for_valid])
        ut_ref = self.path.conditional_vector_field(x, z, t)  # (batch_size, 4, 128, 128)
        return torch.mean(torch.sum(torch.square(ut_theta - ut_ref), dim=-1))