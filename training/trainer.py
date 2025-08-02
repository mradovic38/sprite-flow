from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os

from helpers import model_size_b, MiB, tensor_to_rgba_image
from sampling.conditional_probability_path import GaussianConditionalProbabilityPath
from lr_scheduling import CosineWarmupScheduler
from sampling.sampleable import IterableSampleable


class Trainer(ABC):
    def __init__(self, model: nn.Module, experiment_dir: str = "model") -> None:
        super().__init__()
        self.model = model
        self.experiment_dir = experiment_dir
        self.checkpoint_path = os.path.join(experiment_dir, 'best_model.pt')

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def get_validation_loss(self, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def generate_predictions(self, num_images: int, mode: str = 'train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param num_images: number of images to generate
        :param mode: 'train', 'val' or 'test'
        :return: (ut_theta, z, x, t)
        """
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
            self,
            device: torch.device,
            num_epochs: int,
            batch_size: int = 128,
            lr: float = 1e-3,
            validate_every: int = 1,
            resume: bool = False,
            lr_warmup_steps_frac: float = 0.1,
            num_images_to_save: int = 5,
            save_images_every: int = 10,
            **kwargs
    ) -> None:
        """
        Trains the model and saves the model checkpoints.
        :param device: device to train on
        :param num_epochs: total number of training epochs
        :param batch_size
        :param lr: learning rate
        :param validate_every: validation frequency (number of epochs)
        :param resume: whether to resume training or to start over, overwriting the checkpoint file
        :param lr_warmup_steps_frac: learning rate warmup steps - fraction of the total training steps
        :param num_images_to_save: number of images to save for manual evaluation
        :param save_images_every: how often to save images for manual evaluation (number of epochs)
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
            train_loss = self.get_train_loss(batch_size=batch_size, **kwargs)
            train_loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            log = {"train_loss": f"{train_loss.item():.4f}"}

            if validate_every > 0 and (epoch + 1) % validate_every == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self.get_validation_loss(batch_size=batch_size, **kwargs)
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

                self.model.train()

            if save_images_every > 0 and (epoch + 1) % save_images_every == 0:
                self.model.eval()
                with torch.no_grad():
                    self._save_images(num_images_to_save, epoch + 1)
                self.model.train()

            pbar.set_postfix(log)

        # Final eval
        self.model.eval()

    def _save_images(self, num_images_to_save: int, epoch: int) -> None:
        """
        Saves `num_images_to_save` number of images from epoch `epoch` for manual evaluation
        :param num_images_to_save: number of images to save for manual evaluation
        :param epoch: current epoch number
        """
        self.model.eval()
        os.makedirs(self.experiment_dir, exist_ok=True)
        output_dir = os.path.join(self.experiment_dir, f"epoch-{epoch}")
        os.makedirs(output_dir, exist_ok=True)

        # Sample images
        ut_theta, _, _, _ = self.generate_predictions(num_images_to_save, mode='val')

        for i in range(num_images_to_save):
            img = tensor_to_rgba_image(ut_theta[i])
            img.save(os.path.join(output_dir, f"image_{i + 1}.png"))


class UnguidedTrainer(Trainer):
    def __init__(
            self,
            path: GaussianConditionalProbabilityPath,
            model: nn.Module,
            experiment_dir: str = 'unet'
    ) -> None:
        super().__init__(model, experiment_dir)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        return self._compute_loss(batch_size, mode='train')

    def get_validation_loss(self, batch_size: int) -> torch.Tensor:
        return self._compute_loss_all_batches(batch_size, mode='val')

    def get_test_loss(self, batch_size: int) -> torch.Tensor:
        return self._compute_loss_all_batches(batch_size, mode='test')

    def generate_predictions(self, num_images: int, mode: str = 'train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sample from p_data
        z = self.path.p_data.sample(num_images, mode=mode)
        device = z.device

        # Sample t and x
        t = torch.rand(num_images, 1, 1, 1, device=device)
        x = self.path.sample_conditional_path(z, t)

        # Regress
        ut_theta = self.model(x, t)  # (num_images, 4, 128, 128)
        return ut_theta, z, x, t

    def _compute_loss(self, batch_size: int, mode: str = 'train') -> torch.Tensor:
        # Generate predictions
        ut_theta, z, x, t = self.generate_predictions(num_images=batch_size, mode=mode)
        # Calculate and return loss
        ut_ref = self.path.conditional_vector_field(x, z, t)  # (batch_size, 4, 128, 128)
        return torch.mean(torch.sum(torch.square(ut_theta - ut_ref), dim=-1))

    def _compute_loss_all_batches(self, batch_size: int, mode: str = 'val') -> torch.Tensor:
        total_loss = 0.0
        num_batches = 0

        if isinstance(self.path.p_data, IterableSampleable):
            # Finite dataset => Iterate over the dataset in batches
            for z in self.path.p_data.iterate_dataset(batch_size=batch_size, mode=mode):
                device = z.device
                t = torch.rand(z.size(0), 1, 1, 1, device=device)
                x = self.path.sample_conditional_path(z, t)
                ut_theta = self.model(x, t)
                ut_ref = self.path.conditional_vector_field(x, z, t)  # (batch_size, 4, 128, 128)
                loss = torch.mean(torch.sum(torch.square(ut_theta - ut_ref), dim=-1))
                total_loss += loss.item()
                num_batches += 1

            if num_batches == 0:
                raise ValueError(f"No batches found in mode={mode} dataset.")

            return torch.tensor(total_loss / num_batches, device=device)

        else:
            # Fallback: use one batch
            return self._compute_loss(batch_size=batch_size, mode=mode)

        