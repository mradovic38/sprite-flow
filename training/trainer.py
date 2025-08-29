from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
import csv

from sampling.conditional_probability_path import GaussianConditionalProbabilityPath
from sampling.sampleable import IterableSampleable
from models.conditional_vector_field import ConditionalVectorField
from training.evaluation import EvaluationMetric
from training.lr_scheduling import CosineWarmupScheduler
from training.ema import EMA
from diff_eq.ode_sde import UnguidedVectorFieldODE
from diff_eq.simulator import EulerSimulator
from utils.helpers import model_size_b, MiB, tensor_to_rgba_image


class Trainer(ABC):
    def __init__(
            self,
            model: nn.Module,
            eval_metric: EvaluationMetric,
            experiment_dir: str = "model",
            ema: EMA = None
    ) -> None:
        """
        :param model: pytorch model
        :param eval_metric: evaluation metric to be used
        :param experiment_dir: path to experiment directory
        :param ema: EMA instance, defaults to no EMA
        """
        super().__init__()
        self.model = model
        self.experiment_dir = experiment_dir
        self.checkpoint_path = os.path.join(experiment_dir, 'best_model.pt')
        self.log_path = os.path.join(experiment_dir, 'training_log.csv')
        self.eval_metric = eval_metric
        self.ema = ema

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def evaluate(
            self,
            batch_size: int,
            device: torch.device,
            num_timesteps: int = 100,
            mode: str = 'val',
            **kwargs) -> torch.Tensor:
        """
        Evaluate the model on the evaluation metric
        :param batch_size
        :param device: Device to perform computation of the metric on
        :param num_timesteps: With how many timesteps to simulate denoising
        :param mode: 'val' or 'test'
        :return: Computed evaluation metric
        """
        pass

    @abstractmethod
    def save_images(self, num_images_to_save: int, epoch: int, device: torch.device, num_timesteps: int = 100) -> None:
        """
        Saves `num_images_to_save` number of images from epoch `epoch` for manual evaluation
        :param num_images_to_save: number of images to save for manual evaluation
        :param epoch: current epoch number
        :param device: device to perform calculations on
        :param num_timesteps: number of timesteps of the denoising process
        """
        pass

    @abstractmethod
    def generate_predictions(
            self,
            num_images: int,
            mode: str = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param num_images: number of images to generate
        :param mode: 'train', 'val' or 'test'
        :return: (ut_theta, z, x, t)
        """
        pass

    def get_optimizer(self, lr: float, weight_decay: float = 0):
        if weight_decay > 0:
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(
            self,
            device: torch.device,
            num_epochs: int,
            batch_size: int = 128,
            val_batch_size: int = 128,
            num_val_batches: Optional[int] = None,
            lr: float = 1e-3,
            weight_decay: float = 0,
            validate_every: int = 1,
            val_timesteps: int = 100,
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
        :param val_batch_size: Batch size for computing validation metrics
        :param num_val_batches: On how many batches to perform validation
        :param lr: learning rate
        :param weight_decay: Weight decay - if 0, uses Adam, if >0 uses AdamW as optimizer
        :param validate_every: validation frequency (number of epochs)
        :param val_timesteps: number of denoising timesteps for validation
        :param resume: whether to resume training or to start over, overwriting the checkpoint file
        :param lr_warmup_steps_frac: learning rate warmup steps - fraction of the total training steps
        :param num_images_to_save: number of images to save for manual evaluation
        :param save_images_every: how often to save images for manual evaluation (number of epochs)
        """
        # Print model size
        size_b = model_size_b(self.model)
        print(f'Model size: {size_b / MiB:.4f} MiB')

        self.model.to(device)
        opt = self.get_optimizer(lr, weight_decay)
        warmup_steps = lr_warmup_steps_frac * num_epochs
        scheduler = CosineWarmupScheduler(optimizer=opt, num_warmup_steps=warmup_steps, num_training_steps=num_epochs)

        start_epoch = 0
        best_val_metric = float("inf")
        last_val_metric = "NA"
        last_best_val_metric="NA (epoch NA)"

        # If first run, create CSV header
        if not resume or not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_metric"])

        # Optionally load from checkpoints
        if resume and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state"])
            opt.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            best_val_metric = checkpoint["best_val_metric"]
            start_epoch = checkpoint["epoch"] + 1
            if self.ema and "ema_state" in checkpoint and checkpoint["ema_state"] is not None:
                self.ema.load_state_dict(checkpoint["ema_state"])
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
            if self.ema:
                self.ema.update()

            log = {
                "train_loss": f"{train_loss.item():.4f}",
                "val_metric": last_val_metric,
                "best_val_metric": last_best_val_metric
            }

            if validate_every > 0 and (epoch + 1) % validate_every == 0:
                self.model.eval()
                with torch.no_grad():
                    val_metric = self.evaluate(
                        batch_size=val_batch_size,
                        mode='val',
                        device=device,
                        num_val_batches=num_val_batches
                    )
                    val_metric_value = val_metric.item()
                    last_val_metric = val_metric_value
                    log["val_metric"] = f"{last_val_metric:.4f}"

                # Save if best
                if val_metric_value < best_val_metric:
                    best_val_metric = val_metric_value
                    last_best_val_metric = f"{best_val_metric:.4f} (epoch {epoch})"
                    log["best_val_metric"] = last_best_val_metric
                    torch.save({
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val_metric": best_val_metric,
                        "ema_state": self.ema.state_dict() if self.ema else None,
                    }, self.checkpoint_path)

                self.model.train()

            # Save images
            if save_images_every > 0 and (epoch + 1) % save_images_every == 0:
                self.model.eval()
                with torch.no_grad():
                    self.save_images(num_images_to_save, epoch, device, val_timesteps)
                self.model.train()

            # Log to CSV
            with open(self.log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    train_loss.item(),
                    last_val_metric
                ])

            pbar.set_postfix(log)

        self.model.eval()


class UnguidedTrainer(Trainer):
    def __init__(
            self,
            path: GaussianConditionalProbabilityPath,
            model: nn.Module,
            eval_metric: EvaluationMetric,
            experiment_dir: str = 'unet',
            ema: EMA = None
    ) -> None:
        super().__init__(model, eval_metric, experiment_dir, ema)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        return self._compute_loss(batch_size, mode='train')

    def generate_predictions(
            self,
            num_images: int,
            mode: str = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        loss = torch.mean((ut_theta - ut_ref) ** 2)
        return loss

    def evaluate(
            self,
            batch_size: int,
            device: torch.device,
            num_timesteps: int = 100,
            mode: str = 'val',
            num_batches: Optional[int] = None
    ) -> torch.Tensor:
        assert isinstance(self.path.p_data, IterableSampleable) and isinstance(self.model, ConditionalVectorField)

        if self.ema:
            self.ema.apply_shadow()

        ode = UnguidedVectorFieldODE(self.model)
        simulator = EulerSimulator(ode)

        self.eval_metric.prepare(device)

        # Loop over validation/test dataset
        for real_batch in self.path.p_data.iterate_dataset(batch_size, mode=mode):
            if num_batches:
                if num_batches == 0:
                    break
                num_batches -= 1

            B = real_batch.shape[0]
            real_batch = real_batch.to(device)  # (B, 4, H, W)

            # Generate matching number of fake images
            ts = torch.linspace(0, 1, steps=num_timesteps, device=device).view(1, -1, 1, 1, 1).expand(B, -1, 1, 1, 1)
            x0 = self.path.p_simple.sample(B).to(device)  # (B, 4, H, W)
            generated = simulator.simulate(x0, ts)[:, :4]  # RGBA

            # Update FID
            self.eval_metric.evaluate_batch(real_batch, generated, device)

        if self.ema:
            self.ema.restore()

        return self.eval_metric.compute()

    def save_images(self, num_images_to_save: int, epoch: int, device: torch.device, num_timesteps: int = 100) -> None:
        assert isinstance(self.path.p_data, IterableSampleable) and isinstance(self.model, ConditionalVectorField)

        os.makedirs(self.experiment_dir, exist_ok=True)
        output_dir = os.path.join(self.experiment_dir, f"epoch-{epoch}")
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()

        if self.ema:
            self.ema.apply_shadow()

        # Create time steps
        ts = torch.linspace(0, 1, steps=num_timesteps).view(1, -1, 1, 1, 1)
        ts = ts.expand(num_images_to_save, -1, 1, 1, 1).to(device)

        # Sample from prior and simulate
        x0 = self.path.p_simple.sample(num_images_to_save).to(ts.device)
        ode = UnguidedVectorFieldODE(self.model)
        simulator = EulerSimulator(ode)
        generated = simulator.simulate(x0, ts)  # (B, 4, H, W)

        for i in range(num_images_to_save):
            img_tensor = generated[i]
            img = tensor_to_rgba_image(img_tensor)  # Expects a tensor in [-1, 1] or [0, 1], shape (4, H, W)
            img.save(os.path.join(output_dir, f"image_{i}.png"))

        if self.ema:
            self.ema.restore()