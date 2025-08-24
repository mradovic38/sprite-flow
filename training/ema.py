from torch import nn
import torch


class EMA:
    def __init__(self, model: nn.Module, max_decay: float = 0.999) -> None:
        """
        Exponential Moving Average of model weights and buffers.
        :param model: pytorch model
        :param max_decay: Maximum value of the decay factor of EMA
        """
        self.model = model
        self.max_decay = max_decay
        self.shadow = {}
        self.backup = {}
        self.step_count = 0  # Track steps internally

        # Store initial parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

        # Store initial buffers (BatchNorm running stats, etc.)
        for name, buffer in model.named_buffers():
            self.shadow[name] = buffer.clone().detach()

    def update(self) -> None:
        """
        Updates EMA weights and buffers with adaptive decay.
        """
        # Calculate adaptive decay that grows from 0 to max_decay
        decay = min(self.max_decay, (1 + self.step_count) / (10 + self.step_count))
        self.step_count += 1

        with torch.no_grad():
            # Update parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = decay * self.shadow[name] + (1.0 - decay) * param.detach()

            # Update buffers
            for name, buffer in self.model.named_buffers():
                self.shadow[name] = decay * self.shadow[name] + (1.0 - decay) * buffer.detach()

    def apply_shadow(self):
        """
        Temporarily swaps in the EMA weights and buffers into the model.
        """
        # Backup and apply parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.clone()
                param.data.copy_(self.shadow[name])

        # Backup and apply buffers
        for name, buffer in self.model.named_buffers():
            self.backup[name] = buffer.clone()
            buffer.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """
        Copies back the original training weights and buffers from the backup.
        """
        # Restore parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])

        # Restore buffers
        for name, buffer in self.model.named_buffers():
            if name in self.backup:
                buffer.data.copy_(self.backup[name])

        self.backup = {}

    def state_dict(self):
        """
        Returns the EMA shadow state dict for saving.
        """
        return {
            'shadow': self.shadow.copy(),
            'step_count': self.step_count,
            'max_decay': self.max_decay
        }

    def load_state_dict(self, state_dict):
        """
        Loads EMA shadow weights from state dict.
        """
        self.shadow = state_dict['shadow'].copy()
        self.step_count = state_dict.get('step_count', 0)
        self.max_decay = state_dict.get('max_decay', self.max_decay)