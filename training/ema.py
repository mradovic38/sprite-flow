from torch import nn
import torch


class EMA:
    def __init__(self, model: nn.Module, max_decay: float = 0.9995, warmup_steps: int = 1000, update_every: int = 1) -> None:
        """
        Exponential Moving Average of model weights and buffers.
        :param model: pytorch model
        :param max_decay: Maximum value of the decay factor of EMA
        :param warmup_steps: Warmup steps for EMA - on that step EMA will reach max_decay
        :param update_every: Number of steps between EMA updates
        """
        self.model = model
        self.max_decay = max_decay
        self.shadow = {}
        self.backup = {}
        self.step_count = 1  # Track steps internally
        self.warmup_steps = warmup_steps
        self.update_every = update_every

        # Store initial parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """
        Updates EMA weights and buffers with adaptive decay.
        """
        if self.step_count % self.update_every != 0:
            return

        # Calculate adaptive decay that grows from 0 to max_decay
        decay = min(self.max_decay, (1 + self.step_count) / (self.warmup_steps + self.step_count))
        self.step_count += 1

        with torch.no_grad():
            # Update parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    new_shadow = (1.0 - decay) * param.data + decay * self.shadow[name]
                    self.shadow[name] = new_shadow.clone()

    def apply_shadow(self):
        """
        Temporarily swaps in the EMA weights and buffers into the model.
        """
        # Backup and apply parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self) -> None:
        """
        Copies back the original training weights and buffers from the backup.
        """
        # Restore parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data =  self.backup[name]

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
        self.step_count = state_dict.get('step_count', 0)
        self.max_decay = state_dict.get('max_decay', self.max_decay)

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in state_dict['shadow']:
                self.shadow[name] = state_dict['shadow'][name].clone()