from torch.optim.lr_scheduler import LambdaLR
import math

class CosineWarmupScheduler(LambdaLR):
    """
    Implementation of the learning rate scheduler with warmup and cosine decay.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))