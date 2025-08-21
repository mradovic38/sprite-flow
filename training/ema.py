from torch import nn

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        """
        Exponential Moving Average of model weights.
        :param model: pytorch model
        :param decay: decay factor of EMA
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # store initial parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    def update(self) -> None:
        """
        Updates EMA weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()

    def apply_shadow(self):
        """
        Temporarily swaps in the EMA weights into the model so they can be used for evaluation or inference.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """
        Copies back the original training weights from the backup.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}