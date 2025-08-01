import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
from PIL import Image


MiB = 1024 ** 2

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    :param model: PyTorch model
    :return: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


def tensor_to_rgba_image(tensor: torch.Tensor) -> Image.Image:
    """
    Converts a (4, H, W) tensor to a transparent PNG image. Assumes 4 channels: R, G, B, A in [0, 1] range.
    :param tensor: tensor to convert
    :return: PNG image (RGBA)
    """
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.shape[0] == 1:  # grayscale, replicate to RGB
        tensor = tensor.expand(4, -1, -1)
    elif tensor.shape[0] == 3:  # no alpha, add full alpha
        alpha = torch.ones(1, *tensor.shape[1:])
        tensor = torch.cat((tensor, alpha), dim=0)
    elif tensor.shape[0] != 4:
        raise ValueError("Expected tensor with 1, 3, or 4 channels")

    return to_pil_image(tensor, mode='RGBA')