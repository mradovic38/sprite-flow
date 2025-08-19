import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
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
    Converts a (C, H, W) tensor to a transparent PNG image.
    Assumes 1, 3, or 4 channels, values in [-1, 1].
    Returns RGBA PIL image.
    """
    # Rescale from [-1,1] to [0,1]
    tensor = (tensor + 1) / 2
    tensor = tensor.detach().cpu().clamp(0, 1)

    # Handle channel count
    if tensor.shape[0] == 1:  # grayscale → replicate RGB + full alpha
        rgb = tensor.expand(3, -1, -1)
        alpha = torch.ones(1, *tensor.shape[1:])
        tensor = torch.cat((rgb, alpha), dim=0)
    elif tensor.shape[0] == 3:  # RGB → add full alpha
        alpha = torch.ones(1, *tensor.shape[1:])
        tensor = torch.cat((tensor, alpha), dim=0)
    elif tensor.shape[0] == 4:  # already RGBA → ok
        pass
    else:
        raise ValueError("Expected tensor with 1, 3, or 4 channels")

    return to_pil_image(tensor, mode='RGBA')


def rgba_to_rgb(images: torch.Tensor) -> torch.Tensor:
    """
    Converts RGBA images to RGB color space.
    :param images: tensor with RGBA images, shape (num_images, 4, H, W)
    :return: RGB tensor with shape (num_images, 3, H, W)
    """
    # images: (N, 4, H, W), values assumed in [-1, 1] or [0, 1]
    rgb = images[:, :3]  # take first 3 channels

    alpha = images[:, 3:4]  # alpha channel, shape (N,1,H,W)
    # Blend with white background: rgb * alpha + (1-alpha)*white(=1)
    rgb = rgb * alpha + (1 - alpha) * 1.0
    return rgb


def resize_images(images: torch.Tensor, size=(299,299)) -> torch.Tensor:
    """
    Resizes images to given size.
    :param images: tensor of images, shape (num_images, C, H, W)
    :param size: desired output image dimensions
    :return: resized tensor of images, shape (num_images, C, new_H, new_W)
    """
    return F.interpolate(images, size=size, mode='bilinear', align_corners=False)