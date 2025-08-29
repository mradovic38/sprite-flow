import gc
from typing import List
import glob

import torch
from torch import nn
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

def tensor_to_rgba_image(tensor: torch.Tensor) -> List[Image.Image]:
    """
    Converts a tensor to RGBA PIL image(s).
    :param tensor: Tensor with values in [0, 1], shape (N, C, H, W) or (C, H, W)
    :return: RGBA PIL images.
    """
    if tensor.ndim == 3:  # (C, H, W)
        tensor = tensor.unsqueeze(0)  # add batch dim

    images: List[Image.Image] = []
    for img in tensor:  # iterate over batch
        if img.shape[0] == 1:  # grayscale → replicate RGB + full alpha
            rgb = img.expand(3, -1, -1)
            alpha = torch.ones(1, *img.shape[1:])
            img = torch.cat((rgb, alpha), dim=0)
        elif img.shape[0] == 3:  # RGB → add full alpha
            alpha = torch.ones(1, *img.shape[1:])
            img = torch.cat((img, alpha), dim=0)
        elif img.shape[0] == 4:  # already RGBA
            pass
        else:
            raise ValueError("Expected tensor with 1, 3, or 4 channels")

        img = (img * 255).byte().permute(1, 2, 0).numpy()  # (H, W, 4)
        images.append(Image.fromarray(img, mode="RGBA"))

    return images

def rgba_to_rgb(images: torch.Tensor) -> torch.Tensor:
    """
    Converts RGBA images to RGB color space.
    :param images: tensor with RGBA images, shape (num_images, 4, H, W)
    :return: RGB tensor with shape (num_images, 3, H, W)
    """
    # images: (N, 4, H, W), values assumed in [0, 1]
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
    return F.interpolate(images, size=size, mode='nearest-exact')

def normalize_to_unit(images: torch.Tensor) -> torch.Tensor:
    """
    Normalizes images from [-1, 1] to [0, 1] range.
    :param images: images to normalize
    :return: normalized images
    """
    # [-1,1] -> [0,1]
    return ((images + 1) / 2).clamp(0, 1)

def clear_cuda() -> None:
    gc.collect()                          # Python garbage collection
    torch.cuda.empty_cache()              # clears cached memory
    torch.cuda.ipc_collect()              # releases shared memory (optional)

def save_generated_assets(images: List[Image.Image], num_timesteps: int, path: str = 'assets/unet') -> None:
    """
    Saves generated images to a dedicated folder.
    :param images: List of generated images
    :param num_timesteps: Number of timesteps that generated the images
    :param path: Path to folder to save generated images
    """
    j = 0
    for i, img in enumerate(images):
        while True:
            if not glob.glob(f"{path}/image_{i + j}"):
                img.save(f"assets/unet/image_{i + j}-{num_timesteps}.png")
                break
            j += 1