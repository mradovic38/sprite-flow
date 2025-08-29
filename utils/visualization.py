from typing import List

import pandas as pd
import math
import matplotlib.pyplot as plt
from PIL.Image import Image


def visualize_training_logs(
        log_path: str = 'experiments/unet/training_log.csv',
        loss_scale: str = "linear",
        metric_scale: str = "linear"
):
    """
    Plot training loss and validation metric over epochs.
    :param log_path: path to log CSV file
    :param loss_scale: scale on which scale to plot loss values (https://matplotlib.org/stable/users/explain/axes/axes_scales.html)
    :param metric_scale: scale on which scale to plot validation metric values
    :return:
    """
    df = pd.read_csv(log_path)

    # Ensure epoch is int
    df["epoch"] = df["epoch"].astype(int)

    # Convert columns to numeric
    df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
    df["val_metric"] = pd.to_numeric(df["val_metric"], errors="coerce")

    # === Prepare Validation Data ===
    validation_every = df["val_metric"].isna().sum() + 1
    validated_epochs = df[df["epoch"].apply(lambda e: (e + 1) % validation_every == 0)]
    validated_epochs = validated_epochs[validated_epochs["val_metric"].notna()]

    # === Plot Training Loss ===
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")

    # Mark best model
    if not validated_epochs.empty:
        best_train_idx = df["val_metric"].idxmin()
        best_train = df.loc[best_train_idx, "train_loss"]

        best_epoch = df.loc[best_train_idx, "epoch"]

        plt.scatter(best_epoch, best_train, color="red", zorder=6,
                    label=f"Best Model (epoch {best_epoch})")

    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.xlim(df["epoch"].min() - .5, df["epoch"].max() + .5)
    plt.tight_layout()
    plt.yscale(loss_scale)
    plt.show()

    # === Plot Validation Metric ===
    plt.figure(figsize=(8, 5))
    plt.plot(validated_epochs["epoch"], validated_epochs["val_metric"], label="Val Metric", color="green")

    # Mark best model
    if not validated_epochs.empty:
        best_val_idx = df["val_metric"].idxmin()
        best_val = df.loc[best_val_idx, "val_metric"]

        best_epoch = df.loc[best_val_idx, "epoch"]

        plt.scatter(best_epoch, best_val, color="red", zorder=6,
                    label=f"Best Model (epoch {best_epoch}; val metric: {best_val:.4f})")

    plt.ylabel("Val Metric")
    plt.xlabel("Epoch")
    plt.title("Val Metric over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.xlim(df["epoch"].min() - .5, df["epoch"].max() + .5)
    plt.tight_layout()
    plt.yscale(metric_scale)
    plt.show()


def plot_generated_images(images: List[Image], max_row: int = 5, figsize_per_image: int = 3) -> None:
    """
    Plots a list of PIL images.
    :param images: List of PIL Image objects
    :param max_row: Maximum number of images to plot per row
    :param figsize_per_image: Size of each image in the figure (default 3)
    """
    n_images = len(images)
    n_cols = min(max_row, n_images)
    n_rows = math.ceil(n_images / n_cols)

    plt.figure(figsize=(figsize_per_image * n_cols, figsize_per_image * n_rows))

    for i, img in enumerate(images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()