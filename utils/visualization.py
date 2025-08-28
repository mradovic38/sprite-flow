import pandas as pd
import matplotlib.pyplot as plt


def visualize_training_logs(
        log_path: str = 'experiments/unet/training_log.csv',
        scale: str = "linear"
):
    """
    Plot training loss and validation metric over epochs.
    :param log_path: path to log CSV file
    :param scale: on which scale to plot loss/metric values (https://matplotlib.org/stable/users/explain/axes/axes_scales.html)
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
    plt.yscale(scale)
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
    plt.yscale(scale)
    plt.show()