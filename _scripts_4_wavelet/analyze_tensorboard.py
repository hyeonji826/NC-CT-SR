import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16


def clean_plot(log_dir):
    log_dir = Path(log_dir)
    event_files = list(log_dir.glob("events.out.tfevents.*"))

    if not event_files:
        print(f"No event files found in {log_dir}")
        return

    event_file = event_files[0]
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()

    tags = ea.Tags()['scalars']
    data = {}

    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": np.array([e.step for e in events]),
            "values": np.array([e.value for e in events])
        }

    # -------------------------------------
    # Extract key metrics
    # -------------------------------------
    train_loss = data.get("Epoch/train_loss", None)
    val_loss = data.get("Epoch/val_loss", None)
    lr_data = data.get("Epoch/learning_rate", None)
    n2n_total = data.get("Train/n2n_total", None)
    wavelet = data.get("Train/wavelet", None)

    # -------------------------------------
    # Create cleaner figure
    # -------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training Overview", fontweight="bold")

    # -------------------------------------
    # 1. Train Loss Curve
    # -------------------------------------
    if train_loss:
        ax = axes[0, 0]
        ax.plot(train_loss["steps"], train_loss["values"], label="Train Loss", color="royalblue")
        ax.set_title("Train Loss (per epoch)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

    # -------------------------------------
    # 2. Validation Loss Curve
    # -------------------------------------
    if val_loss:
        ax = axes[0, 1]
        ax.plot(val_loss["steps"], val_loss["values"], label="Val Loss", color="crimson")
        ax.set_title("Validation Loss (per epoch)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

    # -------------------------------------
    # 3. Train vs Val (Overlay)
    # -------------------------------------
    if train_loss and val_loss:
        ax = axes[1, 0]
        ax.plot(train_loss["steps"], train_loss["values"], "b-", label="Train", linewidth=2)
        ax.plot(val_loss["steps"], val_loss["values"], "r-", label="Val", linewidth=2)
        ax.set_title("Train vs Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    # -------------------------------------
    # 4. Learning Rate Schedule
    # -------------------------------------
    if lr_data:
        ax = axes[1, 1]
        ax.plot(lr_data["steps"], lr_data["values"], "g-", linewidth=2)
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = log_dir.parent / "training_clean.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Saved clean summary → {out_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = r"E:\LD-CT SR\Outputs\ns_n2n_experiments\unet3d_transformer_baseline\logs"

    clean_plot(log_dir)
