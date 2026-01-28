import matplotlib.pyplot as plt
import numpy as np


def viz_history(history):
    train_epochs = history["train"]["epoch"]
    val_epochs = history["val"]["epoch"]

    epochs = range(1, len(train_epochs) + 1)

    # ---- Training metrics ----
    train_loss = [e["loss"] for e in train_epochs]
    grad_norm = [e["grad_norm"] for e in train_epochs]
    param_norm = [e["param_norm"] for e in train_epochs]
    fg_ratio = [e["fg_ratio"] for e in train_epochs]

    # ---- Validation metrics ----
    val_loss = [e["loss"] for e in val_epochs]
    val_mask_loss = [e["mask_loss"] for e in val_epochs]

    # Traning metrics
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))
    n_epochs = len(train_loss)

    ax[0][0].plot(train_loss)
    ax[0][0].set_title("Train Loss")
    ax[0][0].set_xlabel("Epoch")
    ax[0][0].set_ylabel("Loss")
    ax[0][0].set_xticks(np.arange(n_epochs), np.arange(1, n_epochs+1))

    ax[0][1].plot(grad_norm)
    ax[0][1].set_title("Gradient Norm")
    ax[0][1].set_xlabel("Epoch")
    ax[0][1].set_ylabel("Norm")
    ax[0][1].set_xticks(np.arange(n_epochs), np.arange(1, n_epochs+1))

    ax[1][0].plot(param_norm)
    ax[1][0].set_title("Parameter Norm")
    ax[1][0].set_xlabel("Epoch")
    ax[1][0].set_ylabel("Norm")
    ax[1][0].set_xticks(np.arange(n_epochs), np.arange(1, n_epochs+1))

    ax[1][1].plot(fg_ratio)
    ax[1][1].set_title("Foreground Ratio")
    ax[1][1].set_xlabel("Epoch")
    ax[1][1].set_ylabel("Ratio")
    ax[1][1].set_xticks(np.arange(n_epochs), np.arange(1, n_epochs+1))

    plt.tight_layout()
    plt.show()

    # Validation metrics
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    
    ax[0].plot(val_loss)
    ax[0].set_title("Validation Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_xticks(np.arange(n_epochs), np.arange(1, n_epochs+1))

    ax[1].plot(val_mask_loss)
    ax[1].set_title("Validation Mask Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Norm")
    ax[1].set_xticks(np.arange(n_epochs), np.arange(1, n_epochs+1))

    plt.tight_layout()
    plt.show()
