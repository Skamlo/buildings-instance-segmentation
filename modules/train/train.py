from tqdm import tqdm
import torch
from modules.train.validate_one_epoch import validate_one_epoch
from modules.train.metrics import compute_grad_norm, compute_param_norm


def train(model, optimizer, lr_scheduler, train_loader, val_loader, device, num_epochs, save_dir="./models"):
    history = {
        "train": {
            "batch": [],
            "epoch": []
        },
        "val": {
            "epoch": []
        }
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0   # Gradient Norm (per-batch)
        epoch_fg_ratio = 0.0    # Foreground Ratio (per-batch)
        epoch_param_norm = 0.0  # Param (weight) Norm (per-batch, after update)

        num_batches = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
            num_batches += 1

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward -> get loss dict (Mask R-CNN in train mode returns dict of losses)
            loss_dict = model(images, targets)
            loss = sum(v.mean() for v in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()

            # Compute grad norm BEFORE step (gradients are populated)
            grad_norm = compute_grad_norm(model)

            optimizer.step()

            # Compute parameter norm AFTER step (reflects updated params)
            param_norm = compute_param_norm(model)

            # Robust FG ratio calculation (handles empty masks)
            try:
                masks_cat = torch.cat([t["masks"] for t in targets], dim=0)
                if masks_cat.numel() == 0:
                    fg_ratio = 0.0
                else:
                    fg_ratio = masks_cat.float().mean().item()
            except Exception:
                fg_ratio = 0.0

            # Append per-batch metrics (no LR)
            history["train"]["batch"].append({
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "param_norm": param_norm,
                "fg_ratio": fg_ratio
            })

            # Accumulate for epoch averages
            epoch_loss += loss.item()
            epoch_grad_norm += grad_norm
            epoch_param_norm += param_norm
            epoch_fg_ratio += fg_ratio

        # Defensive: avoid division by zero
        if num_batches == 0:
            avg_loss = 0.0
            avg_grad = 0.0
            avg_param = 0.0
            avg_fg = 0.0
        else:
            avg_loss = epoch_loss / num_batches
            avg_grad = epoch_grad_norm / num_batches
            avg_param = epoch_param_norm / num_batches
            avg_fg = epoch_fg_ratio / num_batches

        # ---- Epoch averages ----
        history["train"]["epoch"].append({
            "loss": avg_loss,
            "grad_norm": avg_grad,
            "param_norm": avg_param,
            "fg_ratio": avg_fg
        })

        # ---- Validation (loss-only) ----
        val_metrics = validate_one_epoch(model, val_loader, device)
        history["val"]["epoch"].append(val_metrics)

        lr_scheduler.step()

        # Print summary
        print(f"\nEpoch {epoch} Summary")
        print(f" Train Loss:   {history['train']['epoch'][-1]['loss']:.4f}")
        print(f" Grad Norm:    {history['train']['epoch'][-1]['grad_norm']:.4f}")
        print(f" Param Norm:   {history['train']['epoch'][-1]['param_norm']:.4f}")
        print(f" FG Ratio:     {history['train']['epoch'][-1]['fg_ratio']:.4f}")
        print(f" Val Loss:     {val_metrics['loss']:.4f}")
        print(f" Val MaskLoss: {val_metrics['mask_loss']:.4f}")

        torch.save(model.state_dict(), f"{save_dir}/instance_segmentation_{epoch}.pth")

    return history
