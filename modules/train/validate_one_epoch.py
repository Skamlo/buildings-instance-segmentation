from tqdm import tqdm
import torch


def validate_one_epoch(model, data_loader, device):
    """
    Validation pass that computes losses ONLY.
    """
    model.train()

    running_loss = 0.0
    running_mask_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(v.mean() for v in loss_dict.values())
            mask_loss = loss_dict["loss_mask"].mean()

            running_loss += losses.item()
            running_mask_loss += mask_loss.item()

    return {
        "loss": running_loss / len(data_loader),
        "mask_loss": running_mask_loss / len(data_loader)
    }
