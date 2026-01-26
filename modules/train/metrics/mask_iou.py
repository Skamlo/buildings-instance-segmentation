import torch


def mask_iou(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum().float()
    union = (pred_mask | gt_mask).sum().float()
    return (intersection / union) if union > 0 else torch.tensor(0.0)
