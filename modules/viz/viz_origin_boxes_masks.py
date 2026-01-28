import os
import random
import colorsys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from modules.transformer import get_transformer


def random_color(seed=None):
    """Return a random color as an (R, G, B) tuple in 0..255."""
    if seed is not None:
        rnd = random.Random(seed)
        r, g, b = colorsys.hls_to_rgb(rnd.random(), 0.5, 1)
    else:
        r, g, b = colorsys.hls_to_rgb(random.random(), 0.5, 1)
    return int(r * 255), int(g * 255), int(b * 255)


def rgb_to_norm_tuple(r, g, b):
    """Return color normalized 0..1 tuple for matplotlib."""
    return (max(0, min(255, r)) / 255.0,
            max(0, min(255, g)) / 255.0,
            max(0, min(255, b)) / 255.0)


def viz_origin_boxes_masks(image_path, model, device,
                        threshold=0.5,
                        figsize=(6, 18),
                        image_alpha_for_masks=0.3,
                        mask_alpha=0.6):
    """
    Visualize Mask R-CNN predictions as 3 subplots:
      (A) original image
      (B) original + boxes
      (C) original with alpha=image_alpha_for_masks + colored masks overlay

    Args:
        image_path: path to image (.tif/.png/...)
        model: a loaded Mask R-CNN / instance segmentation model
        device: torch.device
        threshold: score threshold for showing detections
        include_masks / include_boxes: booleans to toggle features
        figsize: figure size (width, height)
        image_alpha_for_masks: float, alpha of base image in the mask subplot
        mask_alpha: float, alpha used for each mask overlay
    """
    model.eval()
    model.to(device)

    # Load image and transform
    img_pil = Image.open(image_path).convert("RGB")
    transform = get_transformer()
    img_tensor = transform(img_pil).to(device)

    # Inference
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    # Move predictions to CPU / numpy
    boxes = prediction.get('boxes', torch.zeros((0, 4))).cpu().numpy()
    scores = prediction.get('scores', torch.zeros((0,))).cpu().numpy()
    masks = prediction.get('masks', None)
    if masks is not None:
        masks = masks.cpu().numpy()

    # filter by score
    keep_idx = np.where(scores > threshold)[0]

    # Prepare base image as float (0..1)
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    H, W = img_np.shape[:2]

    # Prepare mask stack if masks present
    if masks is not None and masks.size:
        # masks may be (N, 1, H, W) or (N, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        # Ensure boolean masks
        bool_masks = masks > 0.5
    else:
        bool_masks = None

    # Assign colors per kept instance
    colors = {}
    for i in keep_idx:
        colors[i] = random_color()

    # Build a combined RGBA overlay for masks (for subplot C)
    overlay = np.zeros((H, W, 4), dtype=np.float32)  # RGBA
    if (bool_masks is not None):
        for i in keep_idx:
            mask = bool_masks[i]
            if mask.sum() == 0:
                continue
            r, g, b = colors[i]
            r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
            # For overlapping masks we take max alpha (keeps the most visible)
            overlay[..., 0][mask] = r_n
            overlay[..., 1][mask] = g_n
            overlay[..., 2][mask] = b_n
            # set alpha to the max of existing and mask_alpha
            overlay[..., 3][mask] = np.maximum(overlay[..., 3][mask], mask_alpha)

    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    titles = [
        "Original",
        "Original + Boxes",
        f"Image alpha={image_alpha_for_masks} + Masks"
    ]

    for ax, title in zip(axes, titles):
        ax.axis('off')
        ax.set_title(title)

    # Subplot A: original
    axes[0].imshow(img_np)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Subplot B: original + boxes
    axes[1].imshow(img_np)
    for i in keep_idx:
        if boxes.shape[0] > 0:
            x1, y1, x2, y2 = boxes[i]
            w_box = x2 - x1
            h_box = y2 - y1
            color = rgb_to_norm_tuple(*colors[i])
            rect = patches.Rectangle((x1, y1), w_box, h_box,
                                     linewidth=2, edgecolor=color, facecolor='none')
            axes[1].add_patch(rect)
            # Add score label
            txt_x = max(x1, 0)
            txt_y = max(y1 - 6, 0)
            axes[1].text(txt_x, txt_y, f"{scores[i]:.2f}",
                         fontsize=8, color='white',
                         bbox=dict(facecolor=color, edgecolor='none', pad=1.0))

    # Subplot C: image with lower alpha and masks overlay
    # show base image with lower alpha
    axes[2].imshow(np.zeros_like(img_np))
    axes[2].imshow(img_np, alpha=image_alpha_for_masks)
    # overlay masks
    if (overlay[..., 3].sum() > 0):
        # overlay is RGBA in 0..1
        axes[2].imshow(overlay)

    fig.suptitle(f"Predictions — {os.path.basename(image_path)}", fontsize=14)
    plt.tight_layout()
    plt.show()
