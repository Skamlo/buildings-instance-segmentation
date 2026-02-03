import os
import random
import colorsys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from typing import Tuple
from modules.transformer import get_transformer


def random_color(seed:int=None) -> Tuple[int]:
    if seed is not None:
        rnd = random.Random(seed)
        r, g, b = colorsys.hls_to_rgb(rnd.random(), 0.5, 1)
    else:
        r, g, b = colorsys.hls_to_rgb(random.random(), 0.5, 1)
    return int(r * 255), int(g * 255), int(b * 255)


def rgb_to_norm_tuple(r:int, g:int, b:int) -> Tuple[int]:
    return (max(0, min(255, r)) / 255.0,
            max(0, min(255, g)) / 255.0,
            max(0, min(255, b)) / 255.0)


def viz_origin_boxes_masks(
        image_path, model, device,
        threshold=0.5,
        figsize=(6, 18),
        image_alpha_for_masks=0.3,
        mask_alpha=0.6,
        suptitle=True
    ):
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


    # Plot
    fig, ax = plt.subplots(3, 1, figsize=figsize)

    # Subplot A: Original
    ax[0].imshow(img_np)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Original", fontsize=14)

    # Subplot 2: Boxes
    colors = {}
    for i in keep_idx:
        colors[i] = random_color()

    ax[1].imshow(np.zeros_like(img_np))
    ax[1].imshow(img_np, alpha=image_alpha_for_masks)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("Boxes", fontsize=14)
    for i in keep_idx:
        if boxes.shape[0] > 0:
            x1, y1, x2, y2 = boxes[i]
            w_box = x2 - x1
            h_box = y2 - y1
            color = rgb_to_norm_tuple(*colors[i])
            rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=1, edgecolor=color, facecolor='none')
            ax[1].add_patch(rect)
            txt_x = max(x1, 0)
            txt_y = max(y1 - 6, 0)
            ax[1].text(txt_x, txt_y, f"{scores[i]:.2f}", fontsize=8, color='black', bbox=dict(facecolor=color, edgecolor='none', pad=1.0))
    
    # Subplot 3: Masks
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    if (bool_masks is not None):
        for i in keep_idx:
            mask = bool_masks[i]
            if mask.sum() == 0:
                continue
            r, g, b = colors[i]
            r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
            overlay[..., 0][mask] = r_n
            overlay[..., 1][mask] = g_n
            overlay[..., 2][mask] = b_n
            overlay[..., 3][mask] = np.maximum(overlay[..., 3][mask], mask_alpha)

    ax[2].imshow(np.zeros_like(img_np))
    ax[2].imshow(img_np, alpha=image_alpha_for_masks)
    ax[2].set_title("Masks", fontsize=14)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    if (overlay[..., 3].sum() > 0):
        ax[2].imshow(overlay)

    if suptitle:
        fig.suptitle(f"Predictions — {os.path.basename(image_path)}", fontsize=20, y=0.99)
        
    plt.tight_layout()
    plt.show()
