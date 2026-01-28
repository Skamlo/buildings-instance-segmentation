import os
import random
import colorsys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from modules.transformer import get_transformer


def random_color() -> str:
    r, g, b = colorsys.hls_to_rgb(random.random(), 0.5, 1)
    r = int(r*255)
    g = int(g*255)
    b = int(b*255)
    return r, g, b
    

def rgb_to_hex(r, g, b):
    r, g, b = [max(0, min(255, int(x))) for x in (r, g, b)]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def viz_results(image_path, model, device, threshold=0.5,
                include_masks=True, include_boxes=True, include_image=True):
    """
    Visualizes Mask R-CNN predictions on an image using an already loaded model.
    
    Args:
        image_path (str): Path to the .TIF or .png image.
        model (nn.Module): The loaded Mask R-CNN model.
        device (torch.device): Device to run inference on.
        threshold (float): Confidence score threshold for displaying objects.
    """
    # 1. Prepare Model for Inference
    model.eval()
    model.to(device)

    # 2. Prepare Image
    # Using your robust loading logic to ensure 3 channels
    img_pil = Image.open(image_path).convert("RGB")
    transform = get_transformer()
    img_tensor = transform(img_pil).to(device)

    # 3. Inference
    with torch.no_grad():
        # Mask R-CNN expects a list of tensors
        prediction = model([img_tensor])

    # 4. Process Results
    pred = prediction[0]
    # Move to CPU for plotting
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    masks = pred['masks'].cpu().numpy()

    # Filter indices based on score threshold
    mask_indices = np.where(scores > threshold)[0]
    
    # 5. Plotting
    fig, ax = plt.subplots(1, figsize=(12, 12))
    if include_image:
        ax.imshow(img_pil)
    else:
        ax.imshow(np.zeros_like(img_pil))

    for i in mask_indices:
        color = random_color()
        
        # Draw Bounding Box
        if include_boxes:
            box = boxes[i]
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                linewidth=2, edgecolor=rgb_to_hex(*color), facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.text(box[0], box[1]-5, f"Building: {scores[i]:.2f}", 
                    color='black', weight='bold', backgroundcolor=rgb_to_hex(*color), size=8)

        # Draw Mask Overlay
        if include_masks:
            alpha = 0.4 if include_image else 1.0

            mask = masks[i, 0] > 0.5
            color_mask = np.zeros((*mask.shape, 4)) 
            color_mask[mask] = [color[0]/255, color[1]/255, color[2]/255, alpha]
            ax.imshow(color_mask)

    plt.axis('off')
    plt.title(f"Predictions for {os.path.basename(image_path)}")
    plt.show()
