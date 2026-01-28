import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
def visualize_prediction(image_tensor, prediction, threshold=0.5):
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    overlay = img.copy()
    
    boxes = prediction['boxes'].cpu().numpy()
    masks = prediction['masks'].cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    colors = plt.cm.get_cmap('tab10')

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        
        mask_crop = masks[i, 0]        
        box_h = y2 - y1
        box_w = x2 - x1
        
        if box_h <= 0 or box_w <= 0: continue
        mask_resized = cv2.resize(mask_crop, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
        
        # Binarize mask
        mask_binary = mask_resized > threshold
        
        # --- 3. Apply Color Overlay ---
        color = colors(i % 10)[:3] # Get RGB tuple
        
        # Apply color to the region defined by the mask and box
        # We clamp coordinates to ensure we don't go out of image bounds
        y_start, y_end = max(0, y1), min(img.shape[0], y2)
        x_start, x_end = max(0, x1), min(img.shape[1], x2)
        
        # Adjust mask slice if box was clipped at image borders
        mask_h_slice = slice(y_start - y1, (y_start - y1) + (y_end - y_start))
        mask_w_slice = slice(x_start - x1, (x_start - x1) + (x_end - x_start))
        
        region = overlay[y_start:y_end, x_start:x_end]
        mask_slice = mask_binary[mask_h_slice, mask_w_slice]
        
        # Blend: 0.6 * original + 0.4 * color
        region[mask_slice] = region[mask_slice] * 0.6 + np.array(color) * 0.4
        overlay[y_start:y_end, x_start:x_end] = region

        # --- 4. Draw Bounding Box ---
        rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label/score if available (optional)
        ax.text(x1, y1, f'Obj {i}', color='white', fontsize=10, backgroundcolor=color)

    ax.imshow(overlay)
    ax.axis('off')
    plt.title("Instance Segmentation Result")
    fig.canvas.draw()
    
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data