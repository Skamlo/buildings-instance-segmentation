import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.model.object_detection import RCNN
from modules.model.segmentation import UNet


class InstanceSegmentation(nn.Module):
    def __init__(self, crop_size=256):
        super().__init__()
        self.detector = RCNN()
        self.segmentor = UNet()
        self.crop_size = crop_size
        
    def forward(self, images, targets=None):
        """
        Two modes:
        1. Training: We usually train Detector and Segmentor somewhat separately.
           However, here is a pipeline logic.
        2. Inference: Run Detector -> Crop Boxes -> Run Segmentor on Crops.
        """
        
        # --- 1. Run Detector ---
        # During training, faster-rcnn returns losses automatically
        if self.training and targets is not None:
            det_loss = self.detector(images, targets)
            
            # For the UNet training, we need to extract crops based on Ground Truth boxes
            # to ensure the UNet learns to segment correctly aligned objects.
            unet_loss = self._train_step_unet(images, targets)
            
            # Combine losses
            total_loss = sum(loss for loss in det_loss.values()) + unet_loss
            return total_loss
            
        else:
            # --- Inference Mode ---
            detections = self.detector(images)
            
            final_results = []
            
            for i, det in enumerate(detections):
                boxes = det['boxes']
                scores = det['scores']
                
                # Filter by confidence
                keep = scores > 0.5
                boxes = boxes[keep]
                
                if len(boxes) == 0:
                    final_results.append({'boxes': boxes, 'masks': []})
                    continue

                # Crop images based on boxes
                crops = self._crop_and_resize(images[i], boxes)
                
                # Predict masks
                mask_logits = self.segmentor(crops)
                masks = torch.sigmoid(mask_logits)
                
                final_results.append({'boxes': boxes, 'masks': masks})
                
            return final_results

    def _crop_and_resize(self, image, boxes):
        """
        Crops the image according to boxes and resizes to self.crop_size.
        Image: (C, H, W)
        Boxes: (N, 4)
        """
        crops = []
        # Ensure image is 3D (C, H, W)
        if len(image.shape) != 3:
             raise ValueError(f"Expected image of shape (C, H, W), got {image.shape}")

        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            
            # Clamp coordinates to image dimensions
            h_img, w_img = image.shape[1], image.shape[2]
            x1 = max(0, min(x1, w_img))
            y1 = max(0, min(y1, h_img))
            x2 = max(0, min(x2, w_img))
            y2 = max(0, min(y2, h_img))
            
            # Check for invalid box (width or height is 0)
            if x2 <= x1 or y2 <= y1:
                # Create a placeholder zero tensor if box is invalid
                crop = torch.zeros((image.shape[0], self.crop_size, self.crop_size), device=image.device)
                crops.append(crop.unsqueeze(0))
                continue

            # Crop
            crop = image[:, y1:y2, x1:x2]
            
            # Resize
            # interpolate expects (N, C, H, W), so we unsqueeze(0)
            crop = F.interpolate(crop.unsqueeze(0), size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
            crops.append(crop)
            
        if len(crops) > 0:
            return torch.cat(crops, dim=0) # Returns (N, C, 256, 256)
        return torch.empty((0, image.shape[0], self.crop_size, self.crop_size), device=image.device)

    def _train_step_unet(self, images, targets):
        """
        Helper to calculate U-Net loss using Ground Truth boxes.
        Fixes the dimension error by cropping masks 1-to-1.
        """
        all_img_crops = []
        all_mask_crops = []
        
        for i, t in enumerate(targets):
            boxes = t['boxes']
            masks = t['masks'] # Shape (Num_Objs, H, W)
            img = images[i]    # Shape (C, H, W)
            
            # 1. Crop Images (One image, multiple boxes)
            # This works fine with the standard helper
            img_crops = self._crop_and_resize(img, boxes)
            
            # 2. Crop Masks (One mask per box)
            # We cannot use _crop_and_resize here because that applies ALL boxes to ONE image.
            # We need to apply Box[k] to Mask[k].
            current_mask_crops = []
            for k, box in enumerate(boxes):
                x1, y1, x2, y2 = box.int().tolist()
                
                # Select the specific mask for this object
                mask = masks[k].unsqueeze(0).float() # (1, H, W)
                
                # Clamp
                h_img, w_img = mask.shape[1], mask.shape[2]
                x1 = max(0, min(x1, w_img))
                y1 = max(0, min(y1, h_img))
                x2 = max(0, min(x2, w_img))
                y2 = max(0, min(y2, h_img))

                if x2 <= x1 or y2 <= y1:
                    cropped_mask = torch.zeros((1, self.crop_size, self.crop_size), device=mask.device)
                else:
                    cropped_mask = mask[:, y1:y2, x1:x2]
                    cropped_mask = F.interpolate(cropped_mask.unsqueeze(0), size=(self.crop_size, self.crop_size), mode='nearest')
                    # Remove batch dim added for interpolate -> (1, 256, 256)
                    cropped_mask = cropped_mask.squeeze(0)
                
                current_mask_crops.append(cropped_mask)
            
            if len(current_mask_crops) > 0:
                mask_crops_tensor = torch.stack(current_mask_crops) # (Num_Objs, 1, 256, 256)
                
                all_img_crops.append(img_crops)
                all_mask_crops.append(mask_crops_tensor)
        
        if len(all_img_crops) > 0:
            batch_crops = torch.cat(all_img_crops, dim=0)       # (Total_N, C, 256, 256)
            batch_gt_masks = torch.cat(all_mask_crops, dim=0)   # (Total_N, 1, 256, 256)
            
            # Forward pass U-Net
            pred_masks = self.segmentor(batch_crops)
            
            # Binary Cross Entropy Loss
            return F.binary_cross_entropy_with_logits(pred_masks, batch_gt_masks)
        else:
            return torch.tensor(0.0, device=images[0].device, requires_grad=True)
