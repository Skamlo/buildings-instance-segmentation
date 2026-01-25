import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class InstanceSegDataset(Dataset):
    def __init__(self, images_paths, masks_paths, metadata_path, transforms=None):
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.transforms = transforms
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]

        # 1. Load Image and Global Semantic Mask
        img = Image.open(img_path).convert("RGB")
        semantic_mask = np.array(Image.open(mask_path))
        if semantic_mask.max() > 1:
            semantic_mask = (semantic_mask > 127).astype(np.float32)

        final_boxes = []
        final_masks = []
        
        # 2. Extract Boxes (Handle images not in metadata)
        meta_objects = self.metadata.get(img_id, [])
        
        for obj in meta_objects:
            x, y, w, h = obj["box"]
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            
            instance_mask = np.zeros_like(semantic_mask)
            ix1, iy1, ix2, iy2 = int(xmin), int(ymin), int(xmax), int(ymax)
            
            h_img, w_img = semantic_mask.shape
            ix1, ix2 = max(0, ix1), min(w_img, ix2)
            iy1, iy2 = max(0, iy1), min(h_img, iy2)
            
            instance_mask[iy1:iy2, ix1:ix2] = semantic_mask[iy1:iy2, ix1:ix2]
            
            final_boxes.append([xmin, ymin, xmax, ymax])
            final_masks.append(instance_mask)

        # 3. Convert to Tensors
        if len(final_masks) > 0:
            masks = torch.as_tensor(np.stack(final_masks), dtype=torch.float32)
            boxes = torch.as_tensor(final_boxes, dtype=torch.float32)
            labels = torch.ones((len(final_boxes),), dtype=torch.int64)
        else:
            masks = torch.zeros((0, semantic_mask.shape[0], semantic_mask.shape[1]), dtype=torch.float32)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        if self.transforms:
            img = self.transforms(img)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks, 
            "image_id": torch.tensor([idx])
        }

        return img, target

    def __len__(self):
        return len(self.images_paths)
