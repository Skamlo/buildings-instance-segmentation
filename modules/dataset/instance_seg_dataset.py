import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class InstanceSegDataset(Dataset):
    def __init__(self, images_path, masks_path, metadata_path, transforms=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transforms = transforms

        # Load metadata JSON
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Get file list and ensure we only keep files that exist in metadata
        all_imgs = sorted(os.listdir(self.images_path))
        self.imgs = []
        for img_name in all_imgs:
            img_id = os.path.splitext(img_name)[0]
            if img_id in self.metadata:
                self.imgs.append(img_name)
        
        # Assume masks follow the same naming convention as images
        self.masks = [img_name.replace(os.path.splitext(img_name)[1], ".png") for img_name in self.imgs]

    def __getitem__(self, idx):
        # 1. Identify IDs and Paths
        img_name = self.imgs[idx]
        img_id = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(self.images_path, img_name)
        mask_path = os.path.join(self.masks_path, self.masks[idx])
        
        # 2. Load Image and Instance Mask
        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path)) # 0=BG, 1=Obj1, 2=Obj2...

        # 3. Pull Boxes from Metadata
        # Format in JSON: [xmin, ymin, xmax, ymax]
        meta_objects = self.metadata[img_id]
        boxes = []
        for obj in meta_objects:
            boxes.append(obj["box"])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # 4. Process Masks
        # We split the multi-value mask into a binary stack (N, H, W)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:] # Remove background 0
        
        # Create binary masks for each instance
        masks = mask == obj_ids[:, None, None]
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # 5. Labels (Always 1 for "Building")
        num_objs = len(meta_objects)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        image_id = torch.tensor([idx])

        # 6. Final Target Dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
