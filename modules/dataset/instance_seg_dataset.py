import random
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class WHUBuildingDataset(Dataset):
    def __init__(
        self,
        img_dir="./data/WHU-Building-Dataset/imgs",
        json_file="./data/WHU-Building-Dataset/metadata.json",
        transform=None,
        augment=False,
        flip_prob=0.5,
        rotate_prob=0.5,
        color_jitter_params={"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02},
        seed=None
    ):
        """
        Args:
            transform: a callable applied AFTER augmentations (e.g., normalization, ToTensor)
            augment: whether to apply simple augmentations (keeps masks/boxes consistent)
            flip_prob: probability to apply each horizontal/vertical flip
            rotate_prob: probability to apply a random 90-degree rotation
            color_jitter_params: passed to torchvision.transforms.ColorJitter
        """
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.color_jitter = T.ColorJitter(**color_jitter_params) if color_jitter_params else None

        if seed is not None:
            random.seed(seed)

        with open(json_file, 'r') as f:
            self.metadata = json.load(f)

        # Use image filenames as the primary source
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.tif')]

    def _polygon_to_mask(self, polygon, image_size):
        """
        polygon: list of [x, y] points
        image_size: (width, height)
        returns: numpy array HxW (dtype uint8) with 1 inside polygon
        """
        w_img, h_img = image_size
        mask_im = Image.new('L', (w_img, h_img), 0)
        if polygon and len(polygon) > 2:
            flat_poly = [coord for point in polygon for coord in point]
            ImageDraw.Draw(mask_im).polygon(flat_poly, outline=1, fill=1)
        return np.array(mask_im, dtype=np.uint8)

    def _generate_masks_and_boxes(self, annotations, image_size):
        """
        Given annotations for an image, produce:
         - boxes: list of [xmin, ymin, xmax, ymax]
         - masks: list of numpy arrays (H x W) dtype uint8
        """
        boxes = []
        masks = []
        w_img, h_img = image_size

        for obj in annotations:
            x_min, y_min, w, h = obj['box']
            # skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            boxes.append([x_min, y_min, x_min + w, y_min + h])

            poly_points = obj.get('polygon', [])
            if poly_points and len(poly_points) > 2:
                mask_arr = self._polygon_to_mask(poly_points, (w_img, h_img))
            else:
                # Fallback: produce mask from bbox rectangle
                mask_im = Image.new('L', (w_img, h_img), 0)
                draw = ImageDraw.Draw(mask_im)
                draw.rectangle([x_min, y_min, x_min + w, y_min + h], outline=1, fill=1)
                mask_arr = np.array(mask_im, dtype=np.uint8)

            masks.append(mask_arr)

        return boxes, masks

    def _flip_boxes_horizontally(self, boxes, W):
        flipped = []
        for (xmin, ymin, xmax, ymax) in boxes:
            new_xmin = W - xmax
            new_xmax = W - xmin
            flipped.append([new_xmin, ymin, new_xmax, ymax])
        return flipped

    def _flip_boxes_vertically(self, boxes, H):
        flipped = []
        for (xmin, ymin, xmax, ymax) in boxes:
            new_ymin = H - ymax
            new_ymax = H - ymin
            flipped.append([xmin, new_ymin, xmax, new_ymax])
        return flipped

    def _rotate_boxes_90k(self, boxes, W, H, k):
        if k % 4 == 0:
            return boxes, W, H

        rotated_boxes = []
        for (xmin, ymin, xmax, ymax) in boxes:
            # four corners
            corners = [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax)
            ]
            new_corners = []
            for (x, y) in corners:
                if k % 4 == 1:   # 90 CW
                    x_new = H - 1 - y
                    y_new = x
                elif k % 4 == 2: # 180
                    x_new = W - 1 - x
                    y_new = H - 1 - y
                elif k % 4 == 3: # 270 CW (or 90 CCW)
                    x_new = y
                    y_new = W - 1 - x
                new_corners.append((x_new, y_new))
            xs = [c[0] for c in new_corners]
            ys = [c[1] for c in new_corners]
            rotated_boxes.append([min(xs), min(ys), max(xs), max(ys)])

        # rotated image size swaps when k is odd
        if k % 2 == 1:
            return rotated_boxes, H, W
        else:
            return rotated_boxes, W, H

    def _apply_augmentations(self, image_pil, boxes, masks):
        w_img, h_img = image_pil.size

        # Convert masks to PIL for easy transforms
        mask_pils = [Image.fromarray(m.astype(np.uint8) * 255).convert('L') for m in masks]

        # Random horizontal flip
        if random.random() < self.flip_prob:
            image_pil = ImageOps.mirror(image_pil)
            mask_pils = [ImageOps.mirror(m) for m in mask_pils]
            boxes = self._flip_boxes_horizontally(boxes, w_img)

        # Random vertical flip
        if random.random() < self.flip_prob:
            image_pil = ImageOps.flip(image_pil)
            mask_pils = [ImageOps.flip(m) for m in mask_pils]
            boxes = self._flip_boxes_vertically(boxes, h_img)

        # Random 90-degree rotation (with some probability)
        if random.random() < self.rotate_prob:
            # choose k in {1,2,3} (k*90 deg clockwise)
            k = random.choice([1, 2, 3])
            # PIL transpose operations: ROTATE_90 rotates 90 degrees CCW, so map accordingly
            if k == 1:   # 90 CW -> PIL ROTATE_270
                image_pil = image_pil.transpose(Image.ROTATE_270)
                mask_pils = [m.transpose(Image.ROTATE_270) for m in mask_pils]
            elif k == 2:
                image_pil = image_pil.transpose(Image.ROTATE_180)
                mask_pils = [m.transpose(Image.ROTATE_180) for m in mask_pils]
            elif k == 3:
                image_pil = image_pil.transpose(Image.ROTATE_90)
                mask_pils = [m.transpose(Image.ROTATE_90) for m in mask_pils]

            boxes, new_W, new_H = self._rotate_boxes_90k(boxes, w_img, h_img, k)
            w_img, h_img = new_W, new_H  # update for possible further ops

        # Color jitter (image-only)
        if self.color_jitter is not None:
            image_pil = self.color_jitter(image_pil)

        # Convert masks back to numpy arrays of 0/1
        masks = [(np.array(m_pil, dtype=np.uint8) > 127).astype(np.uint8) for m_pil in mask_pils]

        return image_pil, boxes, masks

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_id = os.path.splitext(img_filename)[0]
        img_path = os.path.join(self.img_dir, img_filename)

        # Robust image loading (Ensures 3 channels for parallelism/normalization)
        image_pil = Image.open(img_path).convert("RGB")
        w_img, h_img = image_pil.size

        boxes = []
        masks = []

        # Check if this image has entries in metadata
        if img_id in self.metadata:
            annotations = self.metadata[img_id]
            boxes, masks = self._generate_masks_and_boxes(annotations, (w_img, h_img))

        num_objs = len(boxes)

        if num_objs > 0:
            # Possibly apply augmentations that keep everything in sync
            if self.augment:
                image_pil, boxes, masks = self._apply_augmentations(image_pil, boxes, masks)

            # Convert boxes, masks to tensors
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
            masks_tensor = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8) if len(masks) > 0 else torch.zeros((0, h_img, w_img), dtype=torch.uint8)
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        else:
            # Handle images with no buildings
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            masks_tensor = torch.zeros((0, h_img, w_img), dtype=torch.uint8)
            area = torch.as_tensor([], dtype=torch.float32)

        labels = torch.ones((num_objs,), dtype=torch.int64) 
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        image_id_tensor = torch.tensor([idx])

        target = {
            "boxes": boxes_tensor,
            "labels": labels,
            "masks": masks_tensor,
            "image_id": image_id_tensor,
            "area": area,
            "iscrowd": iscrowd
        }

        # Final image transform (user-specified), else fallback to tensor [C,H,W] floats in [0,1]
        if self.transform:
            # if user transform expects both image and target, they can handle; else we pass image only
            image = self.transform(image_pil)
        else:
            image = TF.to_tensor(image_pil)

        return image, target
