import cv2
import numpy as np
import json
from PIL import Image
from tqdm import tqdm


def polygons_to_mask(polygons, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)

    for poly in polygons:
        poly = np.asarray(poly, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 1)

    return mask


def generate_masks(dataset_path: str="./data/WHU-Building-Dataset"):
    with open(f"{dataset_path}/metadata.json") as f:
        metadata = json.load(f)

    for img_id in tqdm(metadata.keys()):
        polygons = [segment["polygon"] for segment in metadata[img_id]]
        mask = polygons_to_mask(polygons, 1024, 1024)
        Image.fromarray(mask * 255).save(f"{dataset_path}/masks/{img_id}.png")


if __name__ == "__main__":
    generate_masks()
