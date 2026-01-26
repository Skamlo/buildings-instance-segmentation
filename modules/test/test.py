import numpy as np
from tqdm import tqdm
import torch
from modules.train.metrics import box_iou, mask_iou


def test(model, test_loader, device):
    model.eval()

    all_mask_ious = []
    all_recalls = []
    total_images = 0
    images_with_detections = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for pred, tgt in zip(predictions, targets):
                total_images += 1

                gt_boxes = tgt["boxes"].cpu()
                gt_masks = tgt["masks"].cpu()

                pred_boxes = pred["boxes"].cpu()
                pred_masks = (pred["masks"] > 0.5).cpu()

                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    all_recalls.append(0.0)
                    continue

                images_with_detections += 1

                # Recall@0.5
                matched_gt = set()
                hits = 0

                for pb in pred_boxes:
                    for i, gb in enumerate(gt_boxes):
                        if i in matched_gt:
                            continue
                        if box_iou(pb.tolist(), gb.tolist()) >= 0.5:
                            matched_gt.add(i)
                            hits += 1
                            break

                recall = hits / len(gt_boxes)
                all_recalls.append(recall)

                # Mask IoU
                for i, gt_mask in enumerate(gt_masks):
                    best_iou = 0.0
                    for pm in pred_masks:
                        iou = mask_iou(
                            pm.squeeze(0).byte(),
                            gt_mask.byte()
                        ).item()
                        best_iou = max(best_iou, iou)
                    all_mask_ious.append(best_iou)

    results = {
        "mean_mask_iou": float(np.mean(all_mask_ious)) if all_mask_ious else 0.0,
        "recall_50": float(np.mean(all_recalls)) if all_recalls else 0.0,
        "images_evaluated": total_images,
        "images_with_predictions": images_with_detections
    }

    return results
