import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class RCNN(nn.Module):
    def __init__(self, num_classes=2): # 1 class + background
        super(RCNN, self).__init__()
        # Load Pretrained Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        
        # Replace the classifier head for our number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
    def forward(self, images, targets=None):
        # Returns losses during training, detections during eval
        return self.model(images, targets)
    