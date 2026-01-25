import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class RCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(RCNN, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(
            weights="DEFAULT",
            trainable_backbone_layers=5
        )

        for param in self.model.parameters():
            param.requires_grad = True
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
    def forward(self, images, targets=None):
        return self.model(images, targets)
    