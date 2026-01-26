import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class InstanceSegmentation(nn.Module):
    def __init__(self, num_classes=2):
        super(InstanceSegmentation, self).__init__()
        
        # 1. Load Pretrained Backbone (ResNet50)
        backbone_base = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone_features = nn.Sequential(*list(backbone_base.children())[:-2])
        self.backbone_features.out_channels = 2048
        
        # 2. Define Anchor Generator
        anchor_sizes = ((32, 64, 128, 256, 512),) 
        aspect_ratios = ((0.5, 1.0, 2.0),)
        rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        
        # 3. Define ROI Pooler for Boxes
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # 4. Define ROI Pooler for Masks (New for Mask R-CNN)
        # Masks usually require higher resolution features (e.g., 14x14)
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=14,
            sampling_ratio=2
        )
        
        # 5. Assemble Mask R-CNN
        self.model = MaskRCNN(
            self.backbone_features,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=box_roi_pool,
            mask_roi_pool=mask_roi_pool
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)
