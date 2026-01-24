import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Encoder (Pretrained ResNet18) ---
        # We assume the input to the UNet is a resized crop (e.g. 128x128 or 256x256)
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # We slice ResNet to get intermediate feature maps for skip connections
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64 channels
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)       # 64 channels
        self.enc3 = resnet.layer2                                      # 128 channels
        self.enc4 = resnet.layer3                                      # 256 channels
        self.enc5 = resnet.layer4                                      # 512 channels

        # --- Decoder ---
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(512, 256) # 256 from up1 + 256 from enc4
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128) # 128 from up2 + 128 from enc3
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)  # 64 from up3 + 64 from enc2
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)  # 64 from up4 + 64 from enc1
        
        # Final output layer (upsample back to original resolution if needed, here we keep 1:1 with input)
        # Note: ResNet first conv is stride 2, so we need one more upsample to match input size
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x) # Size/2
        x2 = self.enc2(x1) # Size/4
        x3 = self.enc3(x2) # Size/8
        x4 = self.enc4(x3) # Size/16
        x5 = self.enc5(x4) # Size/32

        # Decoder
        d1 = self.up1(x5)
        # Resize if rounding errors in dimensions
        if d1.size() != x4.size(): d1 = F.interpolate(d1, size=x4.shape[2:])
        d1 = torch.cat([x4, d1], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        if d2.size() != x3.size(): d2 = F.interpolate(d2, size=x3.shape[2:])
        d2 = torch.cat([x3, d2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        if d3.size() != x2.size(): d3 = F.interpolate(d3, size=x2.shape[2:])
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        if d4.size() != x1.size(): d4 = F.interpolate(d4, size=x1.shape[2:])
        d4 = torch.cat([x1, d4], dim=1)
        d4 = self.dec4(d4)

        out = self.final_up(d4)
        out = self.final_conv(out)
        
        return out
