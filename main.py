import numpy as np
import torch
from modules.model.model import InstanceSegmentation

# Initialize
model = InstanceSegmentation()

# Dummy Input
dummy_img = torch.rand(2, 3, 600, 600) # Batch size 2

# Dummy Targets (needed for training mode)
dummy_targets = [
    {
        'boxes': torch.tensor([[50, 50, 100, 100]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64),
        'masks': torch.randint(0, 2, (1, 600, 600), dtype=torch.uint8)
    },
    {
        'boxes': torch.tensor([[150, 150, 200, 200]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64),
        'masks': torch.randint(0, 2, (1, 600, 600), dtype=torch.uint8)
    }
]

# Test Forward Pass (Training)
model.train()
loss = model(dummy_img, dummy_targets)
print(f"Training Loss: {loss.item()}")

# Test Forward Pass (Inference)
model.eval()
with torch.no_grad():
    output = model(dummy_img)
    print("Inference Output Boxes:", output[0]['boxes'])
    print("Inference Output Masks Shape:", output[0]['masks'].shape)
