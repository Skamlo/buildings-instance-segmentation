import json
import torch
from modules.dataloader import get_dataloaders
from modules.model import InstanceSegmentation
from modules.train import train

# Device Configuration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# Paths
IMG_DIR = './data/WHU-Building-Dataset/imgs' 
JSON_FILE = './data/WHU-Building-Dataset/metadata.json'

# Hyperparameters
NUM_CLASSES = 2
BATCH_SIZE = 3
NUM_EPOCHS = 1
LEARNING_RATE = 0.005

# Dataloaders
train_loader, val_loader, test_loader = get_dataloaders(IMG_DIR, JSON_FILE, BATCH_SIZE)

# Model
model = InstanceSegmentation(num_classes=NUM_CLASSES)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

# Train
history = train(model, optimizer, lr_scheduler, train_loader, val_loader, device, NUM_EPOCHS)

# Save history
with open("./models/history.json") as f:
    json.dump(history, f, indent=4)
