from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import json
from modules.dataloader.instance_seg_dataloader import InstanceSegDataloader
from modules.dataset.instance_seg_dataset import InstanceSegDataset
from modules.transforms.transforms import get_transforms
from modules.model.model import InstanceSegmentation
from modules.loss.validate_loss import validate_loss

# Parameters
NUM_EPOCHS = 3
BATCH_SIZE = 6
TEST_SIZE = 0.1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0005

# Paths
IMG_DIR = "./data/WHU-Building-Dataset/imgs"
MASK_DIR = "./data/WHU-Building-Dataset/masks"
META_PATH = "./data/WHU-Building-Dataset/metadata.json"
MODELS_DIR = "./models"

image_paths = list(sorted(os.listdir(IMG_DIR)))
image_paths = [f"{IMG_DIR}/{image_name}" for image_name in image_paths]
masks_paths = list(sorted(os.listdir(MASK_DIR)))
masks_paths = [f"{MASK_DIR}/{mask_name}" for mask_name in masks_paths]

image_paths_train, image_paths_test, masks_paths_train, masks_paths_test = train_test_split(
    image_paths,
    masks_paths,
    test_size=TEST_SIZE,
    random_state=42,
    shuffle=True
)

# Datasets
dataset_train = InstanceSegDataset(
    images_paths=image_paths_train,
    masks_paths=masks_paths_train,
    metadata_path=META_PATH,
    transforms=get_transforms()
)
dataset_test = InstanceSegDataset(
    images_paths=image_paths_test,
    masks_paths=masks_paths_test,
    metadata_path=META_PATH,
    transforms=get_transforms()
)

# Dataloaders
dataloader_train = InstanceSegDataloader(dataset_train, batch_size=BATCH_SIZE)
dataloader_test = InstanceSegDataloader(dataset_test, batch_size=BATCH_SIZE)

# Model
model = InstanceSegmentation()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Traning loop
history = {
    "train_batch_loss": [], # Loss for every single batch
    "train_epoch_loss": [], # Average loss per epoch
    "val_epoch_loss": []    # Average validation loss per epoch
}

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    steps = 0
    
    # Progress bar for training
    train_loop = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for images, targets in train_loop:
        # 1. Move data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 2. Zero Gradients
        optimizer.zero_grad()

        # 3. Forward Pass
        # Returns total_loss (Detector Loss + UNet Loss)
        loss = model(images, targets)

        # Handle DataParallel output (it returns a vector of losses, one per GPU)
        if loss.ndim > 0:
            loss = loss.mean()

        # 4. Backward Pass
        loss.backward()
        optimizer.step()

        # 5. Logging
        loss_value = loss.item()
        history["train_batch_loss"].append(loss_value)
        epoch_loss += loss_value
        steps += 1
        
        # Update progress bar
        train_loop.set_postfix(loss=loss_value)

    # Calculate average training loss for this epoch
    avg_train_loss = epoch_loss / steps
    history["train_epoch_loss"].append(avg_train_loss)
    
    # -----------------------------------------------------
    # TESTING (Validation) AFTER EACH EPOCH
    # -----------------------------------------------------
    avg_val_loss = validate_loss(dataloader_test, model, device)
    history["val_epoch_loss"].append(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}")

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------
    # Save best model or every epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss,
    }, f"{MODELS_DIR}/model_epoch_{epoch+1}.pth")

print("Training Complete.")


# Save history
with open("./history.json", "w") as f:
    json.dump(history, f, indent=4)
