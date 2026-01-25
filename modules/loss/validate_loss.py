import torch
from tqdm import tqdm


def validate_loss(dataloader, model, device):
    model.train()
    val_running_loss = 0.0
    steps = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating", leave=False):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss = model(images, targets)
                
            val_running_loss += loss.item()
            steps += 1
            
    return val_running_loss / steps
