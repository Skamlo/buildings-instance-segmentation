from typing import Tuple
import torch
import torchvision
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from modules.dataset import WHUBuildingDataset
from modules.transformer import get_transformer
from modules.utils import collate_fn


def get_dataloaders(images_path: str, metadata_path: str, batch_size: str=4, proportions: Tuple[float]=(0.8, 0.1, 0.1)):
    if sum(proportions) != 1:
        raise ValueError("Sum of `proportions` value should be equal 1.")

    full_dataset = WHUBuildingDataset(
        images_path,
        metadata_path,
        transform=get_transformer()
    )

    # Calculate split sizes
    train_size = int(proportions[0] * len(full_dataset))
    val_size = int(proportions[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
