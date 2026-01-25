from torch.utils.data import DataLoader
import torchvision.transforms as T
from modules.dataset.instance_seg_dataset import InstanceSegDataset


class InstanceSegDataloader(DataLoader):
    def __init__(self, dataset: InstanceSegDataset, batch_size=2, shuffle=True, num_workers=2, **kwargs):
        def collate_fn(batch):
            return tuple(zip(*batch))

        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=collate_fn, 
            num_workers=num_workers,
            **kwargs
        )
        