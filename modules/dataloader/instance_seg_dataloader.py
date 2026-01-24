from torch.utils.data import DataLoader
import torchvision.transforms as T
from modules.dataset.instance_seg_dataset import InstanceSegDataset


class InstanceSegDataloader(DataLoader):
    def __init__(self, root_dir, batch_size=2, *args, **kwargs):
        transform = T.Compose([T.ToTensor()])

        dataset = InstanceSegDataset(root_dir, transforms=transform)

        collate_fn = lambda batch: tuple(zip(*batch))

        super().__init__(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, *args, **kwargs)
        