import torch
from modules.model import InstanceSegmentation
from modules.viz import viz_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = InstanceSegmentation()
checkpoint = torch.load(
    f"./models/instance_segmentation_1.pth",
    map_location=device
)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# viz_results("./data/WHU-Building-Dataset/imgs/10002521.TIF", model, device, threshold=0.5)
viz_results("./ignore/exmaples/example1.png", model, device, threshold=0.5)
