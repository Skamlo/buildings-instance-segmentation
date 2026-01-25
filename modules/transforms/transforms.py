import torchvision.transforms as T

def get_transforms():
    return T.Compose([
        T.ToTensor(),
    ])
