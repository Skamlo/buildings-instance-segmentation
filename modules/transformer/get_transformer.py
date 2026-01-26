import torchvision


def get_transformer():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
