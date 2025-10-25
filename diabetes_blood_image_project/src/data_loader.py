from medmnist import BloodMNIST
from torchvision import transforms

def load_bloodmnist(train=True):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = BloodMNIST(split='train' if train else 'test', transform=transform, download=True)
    return dataset
