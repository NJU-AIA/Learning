from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader

def get_loader(fashion=False, batch_size=128, root="../../assets"):
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)
    ])
    dataset_cls = FashionMNIST if fashion else MNIST
    dataset = dataset_cls(root, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
