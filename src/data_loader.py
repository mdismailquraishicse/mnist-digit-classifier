from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MnistDataLoader:


    def __init__(self, batch_size:int=64):
        
        self.transform = transforms.ToTensor()

        train_datasets = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=self.transform
        )

        test_datasets = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=self.transform
        )

        self.train_loader = DataLoader(
            dataset=train_datasets,
            batch_size=batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            dataset=test_datasets,
            batch_size=batch_size,
            shuffle=False
        )