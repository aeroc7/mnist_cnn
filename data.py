from torchvision import datasets
from torch.utils.data import DataLoader


class TrainData():
    def __init__(self, directory, transforms, batch_size):
        train_data = datasets.MNIST(
            root=directory, train=True, transform=transforms, download=True)
        test_data = datasets.MNIST(
            root=directory, train=False, transform=transforms, download=True)
        self.train_loader = DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(
            dataset=test_data, batch_size=batch_size, shuffle=True)

    def train_data(self):
        return self.train_loader

    def test_data(self):
        return self.test_loader
