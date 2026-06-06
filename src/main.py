import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_digit_classifier import DigitClassifier


epoches = 5

def main():

    print("Hello from mnist-digit-classifier!")
    transform = transforms.ToTensor()
    model = DigitClassifier()

    train_datasets = datasets.MNIST(
        root = "./data",
        train = True,
        download = True,
        transform = transform
    )

    test_datasets = datasets.MNIST(
        root = "./data",
        train = False,
        download = True,
        transform = transform
    )

    train_loader = DataLoader(
        dataset = train_datasets,
        shuffle = True,
        batch_size = 64
    )

    test_loader = DataLoader(
        dataset = test_datasets,
        shuffle = False,
        batch_size = 64
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(epoches):
        model.train(mode = True)
        total_loss = 0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch + 1} loss: {total_loss/len(train_loader):.4f}")


if __name__ == "__main__":
    main()
