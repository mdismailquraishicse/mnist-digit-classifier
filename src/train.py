import torch
import torch.nn as nn
from data_loader import MnistDataLoader
from mnist_digit_classifier import DigitClassifier


class Trainer:


    def __init__(self, learning_rate:float):
        
        self.model = DigitClassifier()
        self.data_loader = MnistDataLoader()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)


    def train(self, epochs:int):

        train_loader = self.data_loader.train_loader
        for epoch in range(epochs):
            total_loss = 0
            self.model.train()
            for images, labels in train_loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch: {epoch+1} loss: {total_loss/len(train_loader)}")