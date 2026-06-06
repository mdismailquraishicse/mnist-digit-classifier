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


    def evaluate(self):

        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        test_loader = self.data_loader.test_loader

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f} Test accuracy: {accuracy:.4f}")