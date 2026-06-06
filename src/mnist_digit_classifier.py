import torch.nn as nn


class DigitClassifier(nn.Module):


    def __init__(self):

        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )


    def forward(self, X):

        return self.model(X)