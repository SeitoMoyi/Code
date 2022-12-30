import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2304, 64),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i == 0:
                break
        return x