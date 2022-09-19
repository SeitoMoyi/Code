import torch
from torch import nn


class NM(nn.Module):
    def __init__(self):
        super(NM, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        input = self.model(input)
        return input


if __name__ == "__main__":
    nm = NM()
    input = torch.ones(64,3,32,32)
    output = nm(input)
    print(output.shape)