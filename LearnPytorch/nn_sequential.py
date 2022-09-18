import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./cifar10')

class NM(nn.Module):
    def __init__(self):
        super(NM, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

nm = NM()
print(nm)

input = torch.ones((64,3,32,32))
output = nm(input)
print(output.shape)

writer = SummaryWriter("logs_seq")
writer.add_graph(nm,input)
writer.close()