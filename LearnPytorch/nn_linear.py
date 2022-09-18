import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./cifar10",train = False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64,drop_last=True)

class NM(nn.Module):
    def __init__(self):
        super(NM, self).__init__()
        self.linear = Linear(196608,10)

    def forward(self,input):
        output = self.linear(input)
        return output

nm = NM()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = nm(output)
    print(output.shape)