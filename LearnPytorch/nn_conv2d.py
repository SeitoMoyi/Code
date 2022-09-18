import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./cifar10",train=False, transform= torchvision.transforms.ToTensor(),download= True)
dataloader = DataLoader(dataset, batch_size=64)

class NM(nn.Module):
    def __init__(self):
        super(NM, self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6, kernel_size=(3,3))

    def forward(self,x):
        x = self.conv1(x)
        return x

nm = NM()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs,targets = data
    output = nm(imgs)
    print(output.shape)
    print(imgs.shape)
    writer.add_images("input", imgs, step)

    output = torch.reshape(output,[-1,3,30,30])
    writer.add_images("output", output, step)
    step  += 1

