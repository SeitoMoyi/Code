import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],[-1,3]],dtype=torch.float32)
input = torch.reshape(input,[-1,1,2,2])
print(input.shape)

dataset = torchvision.datasets.CIFAR10("./cifar10",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

class NM(nn.Module):
    def __init__(self):
        super(NM, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = Sigmoid()
    def forward(self,input):
        output = self.sigmoid(input)
        return output

nm = NM()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("sigmoid_input", imgs, step)
    output = nm(imgs)
    writer.add_images("sigmoid_output",output,step)
    step += 1

writer.close()