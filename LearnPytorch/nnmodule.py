import torch
from torch import nn


class NnModule(nn.Module):
    def __init__(self):
        super(NnModule, self).__init__()

    def forward(self,input):
        output = input + 1
        return output

nnmodule = NnModule()
x = torch.tensor(1.0)
output = nnmodule(x)
print(output)