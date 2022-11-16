import torch
import torchvision.models
# 加载方式1
from torch import nn
from model_save import *
model = torch.load("vgg16_method.pth")
# print(model)

# 加载方式2
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)

# 陷阱1
# class NM(nn.Module):
#     def __init__(self):
#         super(NM, self).__init__()
#         self.conv = nn.Conv2d(3,64,kernel_size=3)
#
#     def forward(self,x):
#         x = self.conv(x)
#         return x

model = torch.load("nm_method.pth")
print(model)
