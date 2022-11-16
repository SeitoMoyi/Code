import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1
torch.save(vgg16, "vgg16_method.pth")

# 保存方式2(官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 陷阱
class NM(nn.Module):
    def __init__(self):
        super(NM, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self,x):
        x = self.conv(x)
        return x

nm = NM()
torch.save(nm, "nm_method.pth")