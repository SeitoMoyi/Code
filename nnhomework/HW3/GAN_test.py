import os
import sys
import torch
from torch import nn
from torchvision.utils import save_image


batch_size = 256
num_epoch = 100
z_dimension = 100
res_path = sys.argv[1]
dir_path = './GAN_model'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建文件夹
if not os.path.exists(res_path):
    os.mkdir(res_path)


# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (256,100,1,1)
            nn.ConvTranspose2d(z_dimension, 512, 6, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # (256,512,6,6)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # (256,256,12,12)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (256,128,24,24)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (256,64,48,48)
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            # (256,3,96,96)
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


def to_img(x):
    out = 0.5 * (x + 1)
    return out


if __name__ == '__main__':
    try:
        G = torch.load(dir_path + '/G.pth',map_location=device)
        G.eval()
        print('Model loaded. Testing.')
        z = torch.randn(batch_size, z_dimension, 1, 1).to(device)  # 随机噪声
        fake_img = G(z)
        fake_image = to_img(fake_img.cpu())
        save_image(fake_image, res_path + '/GAN_test_img.png')
    except:
        print('[ERROR] Model inexist or loaded failed.')




