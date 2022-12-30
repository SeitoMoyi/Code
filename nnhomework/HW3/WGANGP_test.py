import os
import sys
import torch
from torch import nn
from torchvision.utils import save_image


batch_size = 256
num_epoch = 100
z_dimension = 100
res_path = sys.argv[1]
dir_path = './WGANGP_model'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建文件夹
if not os.path.exists(res_path):
    os.mkdir(res_path)


# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            # (256,100,1,1)
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # (256,512,4,4)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # (256,256,8,8)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (256,128,16,16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (256,64,32,32)
            nn.ConvTranspose2d(64, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # (256,3,96,96)
        )

    def forward(self, x):
        x = self.generator(x)
        return x

    def weight_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data.normal_(0, 0.02)
        elif class_name.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)


def to_img(x):
    out = 0.5 * (x + 1)
    return out


if __name__ == '__main__':
    try:
        G = torch.load(dir_path + '/G.pth', map_location=device)
        G.eval()
        print('Model loaded. Testing.')
        z = torch.randn(batch_size, z_dimension, 1, 1).to(device)  # 随机噪声
        fake_img = G(z)
        fake_image = to_img(fake_img.cpu())
        save_image(fake_image, res_path + '/WGANGP_test_img.png')
    except:
        print('[ERROR] Model inexist or loaded failed.')

