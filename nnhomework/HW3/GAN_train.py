import os
import sys
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm


batch_size = 256
num_epoch = 100
z_dimension = 100
res_path = './GAN_train_results'
dir_path = './GAN_model'
data_path = 'faces'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建文件夹
if not os.path.exists(res_path):
    os.mkdir(res_path)
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

# 数据集
dataset = ImageFolder(data_path, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 使用正态分布初始化权重参数
def init_ws_bs(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.normal_(m.weight.data, std=0.2)
        init.normal_(m.bias.data, std=0.2)


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


# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (256,3,96,96)
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            # (256,64,48,48)
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # (256,128,24,24)
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # (256,256,12,12)
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # (256,512,6,6)
            nn.Conv2d(512, 1, 6, 1, 0),
            # (256,1,1,1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def to_img(x):
    out = 0.5 * (x + 1)
    return out


if __name__ == '__main__':
    is_print = True

    # 创建对象
    try:
        G = torch.load(dir_path + '/G.pth', map_location=device)
        D = torch.load(dir_path + '/D.pth', map_location=device)
        print('Model loaded. Continue training.')
    except:
        G = Generator().to(device)
        D = Discriminator().to(device)
        print('Model inexist or loaded failed. Start training.')

    # 初始化权重参数
    init_ws_bs(G), init_ws_bs(D)

    lr = 2e-4
    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    real_labels = Variable(torch.ones(batch_size)).to(device)
    fake_labels = Variable(torch.zeros(batch_size)).to(device)

    for epoch in range(num_epoch):
        for i, (img, _) in tqdm(enumerate(dataloader)):
            # ====================训练判别器========================
            # D.zero_grad()
            d_optimizer.zero_grad()
            real_img = img.to(device)
            real_out = D(real_img)
            d_loss_real = criterion(real_out.view(-1), real_labels)
            d_loss_real.backward()

            # 计算生成图片的损失
            z = torch.randn(batch_size, z_dimension, 1, 1).to(device)  # 随机噪声
            fake_img = G(z).detach()  # 避免梯度传到G，因为G不用更新, detach分离
            fake_out = D(fake_img)
            d_loss_fake = criterion(fake_out.view(-1), fake_labels)
            d_loss_fake.backward()

            # 计算loss并更新D
            d_loss = (d_loss_real + d_loss_fake) / 2
            # d_loss.backward()
            d_optimizer.step()

            # ====================训练生成器========================
            if i % 5 == 0:
                # G.zero_grad()
                g_optimizer.zero_grad()
                z = torch.randn(batch_size, z_dimension, 1, 1).to(device)  # 随机噪声
                fake_img = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
                output = D(fake_img)
                g_loss = criterion(output.view(-1), real_labels)
                g_loss.backward()
                g_optimizer.step()

        try:
            fake_image = to_img(fake_img[:16].cpu())
            save_image(fake_image, res_path + '/fake_{}.png'.format(epoch + 1))
            # fake_image = fake_img[:16].cpu()
            # save_image(fake_image, res_path + '/fake_{}.png'.format(epoch + 1), normalize=True)
        except:
            pass
        if is_print:
            is_print = False
            real_image = to_img(real_img.cpu())
            save_image(real_image, res_path + '/real.png')
            # real_image = real_img[:16].cpu()
            # save_image(real_image, res_path + '/real.png', normalize=True)

        print('epoch{}'.format(epoch + 1), '\td_loss:%.5f' % d_loss.data.item(), '\tg_loss:%.5f' % g_loss.data.item())
        torch.save(D, dir_path + '/D.pth')
        torch.save(G, dir_path + '/G.pth')
