import os
import sys
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm


batch_size = 256
num_epoch = 100
z_dimension = 100
res_path = './WGANGP_train_results'
dir_path = './WGANGP_model'
data_path = sys.argv[1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建文件夹
if not os.path.exists(res_path):
    os.mkdir(res_path)
if not os.path.exists(dir_path):
    os.mkdir(dir_path)


# 数据集
dataset = ImageFolder(data_path, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


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


# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            # (256,3,96,96)
            nn.Conv2d(3, 64, 5, 3, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # (256,64,32,32)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # (256,128,16,16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # (256,256,8,8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # (256,512,4,4)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            # (256,1,1,1)
            # nn.Flatten()
        )

    def forward(self, x):
        x = self.discriminator(x)
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


def calculate_gradient_penalty(real_images, fake_images):
    eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3)).to(device)

    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.to(device)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                                    create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty


if __name__ == '__main__':
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = (one * -1).to(device)

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

    D.weight_init()
    G.weight_init()

    lr = 2e-4
    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(num_epoch):
        for i, (img, _) in tqdm(enumerate(dataloader)):
            # ====================训练判别器========================
            for param in D.parameters():
                param.requires_grad = True

            # 计算真实图片的损失
            # D.zero_grad()
            d_optimizer.zero_grad()
            real_img = img.to(device)
            real_img = Variable(real_img)
            real_out = D(real_img)
            d_loss_real = real_out.mean()
            d_loss_real.backward(one)

            # 计算生成图片的损失
            z = torch.randn(batch_size, z_dimension, 1, 1).to(device)  # 随机噪声
            fake_img = G(z).detach()  # 避免梯度传到G，因为G不用更新, detach分离
            fake_out = D(fake_img)
            d_loss_fake = fake_out.mean()
            d_loss_fake.backward(mone)

            # 计算loss并更新D
            gradient_penalty = calculate_gradient_penalty(real_img.data, real_img.data)
            gradient_penalty.backward()

            d_loss = d_loss_fake - d_loss_real + gradient_penalty
            # d_loss.backward()
            d_optimizer.step()

            # ====================训练生成器========================
            if i % 5 == 0:
                for param in D.parameters():
                    param.requires_grad = False

                # G.zero_grad()
                g_optimizer.zero_grad()
                z = torch.randn(batch_size, z_dimension, 1, 1).to(device)  # 随机噪声
                fake_img = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
                output = D(fake_img)  # 经过判别器得到的结果
                g_loss = output.mean()
                g_loss.backward(one)
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

        print('epoch{}'.format(epoch+1), '\td_loss:%.5f' % d_loss.data.item(), '\tg_loss:%.5f' % g_loss.data.item())
        torch.save(D, dir_path + '/D.pth')
        torch.save(G, dir_path + '/G.pth')

