
# 获取网络结构的特征矩阵并可视化
import torch
from model import Model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

#  定义图像预处理过程(要与网络模型训练过程中的预处理过程一致)
# model
data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# create model
# model = AlexNet(num_classes=5)
# model = AlexNet(num_classes=1000)
model = Model()
# load model weights加载预训练权重
# model_weight_path ="./AlexNet.pth"
model_weight_path = "./model_param.pth"
model.load_state_dict(torch.load(model_weight_path))
# 打印出模型的结构
print(model)


# load image
img = Image.open("./rawpic/1.jpg")

# [N, C, H, W]（对图片预处理）
img = data_transform(img)
# expand batch dimension 增加一个banch维度
img = torch.unsqueeze(img, dim=0)
# forward正向传播过程
out_put = model(img)
print(out_put.shape)
print(out_put)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]    维度变换
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(32):
        ax = plt.subplot(4, 8, i+1)# 参数意义：3：图片绘制行数，5：绘制图片列数，i+1：图的索引
        # [H, W, C]
        # 特征矩阵每一个channel对应的是一个二维的特征矩阵，就像灰度图像一样，channel=1
        # plt.imshow(im[:, :, i])
        plt.imshow(im[:, :, i], cmap='gray')
    plt.savefig('conv1-1.jpg')
    plt.show()
