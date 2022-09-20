import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *

train_data = torchvision.datasets.CIFAR10("./cifar10",train=True,download=True,transform= torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./cifar10",train=False,download=True,transform= torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)

print(train_data_size, test_data_size)

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

class NM(nn.Module):
    def __init__(self):
        super(NM, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        input = self.model(input)
        return input

nm = NM()

# 调用GPU训练
if torch.cuda.is_available():
    nm = nm.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 调用GPU训练
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(nm.parameters(),lr = learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0

# 记录测试的次数
total_test_step = 0

# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs")

for i in range(epoch):
    print("第{}轮训练开始".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs,targets = data
        # 调用GPU训练
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = nm(imgs)
        loss = loss_fn(outputs,targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step%100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤开始
    total_accuracy = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # 调用GPU训练
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = nm(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的accuracy：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(nm,"nm_{}.pth".format(i))
    # torch.save(nm.state_dict(),"nm_{}.pth".format(i))
    print("模型已保存")

writer.close()