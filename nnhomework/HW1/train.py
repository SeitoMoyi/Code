import numpy as np
import pandas as pd
import torch
import sys
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter


class TrainDataset(Dataset):
    def __init__(self, filepath='train.csv'):
        self.data = pd.read_csv(filepath)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        img_as_np = np.asarray(self.data.iloc[index][1].split()).astype(float).reshape(48, 48)
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        img_as_tensor = self.transform(img_as_img)
        return img_as_tensor, single_image_label

    def __len__(self):
        return len(self.data.index)


train_dataset = TrainDataset(sys.argv[1])

train_loader = DataLoader(dataset=train_dataset, batch_size=64)

train_data_size = len(train_dataset)

writer = SummaryWriter("logs")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2304, 64),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        x = self.net(x)
        return x


model = Model()


if torch.cuda.is_available():
    model = model.cuda()

loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 100 == 0:
            print("训练次数：{}，loss：{}".format(batch_idx+1, loss.item()))
        writer.add_scalar("train_loss", loss.item(), epoch+1)


def accu(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on train set: %d %%' % (100 * correct / total))
        writer.add_scalar("test_accuracy", correct/total, epoch+1)


if __name__ == '__main__':
    for epoch in range(200):
        print('=============第{}轮训练开始============='.format(epoch+1))
        train(epoch)
        accu(epoch)
        if (epoch+1) % 5 == 0:
            # torch.save(model.state_dict(), "model_param_{}.pth".format(epoch+1))
            torch.save(model.state_dict(), "model_param.pth")
            print("模型已保存")
    writer.close()