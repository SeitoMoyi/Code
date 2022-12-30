import torch
import torchvision
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# y = np.array([1, 0, 0])
# z = np.array([0.2, 0.1, -0.1])
# y_pred = np.exp(z) / np.exp(z).sum()
# loss = (-y * np.log(y_pred)).sum()
# print(loss)


# y = torch.LongTensor([2, 0, 1])
# y_pred1 = torch.Tensor([[0.1, 0.2, 0.9],
#                         [1.1, 0.1, 0.2],
#                         [0.2, 2.1, 0.1]])
# y_pred2 = torch.Tensor([[0.8, 0.2, 0.3],
#                         [0.2, 0.3, 0.5],
#                         [0.2, 0.2, 0.5]])
# criterion = torch.nn.CrossEntropyLoss()
# l1 = criterion(y_pred1, y)
# l2 = criterion(y_pred2, y)
# print('Batch Loss1 = ', l1.data, '\nBatch Loss2 = ', l2.data)


batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST('./Mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./Mnist', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.net(x)
        return x


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
