import numpy as np
import pandas as pd
import torch
import csv
import sys
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class TestDataset(Dataset):
    def __init__(self, filepath='test.csv'):
        self.data = pd.read_csv(filepath)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def __getitem__(self, index):
        img_as_np = np.asarray(self.data.iloc[index][0].split()).astype(float).reshape(48, 48)
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        img_as_tensor = self.transform(img_as_img)
        return img_as_tensor

    def __len__(self):
        return len(self.data.index)


test_dataset = TestDataset(sys.argv[1])

test_loader = DataLoader(dataset=test_dataset, batch_size=64)

test_data_size = len(test_dataset)


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
    model.load_state_dict(torch.load('model_param.pth'))
else:
    model.load_state_dict(torch.load('model_param.pth', map_location=torch.device('cpu')))

if torch.cuda.is_available():
    model = model.cuda()


def test(filename='res.csv'):
    res = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader, 0):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            np.asarray(predicted)
            res = np.append(res, predicted)
            res = res.astype(int)

    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "emotion"])
        i = 0
        while i < test_data_size:
            writer.writerow([i + 1, res[i]])
            i += 1
    print("Saved to " + filename)


# if __name__ == '__main__':
test(sys.argv[2])
