import sys
import cv2
import numpy
import numpy as np
import torch
from Model import *
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn, optim
from sklearn.decomposition import PCA
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def train_transform(image):
    img = T.functional.to_grayscale(image)
    img = cv2.Canny(np.array(img), 170, 300)
    img = Image.fromarray(np.array(img))
    random_horizontal_flip = T.RandomHorizontalFlip(0.5)
    img = random_horizontal_flip(img)
    random_rotation = T.RandomRotation(15, fill=(0,))
    img = random_rotation(img)
    tensor = T.ToTensor()
    return tensor(img)


target_transform = T.Compose([
    T.Grayscale(),
    T.Resize((32, 32)),
    T.RandomHorizontalFlip(0.5),
    T.RandomRotation(15, fill=(0,)),
    T.ToTensor(),
])

train_dataset = ImageFolder('train_data', transform=train_transform)
target_dataset = ImageFolder('test_data', transform=target_transform)
# train_dataset = ImageFolder('train_data', transform=T.ToTensor())
# target_dataset = ImageFolder('test_data', transform=T.ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=False)
test_dataloader = DataLoader(target_dataset, batch_size=50, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device)
label_predictor = LabelPredictor().to(device)
domain_classifier = DomainClassifier().to(device)
class_criterion = nn.CrossEntropyLoss().to(device)
domain_criterion = nn.BCEWithLogitsLoss().to(device)

feature_extractor.load_state_dict(torch.load('feature_extractor.pth', map_location=device))
label_predictor.load_state_dict(torch.load('label_predictor.pth', map_location=device))

result = []
feature_extractor.eval()
label_predictor.eval()

for i, (train_data, train_label) in enumerate(train_dataloader):
    train_data = feature_extractor(train_data)
    result.append(train_data.detach())

train_dr = np.concatenate(result)

pca = PCA(n_components=2)
pca = pca.fit(train_dr)
train_dr = pca.transform(train_dr)

result = []

for i, (test_data,_) in enumerate(train_dataloader):
    test_data = feature_extractor(test_data)
    result.append(test_data.detach())

test_dr = np.concatenate(result)

pca = PCA(n_components=2)
pca = pca.fit(test_dr)
test_dr = pca.transform(test_dr)

plt.figure()
plt.scatter(train_dr[:, 0], train_dr[:, 1], 5
            ,alpha=.3
            ,c='red', label='train')
# plt.scatter(test_dr[:, 0], test_dr[:, 1], 5
#             ,alpha=.3
#             ,c='blue', label='test')
plt.legend()
plt.title('Untrained Model')
# plt.savefig('Trained.jpg')
plt.show()