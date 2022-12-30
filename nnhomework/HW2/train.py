import sys
import cv2
import numpy as np
import torch
from Model import *
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn, optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter


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

train_dataset = ImageFolder(sys.argv[1], transform=train_transform)
target_dataset = ImageFolder(sys.argv[2], transform=target_transform)

train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=50, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device)
label_predictor = LabelPredictor().to(device)
domain_classifier = DomainClassifier().to(device)
class_criterion = nn.CrossEntropyLoss().to(device)
domain_criterion = nn.BCEWithLogitsLoss().to(device)

optimizer_feature = optim.Adam(lr=0.0001, params=feature_extractor.parameters())
optimizer_label = optim.Adam(lr=0.0001, params=label_predictor.parameters())
optimizer_domain = optim.Adam(lr=0.0001, params=domain_classifier.parameters())


def train_epoch(train_dataloader, target_dataloader, lamb):

    domain_loss, label_loss, total_correct, total_num = 0.0, 0.0, 0.0, 0.0

    for i, ((train_data, train_label), (target_data, _)) in enumerate(zip(train_dataloader, target_dataloader)):
        concat_data = torch.concat([train_data, target_data], dim=0)
        domain_label = torch.zeros([train_data.shape[0] + target_data.shape[0], 1])
        domain_label[:train_data.shape[0]] = 1

        if torch.cuda.is_available():
            train_data = train_data.cuda()
            train_label = train_label.cuda()
            concat_data = concat_data.cuda()
            domain_label = domain_label.cuda()

        feature = feature_extractor(concat_data)

        domain_output = domain_classifier(feature.detach())
        loss = domain_criterion(domain_output, domain_label)
        domain_loss += loss.cpu().detach().numpy() if torch.cuda.is_available() else loss.detach().numpy()
        loss.backward()
        optimizer_domain.step()

        label_output = label_predictor(feature[:train_data.shape[0]])
        domain_output = domain_classifier(feature.detach())

        if torch.cuda.is_available():
            label_output = label_output.cuda()
            domain_output = domain_output.cuda()

        loss = class_criterion(label_output, train_label) - lamb * domain_criterion(domain_output, domain_label)
        label_loss += loss.cpu().detach().numpy() if torch.cuda.is_available() else loss.detach().numpy()
        loss.backward()
        optimizer_feature.step()
        optimizer_label.step()

        optimizer_domain.zero_grad()
        optimizer_feature.zero_grad()
        optimizer_label.zero_grad()

        acc_bool = torch.argmax(label_output, dim=1) == train_label.squeeze()
        total_correct += np.sum(acc_bool.cpu().numpy() != 0) if torch.cuda.is_available() else np.sum(acc_bool.numpy() != 0)
        total_num += train_data.shape[0]
        print(i, end='\r')
    return domain_loss / (i+1), label_loss / (i+1), total_correct / total_num

writer = SummaryWriter('./logs')

domainLoss_record, labelLoss_record, Acc_record = [], [], []

for epoch in range(150):
    domainLoss, labelLoss, Acc = train_epoch(train_dataloader, target_dataloader, lamb=0.1)
    domainLoss_record.append(domainLoss)
    labelLoss_record.append(labelLoss)
    Acc_record.append(Acc)
    writer.add_scalar("domainLoss", domainLoss.item(), epoch + 1)
    writer.add_scalar("labelLoss", labelLoss.item(), epoch + 1)
    writer.add_scalar("Acc", Acc.item(), epoch + 1)

    if (epoch+1) % 30 == 0:
        torch.save(feature_extractor.state_dict(), "./feature_extractor.pth")
        torch.save(label_predictor.state_dict(), "./label_predictor.pth")
    print('epoch {}: domain_loss: {:.4f}, label_loss: {:.4f}, acc: {:.4f}'.format(epoch+1, domainLoss, labelLoss, Acc))
