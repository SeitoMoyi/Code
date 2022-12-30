import sys
import torch
import numpy as np
import pandas as pd
from Model import *
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

target_transform = T.Compose([
    T.Grayscale(),
    T.Resize((32, 32)),
    T.RandomHorizontalFlip(0.5),
    T.RandomRotation(15, fill=(0,)),
    T.ToTensor(),
])

target_dataset = ImageFolder('test_data', transform=target_transform)
test_dataloader = DataLoader(target_dataset, batch_size=100, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device)
label_predictor = LabelPredictor().to(device)

feature_extractor.load_state_dict(torch.load('feature_extractor.pth', map_location=device))
label_predictor.load_state_dict(torch.load('label_predictor.pth', map_location=device))

result = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(test_dataloader):
    if torch.cuda.is_available():
        test_data = test_data.cuda()

    label_output = label_predictor(feature_extractor(test_data))
    x = torch.argmax(label_output, dim=1).cpu().detach().numpy() if torch.cuda.is_available() else torch.argmax(label_output, dim=1).detach().numpy()
    result.append(x)

result = np.concatenate(result)

df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
df.to_csv('res.csv', index=False)
print('Saved')
