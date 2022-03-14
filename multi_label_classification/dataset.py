import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import data_spliter
from global_constants import *

features_train, features_valid, target_train, target_valid = data_spliter()


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = self.image_files[index]
        label = self.labels[index]

        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, label


train_transform = A.Compose([
    # A.VerticalFlip(p=.5),
    A.HorizontalFlip(p=.5),
    # A.RandomRotate90(p=.5),
    A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, p=.5),
    # A.GridDistortion(p=.5),
    A.Normalize(mean=MEAN, std=STD, p=1.),
    ToTensorV2()
])
valid_transform = A.Compose([
    A.Normalize(mean=MEAN, std=STD, p=1.),
    ToTensorV2()
])

train_dataset = ClassificationDataset(
    features_train,
    target_train,
    transform=train_transform
)
valid_dataset = ClassificationDataset(
    features_valid,
    target_valid,
    transform=valid_transform
)

print(f'Image shape: {train_dataset[0][0].shape}')
print(f'Number of Labels shape: {train_dataset[0][1].shape[0]}\n')
print(f'Train dataset length: {train_dataset.__len__()}')
print(f'Valid dataset length: {valid_dataset.__len__()}\n')

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    # pin_memory=True
)
valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    # pin_memory=True
)
