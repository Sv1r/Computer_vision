import torch
from torchvision import models

from global_constants import *


class ResNet50(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        resnet.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=.2),
            torch.nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.output = torch.nn.Sigmoid()

    def forward(self, x):
        return self.output(self.base_model(x))


# Initialize the model
resnet_model = ResNet50(NUMBER_OF_CLASSES)
