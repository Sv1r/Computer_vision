import torch

import dataset
import utils
import constants
from model import get_instance_segmentation_model

model = get_instance_segmentation_model(num_classes=dataset.NUMBER_OF_CLASSES)
model.to(utils.device)
params = [i for i in model.parameters() if i.requires_grad]
optimizer = torch.optim.AdamW(params, lr=constants.LEARNING_RATE)

model = utils.train_model(model, optimizer, constants.EPOCHS, dataset.train_dataloader)
torch.save(model, 'instance_model_weapon_multiclass.pth')
