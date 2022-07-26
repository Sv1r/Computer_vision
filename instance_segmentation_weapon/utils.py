import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def collate_fn(batch):
    return tuple(zip(*batch))


def train_model(model_local, optimizer_local, num_epochs, dataloader):
    """Train/Test model"""

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        running_loss = 0.
        running_loss_classifier = 0.
        running_loss_box_reg = 0.
        running_loss_mask = 0.
        running_loss_objectness = 0.

        model_local.train()
        # Iterate over data.
        for images, targets in dataloader:
            images = list(image.to(device).float() for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer_local.zero_grad()

            # forward and backward
            with torch.set_grad_enabled(True):
                loss_dict = model_local(images, targets)
                loss_value = sum(loss for loss in loss_dict.values())

                # backward + optimize only if in training phase
                loss_value.backward()
                optimizer_local.step()

                # statistics
                running_loss += loss_value.item()
                running_loss_classifier += loss_dict['loss_classifier'].item()
                running_loss_box_reg += loss_dict['loss_box_reg'].item()
                running_loss_mask += loss_dict['loss_mask'].item()
                running_loss_objectness += loss_dict['loss_objectness'].item()

        epoch_loss = running_loss / len(dataloader)
        epoch_loss_classifier = running_loss_classifier / len(dataloader)
        epoch_loss_box_reg = running_loss_box_reg / len(dataloader)
        epoch_loss_mask = running_loss_mask / len(dataloader)
        epoch_loss_objectness = running_loss_objectness / len(dataloader)

        print(
            'Loss: {:.4f} Loss Classifier: {:.4f} Loss Box Reg: {:.4f} Loss Mask: {:.4f} Loss Objectness {:.4f}'.format(
                epoch_loss, epoch_loss_classifier, epoch_loss_box_reg, epoch_loss_mask, epoch_loss_objectness
            ), flush=True)

    return model_local
