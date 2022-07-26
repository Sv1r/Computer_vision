import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from global_constants import *
from dataset import train_dataloader, valid_dataloader
from utils import to_numpy
from model import resnet_model

print(f'Torch version: {torch.__version__}')
print(f'Cuda?: {torch.cuda.is_available()}\n')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(model, loss, optimizer, num_epochs):
    """Train/Valid model"""
    # train_loss_history, valid_loss_history = [], []
    # train_accuracy_history, valid_accuracy_history = [], []
    # train_precision_history, valid_precision_history = [], []
    # train_recall_history, valid_recall_history = [], []
    # train_f1_history, valid_f1_history = [], []
    # train_roc_auc_history, valid_roc_auc_history = [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = valid_dataloader
                model.eval()  # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.
            running_precision = 0.
            running_recall = 0.
            running_f1 = 0.
            running_roc_auc = 0.

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs).float()
                    loss_value = loss(preds, labels)
                    preds_labels = (preds >= THRESHOLD).float()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item()
                running_acc += accuracy_score(
                    to_numpy(labels).ravel(), to_numpy(preds_labels).ravel()
                )
                running_precision += precision_score(
                    to_numpy(labels).ravel(), to_numpy(preds_labels).ravel(), average='micro'
                )
                running_recall += recall_score(
                    to_numpy(labels).ravel(), to_numpy(preds_labels).ravel(), average='micro'
                )
                running_f1 += f1_score(
                    to_numpy(labels).ravel(), to_numpy(preds_labels).ravel(), average='micro'
                )
                running_roc_auc += roc_auc_score(
                    to_numpy(labels).ravel(), to_numpy(preds).ravel(), average='micro'
                )

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)
            epoch_precision = running_precision / len(dataloader)
            epoch_recall = running_recall / len(dataloader)
            epoch_f1 = running_f1 / len(dataloader)
            epoch_roc_auc = running_roc_auc / len(dataloader)

            # if phase == 'train':
            #     train_loss_history.append(epoch_loss)
            #     train_accuracy_history.append(epoch_acc)
            #     train_precision_history.append(epoch_precision)
            #     train_recall_history.append(epoch_recall)
            #     train_f1_history.append(epoch_f1)
            #     train_roc_auc_history.append(epoch_roc_auc)
            # else:
            #     valid_loss_history.append(epoch_loss)
            #     valid_accuracy_history.append(epoch_acc)
            #     valid_precision_history.append(epoch_precision)
            #     valid_recall_history.append(epoch_recall)
            #     valid_f1_history.append(epoch_f1)
            #     valid_roc_auc_history.append(epoch_roc_auc)

            print('{} Loss: {:.4f} Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f} ROC-AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_roc_auc
            ),
                flush=True
            )

    return model


loss = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(resnet_model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=.1)

resnet_model = train_model(
    model=resnet_model.to(device), loss=loss, optimizer=optimizer, num_epochs=EPOCHS
)
