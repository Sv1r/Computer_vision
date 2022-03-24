import glob

import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

import cv2
from PIL import Image

import constants
import utils

# Upload labelmap as pandas dataframe
labelmap = pd.read_csv(constants.LABEL_MAP_PATH, sep=':', header=0)
# Create grayscale label dictionary
gray_list = []
for i in labelmap.index:
    gray_values = (labelmap[['r', 'g', 'b']].values[i] * constants.RGB2GRAY).sum().astype(np.uint8)
    gray_list.append(gray_values)

labels_list = [i - 1 for i in labelmap.index[1:]]
LABELS_DICT = dict(zip(gray_list[1:], labels_list))
print(f'Label dictionary:\n{LABELS_DICT}')
# Count number of classes
NUMBER_OF_CLASSES = len(LABELS_DICT) + 1
# Create dataframe with image/classes/object
images = sorted(list(glob.glob(f'{constants.IMAGES}\\*')))
classes = sorted(list(glob.glob(f'{constants.CLASSES}\\*')))
objects = sorted(list(glob.glob(f'{constants.OBJECTS}\\*')))

dict_df = {
    'images': images,
    'classes': classes,
    'objects': objects
}
df = pd.DataFrame.from_dict(dict_df)
# Split data on train/test
train, test = train_test_split(
    df,
    train_size=.8,
    random_state=constants.RANDOM_STATE,
    shuffle=True
)


class WeaponDataset(torch.utils.data.Dataset):
    def __init__(self, data, transforms=None):
        self.images_files = sorted(list(data['images']))
        self.masks_files = sorted(list(data['objects']))
        self.masks_classes = sorted(list(data['classes']))

        self.transforms = transforms

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, index):
        image_path = self.images_files[index]
        mask_path = self.masks_files[index]
        mask_classes_path = self.masks_classes[index]

        image = np.array(Image.open(image_path).convert('RGB')).astype(np.uint8)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_classes = cv2.imread(mask_classes_path, cv2.IMREAD_GRAYSCALE)

        objects_id = np.unique(mask_gray)[1:]
        number_of_objects = len(objects_id)
        # Build mask
        masks = np.zeros((mask_gray.shape[0], mask_gray.shape[1], number_of_objects))
        for i in range(number_of_objects):
            masks[:, :, i] += mask_gray == objects_id[i]
        # Create labels from mask_classes
        labels = np.array([])
        for i in range(number_of_objects):
            local_mask = masks[:, :, i].copy()
            local_mask_classes = mask_classes.copy()
            local_mask_classes[local_mask == 0] = 0
            label = np.unique(local_mask_classes)[1:]
            labels = np.append(labels, label).astype(np.uint8)

        labels = [LABELS_DICT[label] for label in labels]
        # labels = np.ones((number_of_objects,), dtype=np.int64)
        # Set up boxes
        boxes = []
        for i in range(number_of_objects):
            pos = np.where(masks[:, :, i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        # Augmentation
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=masks, bboxes=boxes, bbox_classes=labels)
            image = augmented['image']
            masks = augmented['mask']
            boxes = augmented['bboxes']
            labels = augmented['bbox_classes']

        # Convert to torch data type
        image = image / 255.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((number_of_objects,), dtype=torch.int64)
        # Wrap up into a dict
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        return image, target


train_transform = A.Compose(
    [
        A.VerticalFlip(p=.2),
        A.HorizontalFlip(p=.2),
        A.RandomRotate90(p=.2),
        A.Cutout(num_holes=16, p=.2),
        A.RGBShift(p=.2),
        A.RandomBrightnessContrast(p=.2),
        ToTensorV2(transpose_mask=True, p=1.)
    ],
    bbox_params=A.BboxParams(format='pascal_voc', min_area=1e4, min_visibility=.5, label_fields=['bbox_classes'])
)
test_transform = A.Compose(
    [
        ToTensorV2(transpose_mask=True)
    ],
    bbox_params=A.BboxParams(format='pascal_voc', min_area=1e4, min_visibility=.5, label_fields=['bbox_classes'])
)
train_dataset = WeaponDataset(
    train, train_transform
)
test_dataset = WeaponDataset(
    test, test_transform
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=constants.BATCH_SIZE, shuffle=True, collate_fn=utils.collate_fn
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=constants.BATCH_SIZE, shuffle=True, collate_fn=utils.collate_fn
)
