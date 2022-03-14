import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from global_constants import *


def nonzero_channels(mask_path):
    """Return list of non-zero channels of mask"""
    array = np.load(mask_path)
    array = array.reshape((NUMBER_OF_CLASSES, 352, 352))
    result = np.unique(np.argwhere(array >= THRESHOLD).T[0], axis=0).ravel()

    return list(result)


def parse_label(label_list):
    """Return binary labels list"""
    local_label_array = np.zeros((1, NUMBER_OF_CLASSES)).astype('int')
    for i in label_list:
        local_label_array[0][i] += 1

    return local_label_array[0]


def create_tabular_data():
    """Create tabular representation of data"""
    images = glob.glob(os.path.join(IMAGES_PATH, '*'))
    masks = glob.glob(os.path.join(MASKS_PATH, '*'))

    data = pd.DataFrame({'masks': masks,
                         'images': images})
    data[LABELS] = np.stack(list(np.zeros((1, NUMBER_OF_CLASSES))) * len(images))
    data['unique_labels'] = data['masks'].apply(lambda x: nonzero_channels(x))
    for k, j in tqdm(enumerate(LABELS)):
        for i in range(0, len(data)):
            new_value = list(parse_label(data['unique_labels'].iloc[i]))[k]
            data[j].loc[data.index == i] = new_value

    return data


def data_spliter(dataframe_path=METADATA_TABLE):
    """Split data on train/valid datasets"""
    dataframe = pd.read_csv(dataframe_path, sep=' ', index_col=0)
    images = dataframe['images'].values
    labels = dataframe[LABELS].values

    features_train, features_valid, target_train, target_valid = train_test_split(
        images,
        labels,
        train_size=.9,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    return features_train, features_valid, target_train, target_valid


def to_numpy(tensor):
    """Convert torch.Tensor to NumPy Array"""
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
