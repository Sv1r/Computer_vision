import numpy as np

RGB2GRAY = np.array([.299, .587, .114])

IMAGES = 'data\\JPEGImages'
CLASSES = 'data\\SegmentationClass'
OBJECTS = 'data\\SegmentationObject'
LABEL_MAP_PATH = 'data\\labelmap.txt'

RANDOM_STATE = 42
# Normalize image between 0 and 1
IMAGE_MEAN = [.0, .0, .0]
IMAGE_STD = [1., 1., 1.]

EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
