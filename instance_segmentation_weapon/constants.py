import numpy as np

RGB2GRAY = np.array([.299, .587, .114])

IMAGES = 'data\\JPEGImages'
CLASSES = 'data\\SegmentationClass'
OBJECTS = 'data\\SegmentationObject'
LABEL_MAP_PATH = 'data\\labelmap.txt'

RANDOM_STATE = 0

EPOCHS = 100
LEARNING_RATE = 1e-5
BATCH_SIZE = 2
