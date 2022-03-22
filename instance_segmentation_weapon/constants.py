import numpy as np

RGB2GRAY = np.array([.299, .587, .114])

IMAGES = 'data\\JPEGImages'
CLASSES = 'data\\SegmentationClass'
OBJECTS = 'data\\SegmentationObject'
LABEL_MAP_PATH = 'data\\labelmap.txt'

RANDOM_STATE = 42

EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
