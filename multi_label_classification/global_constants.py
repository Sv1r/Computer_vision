RANDOM_STATE = 42
IMAGE_SIZE = 352
MEAN = [.414, .412, .408]
STD = [.139, .137, .136]
BATCH_SIZE = 15
EPOCHS = 20
THRESHOLD = .75
LABELS = [
    '1_Scab', '2_Orange_peal', '3_Through_hole', '4_Imprint', '5_Attrition',
    '6_Rust', '7_Scratch', '8_Pollution', '9_Mechanical_damage', '10_Insects',
    '11_Stripe', '12_Dross', '13_Other', '14_Oil', '15_Roller_print'
]
NUMBER_OF_CLASSES = len(LABELS)
IMAGES_PATH = 'D:\\multilable_classification_metal\\data\\images'
MASKS_PATH = 'D:\\multilable_classification_metal\\data\\masks'
SAVE_METADATA = 'D:\\multilable_classification_metal\\PyCharm\\tables'
METADATA_TABLE = 'D:\\multilable_classification_metal\\tables\\metadata.csv'
