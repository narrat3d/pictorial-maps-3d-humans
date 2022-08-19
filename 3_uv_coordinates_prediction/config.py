TRAINING_DATA_FOLDER = r"C:\Users\raimund\Downloads\tmp2"

DEBUG = False

IMAGE_SIZE = 256
NUM_FILTERS = 32

if DEBUG:
    BATCH_SIZE = 1
else:
    BATCH_SIZE = 8

INCLUDE_BODY_PARTS = True	

RGB_COLORS = [
    [0.25, 0.25, 0.25],
    [0.75, 0.75, 0.75],
    [1.0, 0, 0],
    [0, 1.0, 0],
    [0, 0, 1.0],
    [1.0, 1.0, 0],
    [0, 1.0, 1.0],
    [1.0, 0, 1.0],
    [0.5, 0.5, 0],
    [0, 0.5, 0.5],
    [0.5, 0, 0.5],
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5],
]

if (INCLUDE_BODY_PARTS):
    NUM_BODY_PARTS = len(RGB_COLORS)
else :
    NUM_BODY_PARTS = 1