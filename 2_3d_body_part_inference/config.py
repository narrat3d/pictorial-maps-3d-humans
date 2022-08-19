import math

DEBUG = False
NUM_FILTERS = 8
IMAGE_SIZE = 64
NUM_SAMPLES = 4000
LEARNING_RATE = 0.0001

CAMERA_ANGLES = {
    "2d": 0,
    "front": 0, 
    "left": math.pi / 2, 
    "back": math.pi,
    "right": 3 * math.pi / 2
}

IMAGE_SIZE_HALF = int(IMAGE_SIZE / 2)
NUM_SAMPLES_HALF = int(NUM_SAMPLES/2)
SCALING_FACTOR = math.sqrt(IMAGE_SIZE_HALF**2 + IMAGE_SIZE_HALF**2 + IMAGE_SIZE_HALF**2)

if DEBUG:
    VIEWS = ["back"]
    BATCH_SIZE = 1
    EPOCHS = 1
else : 
    VIEWS = ["front", "left", "back", "right"]
    EPOCHS = 200
    BATCH_SIZE = 4