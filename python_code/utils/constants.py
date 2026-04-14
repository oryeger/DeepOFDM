from enum import Enum
import numpy as np

HALF = 0.5
TRAIN_PERCENTAGE = 80
SHOW_ALL_ITERATIONS = True
GENIE_CFO = True

NUM_SYMB_PER_SLOT = 14 # 500
FFT_size = 128
FIRST_CP = 11
CP = 9
SAMPLING_RATE = 3.84e6
NUM_SAMPLES_PER_SLOT = int(0.5e-3 * SAMPLING_RATE)

class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'
