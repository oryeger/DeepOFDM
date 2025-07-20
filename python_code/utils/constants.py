from enum import Enum
import numpy as np

BLOCK_LENGTH_FACTOR = 3
HALF = 0.5
TRAIN_PERCENTAGE = 80
IS_COMPLEX = 1
PHASE_OFFSET = np.pi / 3
SHOW_ALL_ITERATIONS = True
CFO_COMP = 'ON_Y' # 'NONE', 'ON_CE', 'ON_Y'
GENIE_CFO = True
NOISE_TO_CE = True
PLOT_MI = False
PLOT_CE_ON_DATA = False

NUM_SYMB_PER_SLOT = 14 # 500
FFT_size = 128
FIRST_CP = 11
CP = 9
SAMPLING_RATE = 3.84e6
# FFT_size = 512
# FIRST_CP = 44
# CP = 36
# SAMPLING_RATE = 15.36e6
NUM_SAMPLES_PER_SLOT = int(0.5e-3 * SAMPLING_RATE)

# Testing

# FFT_size = 1024
# FIRST_CP = 88
# CP = 72
# SAMPLING_RATE = 30.72e6


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'
