from enum import Enum
import numpy as np

NUM_SNRs = 1
BLOCK_LENGTH_FACTOR = 3
HALF = 0.5
N_USERS = 1 # number of users
N_ANTS = 4 # number of antennas
TRAIN_PERCENTAGE = 80
IS_COMPLEX = 1
PHASE_OFFSET = 0 # np.pi / 3
SHOW_ALL_ITERATIONS = True
GENIE_CFO = 'ON_Y' # 'NONE', 'ON_CE', 'ON_Y'
NUM_REs = 12
EPOCHS = 300
ITERATIONS = 1
INTERF_FACTOR = 1
NOISE_TO_CE = True

MOD_GENERAL = 4          # 2: BPSK, 4: QPSK, 16: 16QAM, 64: 64QAM

MOD_PILOT = MOD_GENERAL
MOD_DATA = MOD_GENERAL

NUM_BITS = int(np.log2(MOD_PILOT))

NUM_SYMB_PER_SLOT = 14 # 500
FFT_size = 128
FIRST_CP = 11
CP =9
SAMPLING_RATE = 3.84e6
NUM_SAMPLES_PER_SLOT = int(0.5e-3 * SAMPLING_RATE)


# FFT_size = 1024
# FIRST_CP = 88
# CP = 72
# SAMPLING_RATE = 30.72e6


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'
