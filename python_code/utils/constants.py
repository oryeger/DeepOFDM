from enum import Enum
import numpy as np

NUM_SNRs = 1
BLOCK_LENGTH_FACTOR = 3
HALF = 0.5
N_USERS = 4 # number of users
N_ANTS = 4 # number of antennas
TRAIN_PERCENTAGE = 80
IS_COMPLEX = 1
PHASE_OFFSET = 0 # np.pi / 3
SHOW_ALL_ITERATIONS = True
GENIE_CFO = False
NUM_REs = 12
EPOCHS = 200
ITERATIONS = 2
INTERF_FACTOR = 1
NUM_SYMB_PER_SLOT = 14



MOD_GENERAL = 16          # 2: BPSK, 4: QPSK, 16: 16QAM, 64: 64QAM

MOD_PILOT = MOD_GENERAL
MOD_DATA = MOD_GENERAL

NUM_BITS = int(np.log2(MOD_PILOT))

class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'
