from enum import Enum
import numpy as np

HALF = 0.5
TRAIN_PERCENTAGE = 80
SHOW_ALL_ITERATIONS = True
GENIE_CFO = True

NUM_SYMB_PER_SLOT = 14 # 500
# FFT_size/FIRST_CP/CP/SAMPLING_RATE are a uniformly-scaled-down (here 4x) version of the real
# 5G 30kHz-SCS reference numerology (fs=122.88Msps, FFT_size=4096) - scaling all four together
# keeps SCS (=SAMPLING_RATE/FFT_size) and the CP/symbol-duration ratios exactly matching the
# standard, just at a coarser absolute sample rate for faster simulation. The 4x factor here
# (vs. the previous 32x) was chosen so CP comfortably exceeds the discrete-time channel filter's
# minimum footprint (13 taps at maximum_delay_spread->0, see time_lag_discrete_time_channel) -
# see TLD_channel.py's MAXIMUM_DELAY_SPREAD for the corresponding channel-side fix; both were
# needed together; SAMPLING_RATE alone doesn't help without also bounding maximum_delay_spread.
FFT_size = 512
FIRST_CP = 44
CP = 36
SAMPLING_RATE = 15.36e6
NUM_SAMPLES_PER_SLOT = int(0.5e-3 * SAMPLING_RATE)

class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'
