import os
from pathlib import Path
import time
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.detectors.deepsice2e.deepsice2e_trainer import DeepSICe2eTrainer
from python_code.detectors.deeprx.deeprx_trainer import DeepRxTrainer
from typing import List

import numpy as np
import torch
from python_code import DEVICE, conf
from python_code.utils.metrics import calculate_ber
import matplotlib.pyplot as plt
from python_code.utils.constants import (IS_COMPLEX, TRAIN_PERCENTAGE, CFO_COMP, GENIE_CFO,
                                         FFT_size, FIRST_CP, CP, NUM_SYMB_PER_SLOT, NUM_SAMPLES_PER_SLOT, PLOT_MI,
                                         PLOT_CE_ON_DATA, N_ANTS)

import commpy.modulation as mod

from python_code.channel.modulator import BPSKModulator, QPSKModulator, QAM16Modulator
import pandas as pd

from python_code.channel.channel_dataset import  ChannelModelDataset
from scipy.stats import entropy
from scipy.interpolate import interp1d

from python_code.detectors.sphere.sphere_decoder import SphereDecoder



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

base_dir = Path.home() / "Projects" / "Scratchpad"

aaa = QAM16Modulator.demodulate([3.4251-1.7942j])
print(aaa[0])
