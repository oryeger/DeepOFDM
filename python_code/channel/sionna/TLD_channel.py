import os
from python_code import conf

from python_code.utils.constants import N_ANTS, NUM_SAMPLES_PER_SLOT, SAMPLING_RATE


if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

# Set random seed for reproducibility
sionna.config.seed = 42
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers

from sionna.channel.tr38901 import AntennaArray, CDL, TDL, Antenna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber
from sionna.utils.metrics import compute_ber


class TDLChannel:
    @staticmethod
    def conv_and_noise(y_in: np.ndarray, batch_size: int, noise_var: float) -> np.ndarray:
        tdl = TDL(model="A",
                          delay_spread=float(conf.delay_spread),
                          carrier_frequency=float(conf.carrier_frequency),
                          num_tx_ant=y_in.shape[0],
                          num_rx_ant=N_ANTS,
                          min_speed=conf.speed)

        num_time_samples = NUM_SAMPLES_PER_SLOT
        bandwidth  = SAMPLING_RATE


        l_min, l_max = time_lag_discrete_time_channel(bandwidth)
        l_tot = l_max-l_min+1

        cir = tdl(batch_size, num_time_samples+l_tot-1, bandwidth)
        h_time = cir_to_time_channel(bandwidth, *cir, l_min, l_max, normalize=True)

        channel_time = ApplyTimeChannel(num_time_samples, l_tot=l_tot, add_awgn=True)
        y_tf = tf.convert_to_tensor(y_in)
        y_reshaped = tf.reshape(y_tf, [batch_size, 1, y_in.shape[0], NUM_SAMPLES_PER_SLOT])
        y_reshaped = tf.cast(y_reshaped, h_time.dtype)

        no = tf.convert_to_tensor(float(noise_var))
        y_out = channel_time([y_reshaped, h_time, no])
        y_out = y_out[:,:,:,:-l_tot+1]
        y_out = tf.reshape(y_out, (1, N_ANTS, int(batch_size*NUM_SAMPLES_PER_SLOT)))
        y_out = y_out.numpy()
        # plt.plot(20*np.log10(tf.abs(y_out[5,0,0,0,:].numpy())), '-', color='g', label='Batch 1, Symbol 0')
        # plt.plot(20*np.log10(tf.abs(y_out[5,0,0,13,:].numpy())), '-', color='r', label='Batch 1, Symbol 13')
        # plt.xlabel('RE')
        # plt.ylabel('Amp')
        # plt.title('channel', fontsize=10)
        # plt.legend()
        # plt.grid()
        # plt.tight_layout()
        # plt.show()
        # pass
        return y_out

