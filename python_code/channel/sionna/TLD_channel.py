import os
from python_code import conf

from python_code.utils.constants import N_ANTS, NUM_SAMPLES_PER_SLOT, SAMPLING_RATE
import matplotlib.pyplot as plt


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

if os.name == "posix":
    sionna.rt = None
    sionna.config.enable_rtx = False


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

import numpy as np
from sionna.channel.tr38901 import TDL
from sionna.channel import cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyTimeChannel
from typing import Tuple


class TDLChannel:
    @staticmethod
    def conv_and_noise(y_in: np.ndarray, batch_size: int, noise_var: float, num_slots: int, external_channel: tf.Tensor) -> Tuple[np.ndarray, tf.Tensor]:
        # Set random seed for reproducibility
        sionna.config.seed = conf.channel_seed
        # 42 731

        tdl = TDL(model=conf.TDL_model,
                          delay_spread=float(conf.delay_spread),
                          carrier_frequency=float(conf.carrier_frequency),
                          num_tx_ant=y_in.shape[0],
                          num_rx_ant=N_ANTS,
                          min_speed=conf.speed)


        # num_time_samples = NUM_SAMPLES_PER_SLOT * num_slots
        num_time_samples = NUM_SAMPLES_PER_SLOT
        bandwidth  = SAMPLING_RATE


        l_min, l_max = time_lag_discrete_time_channel(bandwidth)
        l_tot = l_max-l_min+1

        if tf.size(external_channel) == 0:
            cir = tdl(batch_size, num_time_samples+l_tot-1, bandwidth)
            h_time = cir_to_time_channel(bandwidth, *cir, l_min, l_max, normalize=True)
        else:
            h_time = external_channel

        h_cur = abs(h_time[0, 0, 0, 0, 0, 0, :])
        if conf.plot_channel:
            plt.plot(np.abs(h_cur), linestyle='-', color='b', label='h_time, peak@ '+str(np.argmax(h_cur)))
            plt.title('TDL-'+conf.TDL_model+', delay spread='+str(int(round(float(conf.delay_spread)*1e9)))+' nsec', fontsize=10)
            plt.xlabel('time (samples)', fontsize=10)
            plt.ylabel('h', fontsize=10)
            plt.legend()
            plt.grid()
            plt.show()

        channel_time = ApplyTimeChannel(num_time_samples, l_tot=l_tot, add_awgn=True)
        y_tf = tf.convert_to_tensor(y_in)
        TA = np.argmax(h_cur)
        no = tf.convert_to_tensor(float(noise_var))
        y_out = np.zeros([1,1,N_ANTS,NUM_SAMPLES_PER_SLOT*num_slots],dtype=np.complex128)
        for slot in range(num_slots):
            y_reshaped = tf.reshape(y_tf[:,:,slot*num_time_samples:(slot+1)*num_time_samples], [batch_size, 1, y_in.shape[0], num_time_samples])
            y_reshaped = tf.cast(y_reshaped, h_time.dtype)
            y_out_cur_slot = channel_time([y_reshaped, h_time, no])
            y_out_cur_slot = y_out_cur_slot[:,:,:,TA:-l_tot+1+TA]
            y_out[:,:,:,slot*num_time_samples:(slot+1)*num_time_samples] = y_out_cur_slot.numpy()
        y_out = np.reshape(y_out, (1, N_ANTS, int(NUM_SAMPLES_PER_SLOT*num_slots)))
        # y_out = y_out.numpy()
        # np.save('C:\\Projects\\Scratchpad\\output.npy', y_out)
        # y_out_2 = np.load('C:\\Projects\\Scratchpad\\output.npy')

        return y_out, h_time

