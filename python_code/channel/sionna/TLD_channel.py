import os
from python_code import conf

from python_code.utils.constants import NUM_SAMPLES_PER_SLOT, SAMPLING_RATE
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
    if os.name == 'nt':  # Windows
        os.system("pip install sionna >nul 2>&1")
    else:  # Unix/Linux/macOS
        os.system("pip install sionna >/dev/null 2>&1")
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
    def _get_spatial_correlation_matrices(correlation_level: str, num_tx_ant: int, num_rx_ant: int):
        """
        Generate spatial correlation matrices for TX and RX based on 3GPP TS 38.101-4 correlation model.

        Args:
            correlation_level: One of 'none', 'low', 'medium', 'medium_a', 'high', or 'custom'
            num_tx_ant: Number of transmit antennas
            num_rx_ant: Number of receive antennas

        Returns:
            Tuple of (tx_correlation_matrix, rx_correlation_matrix)

        Note:
            3GPP TS 38.101-4 defines correlation using α and β parameters.
            The correlation matrix is: R(n,m) = [(α^|n-m| + β^|n-m|)/2] for n≠m, and 1 for n=m
        """
        if correlation_level == 'none' or correlation_level is None:
            return None, None

        # 3GPP TS 38.101-4 correlation parameters
        # Format: (α, β)
        correlation_params = {
            'low': (0.0, 0.0),           # Low correlation
            'medium': (0.3, 0.9),        # Medium correlation
            'medium_a': (0.3, 0.3874),   # Medium Correlation A
            'high': (0.9, 0.9)           # High correlation
        }

        # Check for custom correlation parameters
        if correlation_level == 'custom':
            # Get custom α and β from config
            alpha = getattr(conf, 'spatial_correlation_alpha', 0.0)
            beta = getattr(conf, 'spatial_correlation_beta', 0.0)
        else:
            params = correlation_params.get(correlation_level)
            if params is None:
                print(f"Warning: Unknown correlation level '{correlation_level}', using 'none'")
                return None, None
            alpha, beta = params

        # Generate correlation matrices using 3GPP model
        def generate_correlation_matrix_3gpp(num_ant, alpha, beta):
            """
            Generate correlation matrix according to 3GPP TS 38.101-4
            R(n,m) = [(α^|n-m| + β^|n-m|)/2] for n≠m
            R(n,n) = 1
            """
            R = np.zeros((num_ant, num_ant), dtype=np.complex128)
            for i in range(num_ant):
                for j in range(num_ant):
                    if i == j:
                        R[i, j] = 1.0
                    else:
                        distance = abs(i - j)
                        R[i, j] = (alpha ** distance + beta ** distance) / 2.0
            return tf.constant(R, dtype=tf.complex64)

        tx_corr = generate_correlation_matrix_3gpp(num_tx_ant, alpha, beta)
        rx_corr = generate_correlation_matrix_3gpp(num_rx_ant, alpha, beta)

        return tx_corr, rx_corr

    @staticmethod
    def conv_cir(y_in: np.ndarray, conv_size: int, noise_var: float, num_slots: int, external_channel: tf.Tensor, seed: int) -> Tuple[np.ndarray, tf.Tensor]:
        # Set random seed for reproducibility
        sionna.config.seed = seed

        # Get spatial correlation matrices if configured
        spatial_corr_level = getattr(conf, 'spatial_correlation', 'none')
        tx_corr, rx_corr = TDLChannel._get_spatial_correlation_matrices(
            spatial_corr_level,
            y_in.shape[0],
            conf.n_ants
        )

        # Create TDL channel with optional spatial correlation
        tdl_params = {
            'model': conf.TDL_model,
            'delay_spread': float(conf.delay_spread),
            'carrier_frequency': float(conf.carrier_frequency),
            'num_tx_ant': y_in.shape[0],
            'num_rx_ant': conf.n_ants,
            'min_speed': conf.speed
        }

        # Add spatial correlation if configured (use separate tx/rx matrices)
        if rx_corr is not None:
            tdl_params['rx_corr_mat'] = rx_corr
        if tx_corr is not None:
            tdl_params['tx_corr_mat'] = tx_corr

        tdl = TDL(**tdl_params)


        # num_time_samples = NUM_SAMPLES_PER_SLOT * num_slots
        num_time_samples = NUM_SAMPLES_PER_SLOT
        bandwidth  = SAMPLING_RATE


        l_min, l_max = time_lag_discrete_time_channel(bandwidth)
        l_tot = l_max-l_min+1

        if tf.size(external_channel) == 0:
            cir = tdl(conv_size, num_time_samples+l_tot-1, bandwidth)
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
        y_out = np.zeros([1,1,conf.n_ants,NUM_SAMPLES_PER_SLOT*num_slots],dtype=np.complex128)
        for slot in range(num_slots):
            y_reshaped = tf.reshape(y_tf[:,:,slot*num_time_samples:(slot+1)*num_time_samples], [conv_size, 1, y_in.shape[0], num_time_samples])
            y_reshaped = tf.cast(y_reshaped, h_time.dtype)
            y_out_cur_slot = channel_time([y_reshaped, h_time, no])
            y_out_cur_slot = y_out_cur_slot[:,:,:,TA:-l_tot+1+TA]
            y_out[:,:,:,slot*num_time_samples:(slot+1)*num_time_samples] = y_out_cur_slot.numpy()
        y_out = np.reshape(y_out, (1, conf.n_ants, int(NUM_SAMPLES_PER_SLOT*num_slots)))
        return y_out, h_time


