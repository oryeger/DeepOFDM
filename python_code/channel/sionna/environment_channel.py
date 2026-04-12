import os
from python_code import conf

from python_code.utils.constants import NUM_SAMPLES_PER_SLOT, SAMPLING_RATE
import matplotlib.pyplot as plt

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import sionna
except ImportError:
    import os as _os
    if _os.name == 'nt':
        _os.system("pip install sionna >nul 2>&1")
    else:
        _os.system("pip install sionna >/dev/null 2>&1")
    import sionna

if os.name == "posix":
    sionna.rt = None
    sionna.config.enable_rtx = False

import tensorflow as tf
import numpy as np

from sionna.channel.tr38901 import UMa, UMi, RMa, PanelArray
from sionna.channel import cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyTimeChannel
from typing import Tuple


class EnvironmentChannel:
    @staticmethod
    def conv_cir(y_in: np.ndarray, conv_size: int, noise_var: float, num_slots: int,
                 external_channel: tf.Tensor, seed: int) -> Tuple[np.ndarray, tf.Tensor]:

        sionna.config.seed = seed

        num_tx = y_in.shape[0]  # number of transmit antennas (users)

        # UT antenna array: num_tx single-pol omni elements
        ut_array = PanelArray(
            num_rows_per_panel=num_tx,
            num_cols_per_panel=1,
            polarization='single',
            polarization_type='V',
            antenna_pattern='omni',
            carrier_frequency=float(conf.carrier_frequency))

        # BS antenna array: dual cross-pol, 2 cols per panel, rows fill the rest
        # Total antennas = num_rows * num_cols * 2 (dual pol)  =>  num_rows = n_ants // 4
        bs_num_cols = 2
        assert conf.n_ants % (bs_num_cols * 2) == 0, \
            f"n_ants={conf.n_ants} must be a multiple of 4 for dual-pol 2-col array (got {conf.n_ants})"
        bs_num_rows = conf.n_ants // (bs_num_cols * 2)
        bs_array = PanelArray(
            num_rows_per_panel=bs_num_rows,
            num_cols_per_panel=bs_num_cols,
            polarization='dual',
            polarization_type='cross',
            antenna_pattern='omni',
            carrier_frequency=float(conf.carrier_frequency))

        # Build channel model
        model_name = conf.channel_model  # 'SUMa', 'SUMi', or 'SRMa'
        o2i = getattr(conf, 'env_o2i_model', 'low')
        enable_pl = getattr(conf, 'env_enable_pathloss', False)
        enable_sf = getattr(conf, 'env_enable_shadow_fading', False)

        if model_name == 'SUMa':
            channel = UMa(carrier_frequency=float(conf.carrier_frequency),
                          o2i_model=o2i,
                          ut_array=ut_array, bs_array=bs_array,
                          direction='uplink',
                          enable_pathloss=enable_pl,
                          enable_shadow_fading=enable_sf)
        elif model_name == 'SUMi':
            channel = UMi(carrier_frequency=float(conf.carrier_frequency),
                          o2i_model=o2i,
                          ut_array=ut_array, bs_array=bs_array,
                          direction='uplink',
                          enable_pathloss=enable_pl,
                          enable_shadow_fading=enable_sf)
        elif model_name == 'SRMa':
            channel = RMa(carrier_frequency=float(conf.carrier_frequency),
                          ut_array=ut_array, bs_array=bs_array,
                          direction='uplink',
                          enable_pathloss=enable_pl,
                          enable_shadow_fading=enable_sf)
        else:
            raise ValueError(f"Unknown environment channel model: {model_name}. Use 'SUMa', 'SUMi', or 'SRMa'.")

        # Network topology: 1 batch, 1 BS, 1 UT, static
        bs_height = float(getattr(conf, 'env_bs_height', 25.0))
        ut_height = float(getattr(conf, 'env_ut_height', 1.5))
        distance  = float(getattr(conf, 'env_distance', 100.0))
        indoor    = bool(getattr(conf, 'env_indoor', False))

        bs_loc          = tf.constant([[[0.0, 0.0, bs_height]]])        # [1, 1, 3]
        ut_loc          = tf.constant([[[distance, 0.0, ut_height]]])   # [1, 1, 3]
        bs_orientations = tf.zeros([1, 1, 3])
        ut_orientations = tf.zeros([1, 1, 3])
        ut_velocities   = tf.zeros([1, 1, 3])
        in_state        = tf.constant([[indoor]])                        # [1, 1]

        channel.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                             ut_velocities, in_state)

        num_time_samples = NUM_SAMPLES_PER_SLOT
        bandwidth = SAMPLING_RATE

        l_min, l_max = time_lag_discrete_time_channel(bandwidth)
        l_tot = l_max - l_min + 1

        if tf.size(external_channel) == 0:
            cir = channel(num_time_samples + l_tot - 1, bandwidth)
            h_time = cir_to_time_channel(bandwidth, *cir, l_min, l_max, normalize=True)
            # Re-normalize so mean power per (rx_ant, tx_ant) element = 1, matching TDL behavior
            mean_pwr = tf.reduce_mean(tf.abs(h_time) ** 2)
            h_time = h_time / tf.cast(tf.sqrt(mean_pwr), h_time.dtype)
            if conf.plot_channel:
                # h_time: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_tot]
                colors = ['g', 'r', 'k', 'b', 'c', 'm', 'orange', 'purple']
                n_rx_ant = h_time.shape[2]
                for ant in range(n_rx_ant):
                    h_ant = abs(h_time[0, 0, ant, 0, 0, 0, :])  # delay taps at time step 0
                    plt.plot(np.abs(h_ant), linestyle='-', color=colors[ant % len(colors)],
                             label=f'ant {ant}, peak@{np.argmax(h_ant)}')
                plt.title(model_name + ', f=' + str(int(float(conf.carrier_frequency) / 1e6)) + ' MHz', fontsize=10)
                plt.xlabel('time (samples)', fontsize=10)
                plt.ylabel('h', fontsize=10)
                plt.legend(fontsize=7)
                plt.grid()
                plt.show()
        else:
            h_time = external_channel

        h_cur = abs(h_time[0, 0, 0, 0, 0, 0, :])
        channel_time = ApplyTimeChannel(num_time_samples, l_tot=l_tot, add_awgn=True)
        y_tf = tf.convert_to_tensor(y_in)
        TA = np.argmax(h_cur)
        no = tf.convert_to_tensor(float(noise_var))

        y_out = np.zeros([1, 1, conf.n_ants, NUM_SAMPLES_PER_SLOT * num_slots], dtype=np.complex128)
        for slot in range(num_slots):
            y_reshaped = tf.reshape(
                y_tf[:, :, slot * num_time_samples:(slot + 1) * num_time_samples],
                [conv_size, 1, y_in.shape[0], num_time_samples])
            y_reshaped = tf.cast(y_reshaped, h_time.dtype)
            y_out_cur_slot = channel_time([y_reshaped, h_time, no])
            y_out_cur_slot = y_out_cur_slot[:, :, :, TA:-l_tot + 1 + TA]
            y_out[:, :, :, slot * num_time_samples:(slot + 1) * num_time_samples] = y_out_cur_slot.numpy()

        y_out = np.reshape(y_out, (1, conf.n_ants, int(NUM_SAMPLES_PER_SLOT * num_slots)))
        return y_out, h_time
