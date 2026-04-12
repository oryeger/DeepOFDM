import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat
from typing import Tuple

from python_code import conf
from python_code.utils.constants import NUM_SAMPLES_PER_SLOT, SAMPLING_RATE
from dir_definitions import ROOT_DIR

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

from sionna.channel import cir_to_time_channel, time_lag_discrete_time_channel, ApplyTimeChannel


class QuadrigaChannel:
    """
    Loads a pre-generated QuaDRiGa channel from a .mat file (produced by
    escnn_matlab/generate_quadriga_channels.m) and applies it as a time-domain
    MIMO channel, matching the interface of TDLChannel.conv_cir.

    Expected .mat file variables (saved by the MATLAB generator):
        coeff  - complex double [n_rx_ant, n_users, n_paths]
                 Combined channel coefficients for all UEs.
        delay  - double [n_users, n_paths]
                 Per-UE path delays in seconds.
    """

    @staticmethod
    def conv_cir(y_in: np.ndarray, conv_size: int, noise_var: float, num_slots: int,
                 external_channel: tf.Tensor, seed: int) -> Tuple[np.ndarray, tf.Tensor]:

        num_time_samples = NUM_SAMPLES_PER_SLOT
        bandwidth = SAMPLING_RATE
        l_min, l_max = time_lag_discrete_time_channel(bandwidth)
        l_tot = l_max - l_min + 1

        assert conf.n_ants >= 4 and conf.n_ants % 4 == 0, (
            f"n_ants={conf.n_ants} is invalid for QuaDRiGa channels. "
            f"Must be a multiple of 4 and >= 4 (layout: 2H x (n_ants/4)V x 2 cross-pol)."
        )

        if tf.size(external_channel) == 0:
            # Load CIR from the pre-generated .mat file for this seed
            mat_dir_raw = getattr(conf, 'quadriga_mat_path', '../QuadrigaChannels')
            mat_dir = mat_dir_raw if os.path.isabs(mat_dir_raw) else os.path.normpath(os.path.join(ROOT_DIR, mat_dir_raw))
            mat_file = os.path.join(mat_dir, f'{conf.channel_model}_seed_{seed}.mat')

            assert os.path.exists(mat_file), (
                f"QuaDRiGa channel file not found: {mat_file}\n"
                f"Run escnn_matlab/generate_quadriga_channels.m to generate it first."
            )

            data = loadmat(mat_file)
            # coeff: [n_rx_ant, n_users, n_paths]
            # delay: [n_users, n_paths] in seconds
            coeff = data['coeff'].astype(np.complex64)
            delay = data['delay'].astype(np.float32)

            n_rx_ant = coeff.shape[0]
            n_users_file = coeff.shape[1]

            assert n_rx_ant == conf.n_ants, (
                f"QuaDRiGa mat file has {n_rx_ant} RX antennas but conf.n_ants={conf.n_ants}. "
                f"Re-generate channels with the current config."
            )
            assert n_users_file == y_in.shape[0], (
                f"QuaDRiGa mat file has {n_users_file} users but signal has {y_in.shape[0]}. "
                f"Re-generate channels with the current config."
            )

            # Format for Sionna cir_to_time_channel:
            #   a:   [batch, num_rx, num_rx_ant, num_tx,   num_tx_ant, num_paths, num_time_steps]
            #        [1,     1,      n_ants,     n_users,  1,          n_paths,   1             ]
            #   tau: [batch, num_rx, num_tx,     num_paths]
            #        [1,     1,      n_users,    n_paths  ]
            # Each user is a separate TX with 1 antenna — num_tx=n_users, num_tx_ant=1
            a = coeff[np.newaxis, np.newaxis, :, :, np.newaxis, :, np.newaxis]   # [1,1,n_ants,n_users,1,n_paths,1]
            tau = delay[np.newaxis, np.newaxis, :, :]                            # [1,1,n_users,n_paths]

            a_tf = tf.constant(a, dtype=tf.complex64)
            tau_tf = tf.constant(tau, dtype=tf.float32)

            # normalize=True: Sionna normalises per (rx_ant, tx_ant) pair.
            # Re-normalize so mean power = 1, matching TDL/Sionna behaviour.
            h_time = cir_to_time_channel(bandwidth, a_tf, tau_tf, l_min, l_max, normalize=True)
            mean_pwr = tf.reduce_mean(tf.abs(h_time) ** 2)
            h_time = h_time / tf.cast(tf.sqrt(mean_pwr), h_time.dtype)

            if conf.plot_channel:
                colors = ['g', 'r', 'k', 'b', 'c', 'm', 'orange', 'purple']
                n_rx_ant = h_time.shape[2]
                for ant in range(n_rx_ant):
                    h_ant = abs(h_time[0, 0, ant, 0, 0, 0, :])
                    plt.plot(np.abs(h_ant), linestyle='-', color=colors[ant % len(colors)],
                             label=f'Ant {ant}, peak@ {np.argmax(h_ant)}')
                plt.title(f'QuaDRiGa {conf.channel_model}', fontsize=10)
                plt.xlabel('time (samples)', fontsize=10)
                plt.ylabel('|h|', fontsize=10)
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
