import numpy as np
from sympy import false

from python_code import conf
from python_code.utils.constants import (NUM_SYMB_PER_SLOT, FFT_size, FIRST_CP,CP, NUM_SAMPLES_PER_SLOT)

import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple

from python_code.analog.iqmm_model import apply_iq_mismatch



H_COEF = 0.8

from python_code.channel.sionna.TLD_channel import TDLChannel


class SEDChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_users: int, num_res: int) -> np.ndarray:
        H_real = SEDChannel.mapping_matrix(n_users, n_ant)
        H_complex = np.zeros((H_real.shape[0], H_real.shape[1], num_res), dtype=complex)
        for re_index in range(num_res):
            H_complex[:,:,re_index] = H_real
        return H_complex


    @staticmethod
    def hadamard_matrix(n):
        """Generate Hadamard matrix of order n (must be power of 2)."""
        if n == 1:
            return np.array([[1]])
        H = SEDChannel.hadamard_matrix(n // 2)
        return np.block([[H, H], [H, -H]])

    @staticmethod
    def mapping_matrix(n_layers, n_rx):
        """Generate mapping matrix using Hadamard construction."""
        # Find nearest power of 2 >= n_rx
        N = 1
        while N < n_rx:
            N *= 2
        H = SEDChannel.hadamard_matrix(N)

        # Normalize
        H = H / np.sqrt(N)

        # Take first n_rx rows and first n_layers columns
        return H[:n_rx, :n_layers]


    @staticmethod
    def _add_fading(H: np.ndarray, n_ant: int, frame_ind: int) -> np.ndarray:
        degs_array = np.array([51, 39, 33, 21])
        fade_mat = H_COEF + (1 - H_COEF) * np.cos(2 * np.pi * frame_ind / degs_array)
        fade_mat = np.tile(fade_mat.reshape(1, -1), [n_ant, 1])
        return H * fade_mat

    @staticmethod
    def apply_td_and_impairments(y_in, td_in_rx, cfo, clip_percentage_in_tx, num_res, n_users, tdl_channel: bool, external_chan: tf.Tensor, iqmm_gain, iqmm_phase, seed) -> Tuple[np.ndarray, tf.Tensor]:
        # # OryEger - one tone
        # if iqmm_gain != 0:
        #     y_in_temp = y_in.copy()
        #     y_in = np.zeros_like(y_in_temp)
        #     index = 3
        #     y_in[0,0,index] = y_in_temp[0,0,index]

        if td_in_rx:
            if not(tdl_channel):
                if y_in.ndim == 4:
                    NUM_SLOTS = int(y_in.shape[2] / NUM_SYMB_PER_SLOT)
                    NUM_SAMPLES_TOTAL = int(NUM_SLOTS * NUM_SAMPLES_PER_SLOT)
                    n_users_int = n_users
                    y = y_in
                else:
                    NUM_SLOTS = int(y_in.shape[1] / NUM_SYMB_PER_SLOT)
                    NUM_SAMPLES_TOTAL = int(NUM_SLOTS * NUM_SAMPLES_PER_SLOT)
                    n_users_int = 1
                    y = np.expand_dims(y_in, axis=0)
                n_input_rx_int = y.shape[1]
                n_users_out_int = n_users_int
            else:
                NUM_SLOTS = int(y_in.shape[1] / NUM_SYMB_PER_SLOT)
                NUM_SAMPLES_TOTAL = int(NUM_SLOTS * NUM_SAMPLES_PER_SLOT)
                n_users_int = y_in.shape[0]
                y = np.expand_dims(y_in, axis=1)
                n_input_rx_int = 1
                n_users_out_int = 1
        else:
            NUM_SLOTS = int(y_in.shape[1] / NUM_SYMB_PER_SLOT)
            NUM_SAMPLES_TOTAL = int(NUM_SLOTS * NUM_SAMPLES_PER_SLOT)
            n_users_int = n_users
            n_input_rx_int = 1;
            n_users_out_int = n_users
            y = np.expand_dims(y_in, axis=1)

        st_full = np.zeros((n_users_int, n_input_rx_int, NUM_SAMPLES_TOTAL), dtype=complex)

        # OFDM modulation:
        for user in range(n_users_int):
            for rx in range(n_input_rx_int):
                st_one_antenna = np.array([])
                for slot_num in range(NUM_SLOTS):
                    cp_length = FIRST_CP
                    for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                        cur_index = slot_num * NUM_SYMB_PER_SLOT + ofdm_symbol
                        s_t = np.fft.ifft(y[user, rx, cur_index, :], n=FFT_size)
                        s_t_with_cp = np.concatenate((s_t[-cp_length:], s_t))
                        st_one_antenna = np.concatenate((st_one_antenna, s_t_with_cp))
                        cp_length = CP
                st_full[user,rx, :] = st_one_antenna

        if (cfo != 0):
            n = np.arange(NUM_SAMPLES_PER_SLOT)
            cfo_phase = 2 * np.pi * cfo * n / FFT_size  # CFO phase shift
            cfo_phase = np.tile(cfo_phase,NUM_SLOTS)
            st_full = st_full * np.exp(1j * cfo_phase)

        if clip_percentage_in_tx<100:
            rms_value = np.mean(np.sqrt(np.mean(np.abs(st_full) ** 2, axis=2)))  # Compute RMS of the signal
            clip_level_12dB = rms_value * (10 ** (12 / 20))  # 12 dB above RMS
            clip_level = (clip_percentage_in_tx / 100) * clip_level_12dB  # Scale by percentage

            magnitude = np.abs(st_full)  # Compute magnitude
            phase = np.angle(st_full)  # Compute phase

            # Apply clipping to magnitude
            magnitude_clipped = np.minimum(magnitude, clip_level)

            # Reconstruct clipped signal with original phase
            st_full = magnitude_clipped * np.exp(1j * phase)
            new_rms_value = np.mean(np.sqrt(np.mean(np.abs(st_full) ** 2, axis=2)))  # Compute RMS of the signal
            st_full = st_full*rms_value/new_rms_value

        if iqmm_gain!=0 or iqmm_phase!=0:
            st_full = apply_iq_mismatch(st_full, iqmm_gain, iqmm_phase)

        if tdl_channel and td_in_rx:
            st_out, chan_out = TDLChannel.conv_cir(st_full,1,0, NUM_SLOTS, external_chan, seed)
            st_full = st_out
        else:
            chan_out = tf.zeros([0], dtype=tf.float32)

        y_out_pre = np.zeros((n_users_out_int,st_full.shape[1],y.shape[2],y.shape[3]), dtype=np.complex64)
        for user in range(n_users_out_int):
            for rx in range(st_full.shape[1]):
                pointer = 0
                index = 0
                for slot_num in range(NUM_SLOTS):
                    cp_length = FIRST_CP
                    for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                        s_t_no_cp = st_full[user, rx, pointer + cp_length:pointer + cp_length + FFT_size]
                        S_no_cp = np.fft.fft(s_t_no_cp, n=FFT_size)
                        y_out_pre[user, rx, index, :] = S_no_cp[:num_res]
                        pointer += (cp_length + FFT_size)
                        cp_length = CP
                        index += 1
        if td_in_rx:
            if y_in.ndim == 4: # Multiple users
                y_out = y_out_pre
            else:
                y_out = np.squeeze(y_out_pre, axis=0)
        else:
            y_out = np.squeeze(y_out_pre, axis=1)

        show_impair_tx = False
        if show_impair_tx:
            fig, axs = plt.subplots(4, 1, figsize=(8, 10))

            # --- Real part ---
            axs[0].stem(np.real(y_in[0, 0, :]), linefmt='b-', markerfmt='bo', basefmt=" ", label='Before')
            axs[0].stem(np.real(y_out[0, 0, :]), linefmt='r--', markerfmt='ro', basefmt=" ", label='After')
            axs[0].set_ylabel('I')
            axs[0].grid(True)
            axs[0].legend()

            # --- Imag part ---
            axs[1].stem(np.imag(y_in[0, 0, :]), linefmt='b-', markerfmt='bo', basefmt=" ", label='Before')
            axs[1].stem(np.imag(y_out[0, 0, :]), linefmt='r--', markerfmt='ro', basefmt=" ", label='After')
            axs[1].set_ylabel('Q')
            axs[1].grid(True)
            axs[1].legend()

            # --- Abs ---
            axs[2].stem(np.abs(y_in[0, 0, :]), linefmt='b-', markerfmt='bo', basefmt=" ", label='Before')
            axs[2].stem(np.abs(y_out[0, 0, :]), linefmt='r--', markerfmt='ro', basefmt=" ", label='After')
            axs[2].set_ylabel('Abs')
            axs[2].grid(True)
            axs[2].legend()

            # --- Constellation diagram ---
            axs[3].scatter(np.real(y_out.flatten()), np.imag(y_out.flatten()), color='r', alpha=0.5,
                           label='After')
            axs[3].scatter(np.real(y_in.flatten()), np.imag(y_in.flatten()), color='b', alpha=0.5,
                           label='Before')
            axs[3].set_xlabel('I')
            axs[3].set_ylabel('Q')
            axs[3].grid(True)
            axs[3].axis('equal')
            axs[3].legend()

            # --- Global title ---
            # fig.suptitle('Impairment effect with clipping = ' + str(clip_percentage_in_tx) + '%', fontsize=14)
            fig.suptitle('Impairment effect with IQMM gain = ' + str(conf.iqmm_gain) + 'dB, phase=' + str(conf.iqmm_phase) + '°', fontsize=14)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        return y_out, chan_out



    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, noise_var: float, num_res: int, cfo_and_iqmm_in_rx: bool, n_users: int, pilots_length: int) -> np.ndarray:
        """
        The MIMO SED Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel function
        :return: received word
        """

        empty_tf_tensor = tf.zeros([0], dtype=tf.float32)
        y = np.zeros((conf.n_ants, s.shape[1], num_res), dtype=complex)
        if not (conf.separate_pilots):
            y_ce = np.zeros((n_users, conf.n_ants, s.shape[1], num_res), dtype=complex)
        else:
            y_ce = np.zeros_like(y, dtype=complex)

        if conf.TDL_model[0] == 'N':

            for re_index in range(num_res):
                    conv = SEDChannel._compute_channel_signal_convolution(h[:,:,re_index], s[:,:,re_index])
                    y[:, :, re_index] = conv

                    if not(conf.separate_pilots):
                        all_values = list(range(n_users))
                        for user in range(n_users):
                            idx = np.setdiff1d(all_values, user)
                            s_cur_user = s[:, :, re_index].copy()
                            s_cur_user[idx,:] = 0
                            conv_ce = SEDChannel._compute_channel_signal_convolution(h[:,:,re_index], s_cur_user)
                            y_ce[user, :, :, re_index] = conv_ce
                    else:
                        s_separate_pilots = np.zeros((s.shape[0],s.shape[1]), dtype=complex)
                        for user in range(n_users):
                            s_separate_pilots[user,user::n_users] = s[user,user::n_users,re_index]
                        conv_ce = SEDChannel._compute_channel_signal_convolution(h[:, :, re_index], s_separate_pilots)
                        y_ce[:, :, re_index] = conv_ce


            if cfo_and_iqmm_in_rx and ((conf.cfo!=0) or (conf.iqmm_gain!=0) or (conf.iqmm_phase!=0)):
                y , _ = SEDChannel.apply_td_and_impairments(y, True, 0, 100, num_res, n_users, False, empty_tf_tensor, 0, 0, conf.channel_seed)
                y_ce , _ = SEDChannel.apply_td_and_impairments(y_ce, True, 0, 100, num_res, n_users, False, empty_tf_tensor, 0, 0, conf.channel_seed)

        else:

            pilot_chunk = int(pilots_length / np.log2(conf.mod_pilot))
            if conf.pilot_channel_seed < 0:
                y, channel_used = SEDChannel.apply_td_and_impairments(s, True, 0, 100, num_res, n_users, True, empty_tf_tensor, 0, 0, conf.channel_seed)
            else:
                y_pilot, channel_used_pilot = SEDChannel.apply_td_and_impairments(s[:,:pilot_chunk,:], True, 0, 100, num_res, n_users, True, empty_tf_tensor, 0, 0, conf.pilot_channel_seed)
                y_data, channel_used_data = SEDChannel.apply_td_and_impairments(s[:,pilot_chunk:,], True, 0, 100, num_res, n_users, True, empty_tf_tensor, 0, 0, conf.channel_seed)
                y = np.concatenate((y_pilot, y_data), axis=1)

            all_values = list(range(n_users))
            if not(conf.separate_pilots):
                for user in range(n_users):
                    idx = np.setdiff1d(all_values, user)
                    s_cur_user = s.copy()
                    s_cur_user[idx, :, :] = 0
                    conv_ce, _ = SEDChannel.apply_td_and_impairments(s_cur_user, True, 0, 100, num_res, 1, True, channel_used, 0, 0, conf.channel_seed)
                    y_ce[user, :, :, :] = conv_ce
            else:
                s_separate_pilots = np.zeros_like(s, dtype=complex)
                for user in range(n_users):
                    s_separate_pilots[user, user::n_users,:] = s[user, user::n_users, :]

                if conf.pilot_channel_seed < 0:
                    conv_ce, _ = SEDChannel.apply_td_and_impairments(s_separate_pilots, True, 0, 100, num_res, 1, True, channel_used, 0, 0, conf.channel_seed)
                else:
                    conv_ce_pilot, _ = SEDChannel.apply_td_and_impairments(s_separate_pilots[:,:pilot_chunk,:], True, 0, 100, num_res, 1, True, channel_used_pilot, 0, 0, conf.pilot_channel_seed)
                    conv_ce_data, _ = SEDChannel.apply_td_and_impairments(s_separate_pilots[:,pilot_chunk:,:], True, 0, 100, num_res, 1, True, channel_used_data, 0, 0, conf.channel_seed)
                    conv_ce = np.concatenate((conv_ce_pilot, conv_ce_data), axis=1)

                y_ce[:, :, :] = conv_ce


        if conf.iqmm_gain != 0 or conf.iqmm_phase != 0 or conf.cfo != 0:
            y , _ = SEDChannel.apply_td_and_impairments(y, True, conf.cfo, 100, num_res, n_users, False, empty_tf_tensor, conf.iqmm_gain, conf.iqmm_phase, conf.channel_seed)
            y_ce , _ = SEDChannel.apply_td_and_impairments(y_ce, True, conf.cfo, 100, num_res, n_users, False, empty_tf_tensor, conf.iqmm_gain, conf.iqmm_phase, conf.channel_seed)

        for re_index in range(num_res):
            w = np.sqrt(noise_var) * (np.random.randn(conf.n_ants, s.shape[1]) + 1j * np.random.randn(conf.n_ants, s.shape[1]))
            y[:, :, re_index] = y[:, :, re_index] + w
            if not conf.separate_pilots:
                for user in range(n_users):
                    y_ce[user,:, :, re_index] = y_ce[user,:, :, re_index] + w
            else:
                y_ce[:, :, re_index] = y_ce[:, :, re_index] + w

        if conf.plot_channel:
            symbol_to_plot = 0
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 4.8))
            axes[0].plot(20 * np.log10(np.abs(y[0, symbol_to_plot, :])), '-', color='g', label='Ant 0')
            axes[0].plot(20 * np.log10(np.abs(y[1, symbol_to_plot, :])), '-', color='r', label='Ant 1')
            axes[0].plot(20 * np.log10(np.abs(y[2, symbol_to_plot, :])), '-', color='k', label='Ant 2')
            axes[0].plot(20 * np.log10(np.abs(y[3, symbol_to_plot, :])), '-', color='b', label='Ant 3')
            axes[0].set_xlabel('Subcarrier')
            axes[0].set_ylabel('Amp (dB)')
            axes[0].set_title('TDL-'+conf.TDL_model+', delay spread='+str(int(round(float(conf.delay_spread)*1e9)))+' nsec', fontsize=10)
            axes[0].legend()
            axes[0].grid()

            axes[1].plot(20 * np.unwrap(np.angle(y[0, symbol_to_plot, :])), '-', color='g', label='Ant 0')
            axes[1].plot(20 * np.unwrap(np.angle(y[1, symbol_to_plot, :])), '-', color='r', label='Ant 1')
            axes[1].plot(20 * np.unwrap(np.angle(y[2, symbol_to_plot, :])), '-', color='k', label='Ant 2')
            axes[1].plot(20 * np.unwrap(np.angle(y[3, symbol_to_plot, :])), '-', color='b', label='Ant 3')
            axes[1].set_xlabel('Subcarrier')
            axes[1].set_ylabel('Phase (Rad)')
            axes[1].set_title('TDL-'+conf.TDL_model+', delay spread='+str(int(round(float(conf.delay_spread)*1e9)))+' nsec', fontsize=10)
            axes[1].legend()
            axes[1].grid()
            fig.tight_layout()
            plt.show()

            colors = ['g', 'r', 'k', 'b']
            user = max(conf.ber_on_one_user,0)
            for ant in range(conf.n_ants):
                plt.plot(
                    20 * np.log10(np.abs(y_ce[user, ant, symbol_to_plot, :])),
                    '-', color=colors[ant], label=f'Ant {ant}'
                )

            plt.xlabel('Subcarrier')
            plt.ylabel('Amp (dB)')
            plt.title(
                f'User {user}, TDL-{conf.TDL_model}, delay spread={int(round(float(conf.delay_spread) * 1e9))} nsec',
                fontsize=10)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
            pass

            # axes[0].plot(20 * np.log10(np.abs(y_ce[conf.ber_on_one_user,0, symbol_to_plot, :])), '-', color='g', label='Ant 0')
            # axes[0].plot(20 * np.log10(np.abs(y_ce[conf.ber_on_one_user,1, symbol_to_plot, :])), '-', color='r', label='Ant 1')
            # axes[0].plot(20 * np.log10(np.abs(y_ce[conf.ber_on_one_user,2, symbol_to_plot, :])), '-', color='k', label='Ant 2')
            # axes[0].plot(20 * np.log10(np.abs(y_ce[conf.ber_on_one_user,3, symbol_to_plot, :])), '-', color='b', label='Ant 3')
            # axes[0].set_xlabel('Subcarrier')
            # axes[0].set_ylabel('Amp (dB)')
            # axes[0].set_title('User '+ str(np.max(conf.ber_on_one_user,0)) + ', TDL-'+conf.TDL_model+', delay spread='+str(int(round(float(conf.delay_spread)*1e9)))+' nsec', fontsize=10)
            # axes[0].legend()
            # axes[0].grid()
            #
            # axes[1].plot(20 * np.unwrap(np.angle(y_ce[conf.ber_on_one_user,0, symbol_to_plot, :])), '-', color='g', label='Ant 0')
            # axes[1].plot(20 * np.unwrap(np.angle(y_ce[conf.ber_on_one_user,1, symbol_to_plot, :])), '-', color='r', label='Ant 1')
            # axes[1].plot(20 * np.unwrap(np.angle(y_ce[conf.ber_on_one_user,2, symbol_to_plot, :])), '-', color='k', label='Ant 2')
            # axes[1].plot(20 * np.unwrap(np.angle(y_ce[conf.ber_on_one_user,3, symbol_to_plot, :])), '-', color='b', label='Ant 3')
            # axes[1].set_xlabel('Subcarrier')
            # axes[1].set_ylabel('Phase (Rad)')
            # axes[1].set_title('User '+ str(np.max(conf.ber_on_one_user,0)) + ', TDL-'+conf.TDL_model+', delay spread='+str(int(round(float(conf.delay_spread)*1e9)))+' nsec', fontsize=10)
            # axes[1].legend()
            # axes[1].grid()
            # fig.tight_layout()
            # plt.show()

        return y,y_ce

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv