import numpy as np
from sympy import false

from python_code import conf
from python_code.utils.constants import (PHASE_OFFSET, NUM_SYMB_PER_SLOT, FFT_size, FIRST_CP,
                                         CP, NUM_SAMPLES_PER_SLOT,NOISE_TO_CE)

import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple



H_COEF = 0.8

from python_code.channel.sionna.TLD_channel import TDLChannel


class SEDChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_users: int, num_res: int, frame_ind: int, fading: bool, spatial: bool, delayspread: bool) -> np.ndarray:
        H_row = np.array([i for i in range(n_ant)])
        H_row = np.tile(H_row, [n_users, 1]).T
        H_column = np.array([i for i in range(n_users)])
        H_column = np.tile(H_column, [n_ant, 1])
        H_real = np.exp(-np.abs(H_row - H_column))
        # Frequency channel
        sigma = 5*num_res/4
        linear_phase_slope = 0.5*num_res/4
        freq_indexes = np.linspace(-num_res // 2, num_res // 2, num_res)
        ChannelFreq = np.exp(-0.5 * (freq_indexes / sigma) ** 2)
        ChannelFreq = np.exp(1j * freq_indexes * linear_phase_slope)*ChannelFreq
        ChannelFreq = ChannelFreq*num_res/np.sum(np.abs(ChannelFreq))


        if not spatial:
            if n_users != 1:
                H_real = np.eye(H_real.shape[0])
            else:
                H_real = np.zeros((H_real.shape[0],1))
                H_real[0,0] = 1


        H_real = H_real * conf.interf_factor
        for i in range(H_real.shape[1]):
            H_real[i,i] = 1

        # np.random.seed(42)
        # real_part = np.random.normal(0, 1, (n_ant, n_user))
        # imag_part = np.random.normal(0, 1, (n_ant, n_user))
        # H_real = real_part + 1j * imag_part

        # H_real = np.array([
        #     [0.8 + 0.2j, 0.75 + 0.25j, 0.78 + 0.22j, 0.79 + 0.21j],
        #     [0.79 + 0.21j, 0.8 + 0.2j, 0.75 + 0.25j, 0.78 + 0.22j],
        #     [0.78 + 0.22j, 0.79 + 0.21j, 0.8 + 0.2j, 0.75 + 0.25j],
        #     [0.75 + 0.25j, 0.78 + 0.22j, 0.79 + 0.21j, 0.8 + 0.2j]
        # ])
        #
        # reg_factor = 0.1
        # H_real = H_real + reg_factor*np.eye(H_real.shape[0])

        H_complex = np.zeros((H_real.shape[0], H_real.shape[1], num_res), dtype=complex)

        if num_res==1:
            complex_scalar = np.exp(1j * PHASE_OFFSET)
            H_complex [:,:,0]= np.array(H_real * complex_scalar, dtype=complex)
        else:
            for re_index in range(num_res):
                if delayspread:
                    H_complex[:,:,re_index] = H_real*ChannelFreq[re_index]
                else:
                    H_complex[:, :, re_index] = H_real

        if fading:
            for re_index in range(num_res):
                H_complex[:,:,re_index] = SEDChannel._add_fading(H_complex[:,:,re_index], n_ant, frame_ind)
        return H_complex

    @staticmethod
    def _add_fading(H: np.ndarray, n_ant: int, frame_ind: int) -> np.ndarray:
        degs_array = np.array([51, 39, 33, 21])
        fade_mat = H_COEF + (1 - H_COEF) * np.cos(2 * np.pi * frame_ind / degs_array)
        fade_mat = np.tile(fade_mat.reshape(1, -1), [n_ant, 1])
        return H * fade_mat

    @staticmethod
    def apply_td_and_impairments(y_in, td_in_rx, cfo, clip_percentage_in_tx, num_res, n_users, tdl_channel: bool, external_chan: tf.Tensor) -> Tuple[np.ndarray, tf.Tensor]:
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

        if tdl_channel:
            st_out, chan_out = TDLChannel.conv_and_noise(st_full,1,0, NUM_SLOTS, external_chan)
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

        return y_out, chan_out



    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float, num_res: int, go_to_td: bool, cfo: int, cfo_and_clip_in_rx: bool, n_users: int) -> np.ndarray:
        """
        The MIMO SED Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel function
        :return: received word
        """

        var = 10 ** (-0.1 * snr)
        empty_tf_tensor = tf.zeros([0], dtype=tf.float32)
        if conf.TDL_model[0] == 'N':
            y = np.zeros((conf.n_ants,s.shape[1],num_res), dtype=complex)
            y_ce = np.zeros((n_users, conf.n_ants, s.shape[1], num_res), dtype=complex)

            for re_index in range(num_res):
                conv = SEDChannel._compute_channel_signal_convolution(h[:,:,re_index], s[:,:,re_index])
                y[:, :, re_index] = conv

                all_values = list(range(n_users))
                for user in range(n_users):
                    idx = np.setdiff1d(all_values, user)
                    s_cur_user = s[:, :, re_index].copy()
                    s_cur_user[idx,:] = 0
                    conv_ce = SEDChannel._compute_channel_signal_convolution(h[:,:,re_index], s_cur_user)
                    y_ce[user, :, :, re_index] = conv_ce

            if cfo_and_clip_in_rx and ((cfo!=0) or (conf.clip_percentage_in_tx<100) or go_to_td ):
                y , _ = SEDChannel.apply_td_and_impairments(y, True, cfo, 100, num_res, n_users, False, empty_tf_tensor)
                y_ce , _ = SEDChannel.apply_td_and_impairments(y_ce, True, cfo, 100, num_res, n_users, False, empty_tf_tensor)

        else:


            y, channel_used = SEDChannel.apply_td_and_impairments(s, True, cfo, 100, num_res, n_users, True, empty_tf_tensor)
            y_ce = np.zeros((n_users, conf.n_ants, s.shape[1], num_res), dtype=complex)
            all_values = list(range(n_users))
            for user in range(n_users):
                idx = np.setdiff1d(all_values, user)
                s_cur_user = s.copy()
                s_cur_user[idx, :, :] = 0
                conv_ce, _ = SEDChannel.apply_td_and_impairments(s_cur_user, True, cfo, 100, num_res, 1, True, channel_used)
                y_ce[user, :, :, :] = conv_ce

        for re_index in range(num_res):
            w = np.sqrt(var) * (np.random.randn(conf.n_ants, s.shape[1]) + 1j * np.random.randn(conf.n_ants, s.shape[1]))
            y[:, :, re_index] = y[:, :, re_index] + w
            if NOISE_TO_CE:
                for user in range(n_users):
                    # w = np.sqrt(var) * (np.random.randn(conf.n_ants, s.shape[1]) + 1j * np.random.randn(conf.n_ants, s.shape[1]))
                    y_ce[user,:, :, re_index] = y_ce[user,:, :, re_index] + w


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
