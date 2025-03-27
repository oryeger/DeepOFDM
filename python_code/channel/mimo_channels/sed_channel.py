import numpy as np

from python_code import conf
from python_code.utils.constants import (N_ANTS , PHASE_OFFSET, INTERF_FACTOR, NUM_SYMB_PER_SLOT, FFT_size, FIRST_CP,
                                         CP, NUM_SAMPLES_PER_SLOT,NOISE_TO_CE)

H_COEF = 0.8


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


                # OryEger - adding more interference
        H_real = H_real * INTERF_FACTOR
        for i in range(H_real.shape[1]):
            H_real[0,i] = 1

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
    def apply_td_and_impairments(y_in, td_in_rx, go_to_td, cfo, num_res, n_users) -> np.ndarray:
        if go_to_td | (cfo != 0): # if cfo != 0 we must go to td
            if td_in_rx:
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
            else:
                NUM_SLOTS = int(y_in.shape[1] / NUM_SYMB_PER_SLOT)
                NUM_SAMPLES_TOTAL = int(NUM_SLOTS * NUM_SAMPLES_PER_SLOT)
                n_users_int = n_users
                y = np.expand_dims(y_in, axis=1)

            st_full = np.zeros((n_users_int, N_ANTS, NUM_SAMPLES_TOTAL), dtype=complex)

            # OFDM modulation:
            for user in range(n_users_int):
                for rx in range(y.shape[1]):
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

            # OFDM demodulation:
            y_out_pre = np.zeros_like(y)
            for user in range(n_users_int):
                for rx in range(y.shape[1]):
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
        else:
            y_out = y_in
        return y_out




    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float, num_res: int, go_to_td: bool, cfo: int, cfo_in_rx: bool, n_users: int) -> np.ndarray:
        """
        The MIMO SED Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel function
        :return: received word
        """
        y = np.zeros((N_ANTS,s.shape[1],num_res), dtype=complex)
        y_ce = np.zeros((n_users, N_ANTS, s.shape[1], num_res), dtype=complex)
        var = 10 ** (-0.1 * snr)
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

        if cfo_in_rx:
            y = SEDChannel.apply_td_and_impairments(y, True, go_to_td, cfo, num_res, n_users)
            y_ce = SEDChannel.apply_td_and_impairments(y_ce, True, go_to_td, cfo, num_res, n_users)

        for re_index in range(num_res):
            w = np.sqrt(var) * (np.random.randn(N_ANTS, s.shape[1]) + 1j * np.random.randn(N_ANTS, s.shape[1]))
            y[:, :, re_index] = y[:, :, re_index] + w
            if NOISE_TO_CE:
                for user in range(n_users):
                    # w = np.sqrt(var) * (np.random.randn(N_ANTS, s.shape[1]) + 1j * np.random.randn(N_ANTS, s.shape[1]))
                    y_ce[user,:, :, re_index] = y_ce[user,:, :, re_index] + w
        return y,y_ce

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
