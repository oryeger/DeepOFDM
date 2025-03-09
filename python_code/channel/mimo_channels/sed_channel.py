import numpy as np

from python_code import conf
from python_code.utils.constants import N_ANTS , PHASE_OFFSET, N_USERS, INTERF_FACTOR, NUM_SYMB_PER_SLOT, FFT_size, FIRST_CP, CP, NUM_SAMPLES_PER_SLOT

H_COEF = 0.8


class SEDChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, num_res: int, frame_ind: int, fading: bool, spatial: bool, delayspread: bool) -> np.ndarray:
        H_row = np.array([i for i in range(n_ant)])
        H_row = np.tile(H_row, [n_user, 1]).T
        H_column = np.array([i for i in range(n_user)])
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
            if N_USERS != 1:
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
    def transmit(s: np.ndarray, h: np.ndarray, snr: float, num_res: int, go_to_td: bool, cfo: int, cfo_in_rx: bool) -> np.ndarray:
        """
        The MIMO SED Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel function
        :return: received word
        """
        y = np.zeros((N_ANTS,s.shape[1],num_res), dtype=complex)
        y_ce = np.zeros((N_USERS, N_ANTS, s.shape[1], num_res), dtype=complex)
        var = 10 ** (-0.1 * snr)
        for re_index in range(num_res):
            conv = SEDChannel._compute_channel_signal_convolution(h[:,:,re_index], s[:,:,re_index])
            w = np.sqrt(var) * (np.random.randn(N_ANTS, s.shape[1]) + 1j * np.random.randn(N_ANTS, s.shape[1]))
            all_values = list(range(N_USERS))
            y[:,:,re_index] = conv + w
            for user in range(N_USERS):
                idx = np.setdiff1d(all_values, user)
                s_cur_user = s[:, :, re_index].copy()
                s_cur_user[idx,:] = 0
                conv_ce = SEDChannel._compute_channel_signal_convolution(h[:,:,re_index], s_cur_user)
                y_ce[user, :, :, re_index] = conv_ce

        if go_to_td | (cfo > 0): # if cfo > 0 we must go to td
            NUM_SLOTS = int(y_ce.shape[2] / NUM_SYMB_PER_SLOT)
            NUM_SAMPLES_TOTAL = int(NUM_SLOTS * NUM_SAMPLES_PER_SLOT)

            # OFDM modulation:
            st_full = np.zeros((N_USERS, N_ANTS, NUM_SAMPLES_TOTAL), dtype=complex)
            for user in range(N_USERS):
                for rx in range(N_ANTS):
                    st_one_antenna = np.array([])
                    for slot_num in range(NUM_SLOTS):
                        cp_length = FIRST_CP
                        for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                            cur_index = slot_num * NUM_SYMB_PER_SLOT + ofdm_symbol
                            s_t = np.fft.ifft(y_ce[user, rx, cur_index, :], n=FFT_size)
                            s_t_with_cp = np.concatenate((s_t[-cp_length:], s_t))
                            st_one_antenna = np.concatenate((st_one_antenna, s_t_with_cp))
                            cp_length = CP
                    st_full[user,rx, :] = st_one_antenna

            if (cfo > 0) & (cfo_in_rx==True):
                n = np.arange(st_full.shape[2])
                cfo_phase = 2 * np.pi * cfo * n / FFT_size  # CFO phase shift
                st_full = st_full * np.exp(1j * cfo_phase)

            # OFDM demodulation:
            y_ce_out = np.zeros_like(y_ce)
            for user in range(N_USERS):
                for rx in range(N_ANTS):
                    pointer = 0
                    index = 0
                    for slot_num in range(NUM_SLOTS):
                        cp_length = FIRST_CP
                        for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                            s_t_no_cp = st_full[user, rx, pointer + cp_length:pointer + cp_length + FFT_size]
                            S_no_cp = np.fft.fft(s_t_no_cp, n=FFT_size)
                            y_ce_out[user, rx, index, :] = S_no_cp[:num_res]
                            pointer += (cp_length + FFT_size)
                            cp_length = CP
                            index += 1
            y_ce = y_ce_out

        for re_index in range(num_res):
            for user in range(N_USERS):
                    y_ce[user,:, :, re_index] = y_ce[user,:, :, re_index] + w
        return y,y_ce

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
