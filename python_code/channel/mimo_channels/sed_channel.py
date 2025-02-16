import numpy as np

from python_code import conf
from python_code.utils.constants import N_ANTS , PHASE_OFFSET, N_USERS

H_COEF = 0.8


class SEDChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, num_res: int, frame_ind: int, fading: bool, spatial: bool) -> np.ndarray:
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
            H_real = np.eye(H_real.shape[0])

        H_complex = np.zeros((H_real.shape[0], H_real.shape[1], num_res), dtype=complex)

        if num_res==1:
            complex_scalar = np.exp(1j * PHASE_OFFSET)
            H_complex [:,:,0]= np.array(H_real * complex_scalar, dtype=complex)
        else:
            for re_index in range(num_res):
                H_complex[:,:,re_index] = H_real*ChannelFreq[re_index]


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
    def transmit(s: np.ndarray, h: np.ndarray, snr: float, num_res: int) -> np.ndarray:
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
                y_ce[user,:, :, re_index] = conv_ce + w
        return y,y_ce

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
