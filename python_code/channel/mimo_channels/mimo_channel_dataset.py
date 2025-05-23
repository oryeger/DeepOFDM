from typing import Tuple

import numpy as np
from numpy.random import default_rng

from python_code import conf
from python_code.channel.mimo_channels.sed_channel import SEDChannel
from python_code.channel.modulator import BPSKModulator
from python_code.utils.constants import N_ANTS
import commpy.modulation as mod
import matplotlib.pyplot as plt
import tensorflow as tf




class MIMOChannel:
    def __init__(self, block_length: int, pilots_length: int, fading_in_channel: bool, spatial_in_channel: bool, delayspread_in_channel: bool,
                 clip_percentage_in_tx: int, cfo: int, go_to_td: bool, cfo_and_clip_in_rx: bool, n_users: int):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = n_users
        self._h_shape = [N_ANTS, n_users]
        self.rx_length = N_ANTS
        self.fading_in_channel = fading_in_channel
        self.spatial_in_channel = spatial_in_channel
        self.delayspread_in_channel = delayspread_in_channel
        self.clip_percentage_in_tx = clip_percentage_in_tx
        self.cfo = cfo
        self.cfo_and_clip_in_rx = cfo_and_clip_in_rx
        self.go_to_td = go_to_td


    def _transmit(self, h: np.ndarray, snr: float, num_res: int, n_users: int, mod_pilot: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tx_pilots = self._bits_generator.integers(0, 2, size=(self._pilots_length, n_users, num_res))
        tx_data = self._bits_generator.integers(0, 2, size=(self._block_length - self._pilots_length, n_users, num_res))

        # OryEger - Just the inner constellation of 16QAM:

        if conf.mod_pilot>4:
            value = 1
            # if conf.ber_on_one_user>=0:
            #     tx_data[1::2, conf.ber_on_one_user] = value
            #     tx_pilots[1::2, conf.ber_on_one_user] = value

            # users_list = list(range(0, n_users))
            # filtered_users_list = [user for user in users_list if user != conf.ber_on_one_user]
            # for user in filtered_users_list:
            #     tx_pilots[1::2, user] = value
            #     tx_data[1::2, user] = value


        tx = np.concatenate([tx_pilots, tx_data])

        # modulation
        if mod_pilot == 2:
            s_pilots = BPSKModulator.modulate(tx_pilots.T)
            s_data = BPSKModulator.modulate(tx_data.T)
        else:
            s_pilots = np.zeros((n_users, int(tx_pilots.shape[0]/np.log2(mod_pilot)), num_res), dtype=complex)
            s_data = np.zeros((n_users, int(tx_data.shape[0]/np.log2(mod_pilot)), num_res), dtype=complex)
            qam = mod.QAMModem(mod_pilot)
            for user in range(n_users):
                for re_index in range(num_res):
                    tx_pilots_cur = tx_pilots[:,:,re_index]
                    s_pilots[user,:, re_index] = qam.modulate(tx_pilots_cur.T[user,:])
                    tx_data_cur = tx_data[:,:,re_index]
                    s_data[user,:, re_index] = qam.modulate(tx_data_cur.T[user,:])

        s = np.concatenate([s_pilots, s_data], axis=1)

        # OryEger - constant tx symbol
        # s = np.abs(s.real) + 1j * np.abs(s.imag)
        # s = np.abs(s.real)
        # assert not (True), "constant tx symbol"


        s_orig = np.copy(s)

        if not(self.cfo_and_clip_in_rx) and ((self.cfo!=0) or (self.clip_percentage_in_tx<100) or (self.go_to_td) ):
            empty_tf_tensor = tf.zeros([0], dtype=tf.float32)
            s, _ = SEDChannel.apply_td_and_impairments(s, False, self.cfo, self.clip_percentage_in_tx, num_res, n_users, False, empty_tf_tensor)

        show_impair = False
        if show_impair:
            plt.subplot(2,1,1)
            plt.plot( np.abs(s[0,0,:]), linestyle='-', color='b', label='After Clipping')
            plt.xlabel('Subcarriers')
            plt.ylabel('Before Impair')
            plt.grid()
            plt.title('Impairment effect')

        if show_impair:
            plt.subplot(2,1,2)
            plt.plot( np.abs(s[0,0,:]), linestyle='-', color='b', label='After Clipping')
            plt.xlabel('Subcarriers')
            plt.ylabel('After Impair')
            plt.grid()
            plt.show()
        pass



        # (dim0, dim1, dim2) = s.shape
        # s_real = np.empty((dim0*2, dim1, dim2), dtype=s.real.dtype)
        # s_real[0::2, :, :] = s.real  # Real parts at even indices
        # s_real[1::2, :, :] = s.imag  # Imaginary parts at odd indices

        # pass through channel
        rx, rx_ce = SEDChannel.transmit(s=s, h=h, snr=snr, num_res=num_res,go_to_td=self.go_to_td,cfo=self.cfo,cfo_and_clip_in_rx=self.cfo_and_clip_in_rx, n_users=n_users)

        rx = np.transpose(rx, (1, 0, 2))
        rx_ce_t = np.zeros((n_users,rx.shape[0],rx.shape[1],rx.shape[2]),dtype=complex)
        for user in range(n_users):
            rx_ce_t[user,:,:,:] = np.transpose(rx_ce[user,:,:,:], (1, 0, 2))
        rx_ce = rx_ce_t
        s_orig = np.transpose(s_orig, (1, 0, 2))

        return tx, rx, rx_ce, s_orig

    def _transmit_and_detect(self, snr: float, num_res: int, index: int, n_users: int, mod_pilot: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        h = SEDChannel.calculate_channel(N_ANTS, n_users, num_res, index, self.fading_in_channel, self.spatial_in_channel, self.delayspread_in_channel)
        tx, rx, rx_ce, s_orig = self._transmit(h, snr,num_res, n_users, mod_pilot)
        return tx, h, rx, rx_ce, s_orig
