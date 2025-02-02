from typing import Tuple

import numpy as np
from numpy.random import default_rng

from python_code import conf
from python_code.channel.mimo_channels.sed_channel import SEDChannel
from python_code.channel.modulator import BPSKModulator
from python_code.utils.constants import N_USERS, N_ANTS, MOD_PILOT, MOD_DATA
import commpy.modulation as mod



class MIMOChannel:
    def __init__(self, block_length: int, pilots_length: int, fading_in_channel: bool, spatial_in_channel: bool):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = N_USERS
        self._h_shape = [N_ANTS, N_USERS]
        self.rx_length = N_ANTS
        self.fading_in_channel = fading_in_channel
        self.spatial_in_channel = spatial_in_channel

    def _transmit(self, h: np.ndarray, snr: float, num_res: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tx_pilots = self._bits_generator.integers(0, 2, size=(self._pilots_length, N_USERS, num_res))
        tx_data = self._bits_generator.integers(0, 2, size=(self._block_length - self._pilots_length, N_USERS, num_res))
        tx = np.concatenate([tx_pilots, tx_data])

        # modulation
        if MOD_PILOT == 2:
            s_pilots = BPSKModulator.modulate(tx_pilots.T)
            s_data = BPSKModulator.modulate(tx_data.T)
        else:
            s_pilots = np.zeros((N_USERS, int(tx_pilots.shape[0]/np.log2(MOD_PILOT)), num_res), dtype=complex)
            s_data = np.zeros((N_USERS, int(tx_data.shape[0]/np.log2(MOD_DATA)), num_res), dtype=complex)
            qam = mod.QAMModem(MOD_PILOT)
            for user in range(N_USERS):
                for re_index in range(num_res):
                    tx_pilots_cur = tx_pilots[:,:,re_index]
                    s_pilots[user,:, re_index] = qam.modulate(tx_pilots_cur.T[user,:])
                    tx_data_cur = tx_data[:,:,re_index]
                    s_data[user,:, re_index] = qam.modulate(tx_data_cur.T[user,:])

        s = np.concatenate([s_pilots, s_data], axis=1)
        (dim0, dim1, dim2) = s.shape
        s_real = np.empty((dim0*2, dim1, dim2), dtype=s.real.dtype)
        s_real[0::2, :, :] = s.real  # Real parts at even indices
        s_real[1::2, :, :] = s.imag  # Imaginary parts at odd indices

        # pass through channel
        rx = SEDChannel.transmit(s=s, h=h, snr=snr, num_res=num_res)

        rx = np.transpose(rx, (1, 0, 2))
        s = np.transpose(s, (1, 0, 2))

        return tx, rx, s

    def _transmit_and_detect(self, snr: float, num_res: int, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        h = SEDChannel.calculate_channel(N_ANTS, N_USERS, num_res, index, self.fading_in_channel, self.spatial_in_channel)
        tx, rx, s_orig = self._transmit(h, snr,num_res)
        return tx, h, rx, s_orig
