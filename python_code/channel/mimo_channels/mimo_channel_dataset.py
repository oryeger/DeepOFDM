from typing import Tuple

import numpy as np
from numpy.random import default_rng

from python_code import conf
from python_code.channel.mimo_channels.sed_channel import SEDChannel
from python_code.channel.modulator import BPSKModulator
from python_code.utils.constants import N_USERS, N_ANTS, MOD_PILOT, MOD_DATA, NUM_SYMB_PER_SLOT
import commpy.modulation as mod
import matplotlib.pyplot as plt




class MIMOChannel:
    def __init__(self, block_length: int, pilots_length: int, fading_in_channel: bool, spatial_in_channel: bool, delayspread_in_channel: bool, clip_percentage_in_tx: int, cfo: int, go_to_td: bool):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = N_USERS
        self._h_shape = [N_ANTS, N_USERS]
        self.rx_length = N_ANTS
        self.fading_in_channel = fading_in_channel
        self.spatial_in_channel = spatial_in_channel
        self.delayspread_in_channel = delayspread_in_channel
        self.clip_percentage_in_tx = clip_percentage_in_tx
        self.cfo = cfo
        self.go_to_td = go_to_td


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

        s_orig = np.copy(s)

        if self.go_to_td:
            FFT_size = 1024
            FIRST_CP = 88
            CP = 72
            SAMPLING_RATE = 30.72e6
            NUM_SAMPLES_PER_SLOT = int(0.5e-3*SAMPLING_RATE)

            NUM_SLOTS = int(s.shape[1]/NUM_SYMB_PER_SLOT)
            NUM_SAMPLES_TOTAL = int(NUM_SLOTS * NUM_SAMPLES_PER_SLOT)

            # OFDM modulation:
            st_full = np.zeros((N_USERS,NUM_SAMPLES_TOTAL), dtype=complex)
            for user in range(N_USERS):
                st_one_user = np.array([])
                for slot_num in range(NUM_SLOTS):
                    cp_length =  FIRST_CP
                    for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                        cur_index = slot_num*NUM_SYMB_PER_SLOT + ofdm_symbol
                        s_t = np.fft.ifft(s[user, cur_index, :], n=FFT_size)
                        s_t_with_cp = np.concatenate((s_t[-cp_length:], s_t))
                        st_one_user = np.concatenate((st_one_user, s_t_with_cp))
                        cp_length = CP
                st_full[user,:] = st_one_user

            if self.cfo > 0:
                n = np.aranges(FFT_size)
                cfo_phase = 2 * np.pi * self.cfo * n / FFT_size  # CFO phase shift
                s_t = s_t * np.exp(1j * cfo_phase)
                s = np.fft.fft(s_t, axis=2, n=1024)
                s = s[:,:,:num_res]


            # OFDM demodulation:
            s_out = np.zeros_like(s)
            for user in range(N_USERS):
                pointer = 0
                index = 0
                for slot_num in range(NUM_SLOTS):
                    cp_length =  FIRST_CP
                    for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                        s_t_no_cp = st_full[user,pointer+cp_length:pointer+cp_length+FFT_size]
                        S_no_cp = np.fft.fft(s_t_no_cp, n=FFT_size)
                        s_out[user,index,:] =  S_no_cp[:num_res]
                        pointer += (cp_length + FFT_size)
                        cp_length = CP
                        index += 1
            s = s_out



        show_impair = False
        if show_impair:
            plt.subplot(2,1,1)
            plt.plot( np.abs(s[0,0,:]), linestyle='-', color='b', label='After Clipping')
            plt.xlabel('Subcarriers')
            plt.ylabel('Before Impair')
            plt.grid()
            plt.title('Impairment effect')

        if self.clip_percentage_in_tx<100:
            s_t = np.fft.ifft(s, axis=2, n=1024)
            # rms_value = np.abs(s_t) ** 2
            # rms_value = np.mean(rms_value, axis=2)
            # rms_value = np.sqrt(rms_value)
            rms_value = np.mean(np.sqrt(np.mean(np.abs(s_t) ** 2, axis=2)))  # Compute RMS of the signal
            clip_level_12dB = rms_value * (10 ** (12 / 20))  # 12 dB above RMS
            clip_level = (self.clip_percentage_in_tx / 100) * clip_level_12dB  # Scale by percentage

            magnitude = np.abs(s_t)  # Compute magnitude
            phase = np.angle(s_t)  # Compute phase

            # Apply clipping to magnitude
            magnitude_clipped = np.minimum(magnitude, clip_level)

            # Reconstruct clipped signal with original phase
            s_t = magnitude_clipped * np.exp(1j * phase)
            new_rms_value = np.mean(np.sqrt(np.mean(np.abs(s_t) ** 2, axis=2)))  # Compute RMS of the signal
            s_t = s_t*rms_value/new_rms_value
            s = np.fft.fft(s_t, axis=2, n=1024)
            s = s[:,:,:num_res]


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
        rx, rx_ce = SEDChannel.transmit(s=s, h=h, snr=snr, num_res=num_res)

        rx = np.transpose(rx, (1, 0, 2))
        rx_ce_t = np.zeros((N_USERS,rx.shape[0],rx.shape[1],rx.shape[2]),dtype=complex)
        for user in range(N_USERS):
            rx_ce_t[user,:,:,:] = np.transpose(rx_ce[user,:,:,:], (1, 0, 2))
        rx_ce = rx_ce_t
        s_orig = np.transpose(s_orig, (1, 0, 2))

        return tx, rx, rx_ce, s_orig

    def _transmit_and_detect(self, snr: float, num_res: int, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        h = SEDChannel.calculate_channel(N_ANTS, N_USERS, num_res, index, self.fading_in_channel, self.spatial_in_channel, self.delayspread_in_channel)
        tx, rx, rx_ce, s_orig = self._transmit(h, snr,num_res)
        return tx, h, rx, rx_ce, s_orig
