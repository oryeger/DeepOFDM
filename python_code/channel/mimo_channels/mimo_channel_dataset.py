from typing import Tuple

import numpy as np
from numpy.random import default_rng

from python_code import conf
from python_code.channel.mimo_channels.sed_channel import SEDChannel
from python_code.channel.modulator import BPSKModulator
import commpy.modulation as mod
import matplotlib.pyplot as plt
import tensorflow as tf

from python_code.coding.ldpc_wrapper import LDPC5GCodec
from python_code.coding.crc_wrapper import CRC5GCodec




class MIMOChannel:
    def __init__(self, block_length: int, pilot_length: int, data_length: int, clip_percentage_in_tx: int, cfo_and_iqmm_in_rx: bool, n_users: int):
        self._block_length = block_length
        self._pilot_length = pilot_length
        self._data_length = data_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = n_users
        self._h_shape = [conf.n_ants, n_users]
        self.rx_length = conf.n_ants
        self.clip_percentage_in_tx = clip_percentage_in_tx
        self.cfo_and_iqmm_in_rx = cfo_and_iqmm_in_rx


    def _transmit(self, h: np.ndarray, noise_var: float, num_res: int, n_users: int, mod_data: int, ldpc_k: int, ldpc_n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        data_length = self._block_length - self._pilot_length
        if conf.mcs<=-1:
            tx_pilots = self._bits_generator.integers(0, 2, size=(self._pilot_length, n_users, num_res))
            tx_data = self._bits_generator.integers(0, 2, size=(data_length, n_users, num_res))
        else:
            if ldpc_k > 3824:
                crc_length = 24
            else:
                crc_length = 16
            codec = LDPC5GCodec(k=(ldpc_k+crc_length), n=ldpc_n)
            crc = CRC5GCodec(crc_length)
            tx_pilots = self._bits_generator.integers(0, 2, size=(self._pilot_length, n_users, num_res))
            tx_data_coded = np.zeros((n_users , data_length*num_res))
            num_slots = int(np.floor(data_length*num_res/ldpc_n))
            remainder = (data_length * num_res) % ldpc_n
            tx_data_uncoded = self._bits_generator.integers(0, 2, size=(n_users,num_slots*ldpc_n))
            for slot in range(num_slots):
                tx_data_crc = crc.encode(tx_data_uncoded[:,slot*ldpc_k:(slot+1)*ldpc_k])
                codewords = codec.encode(tx_data_crc)
                tx_data_coded[:,slot*ldpc_n:(slot+1)*ldpc_n] = codewords
            # Filling the remaining bits with random bits for the ber calculations
            tx_data_coded[:,(num_slots*ldpc_n):data_length*num_res] = self._bits_generator.integers(0, 2, size=(n_users,remainder))
            tx_data = tx_data_coded.reshape(conf.n_users, data_length, conf.num_res).transpose(1, 0, 2).astype(int)

        tx = np.concatenate([tx_pilots, tx_data])

        if conf.mod_pilot>0:
            mod_pilot = conf.mod_pilot
        else:
            mod_pilot = mod_data

        # modulation
        if mod_data == 2:
            s_pilots = BPSKModulator.modulate(tx_pilots.T)
            s_data = BPSKModulator.modulate(tx_data.T)
        else:
            s_pilots = np.zeros((n_users, int(tx_pilots.shape[0]/np.log2(mod_pilot)), num_res), dtype=complex)
            s_data = np.zeros((n_users, int(tx_data.shape[0]/np.log2(mod_data)), num_res), dtype=complex)
            qam = mod.QAMModem(mod_pilot)
            for user in range(n_users):
                for re_index in range(num_res):
                    tx_pilots_cur = tx_pilots[:,:,re_index]
                    s_pilots[user,:, re_index] = qam.modulate(tx_pilots_cur.T[user,:])
            qam = mod.QAMModem(mod_data)
            for user in range(n_users):
                for re_index in range(num_res):
                    tx_data_cur = tx_data[:,:,re_index]
                    s_data[user,:, re_index] = qam.modulate(tx_data_cur.T[user,:])

        # import pandas as pd
        #
        # def int_to_bits(n: int, k: int = 6) -> np.ndarray:
        #     """Return MSB->LSB bit vector of length k."""
        #     return np.array([(n >> (k - 1 - i)) & 1 for i in range(k)], dtype=int)
        #
        # rows = []
        # for idx in range(64):
        #     bits = int_to_bits(idx, k=6)  # 6 bits for 64-QAM
        #     bits_str = "".join(map(str, bits.tolist()))
        #
        #     # Try common shapes for modulate input
        #     # (1) row vector (1, k)
        #     try:
        #         sym = qam.modulate(bits)
        #     except Exception:
        #         # (2) column vector (k, 1)
        #         sym = qam.modulate(bits)
        #
        #     # sym might be scalar, 1-element array, or list
        #     if isinstance(sym, (list, tuple, np.ndarray)):
        #         sym_val = np.asarray(sym).reshape(-1)[0]
        #     else:
        #         sym_val = sym
        #
        #     rows.append({
        #         "index": idx,
        #         "bits": bits_str,
        #         "I": float(np.real(sym_val)),
        #         "Q": float(np.imag(sym_val)),
        #         "symbol": complex(sym_val),
        #     })
        #
        # df = pd.DataFrame(rows)
        # df

        s = np.concatenate([s_pilots, s_data], axis=1)

        # OryEger - constant tx symbol
        # if conf.plot_channel:
            # s = np.abs(s.real)
            # s = np.abs(s.real) + 1j * np.abs(s.imag)
            # assert not (True), "constant tx symbol"


        s_orig = np.copy(s)

        if not self.cfo_and_iqmm_in_rx:
            cfo_tx = conf.cfo
            iqmm_gain = conf.iqmm_gain
            iqmm_phase = conf.iqmm_phase
        else:
            cfo_tx = 0
            iqmm_gain = 0
            iqmm_phase = 0

        # OryEger - one tone
        if conf.plot_channel:
            s_temp = s.copy()
            s = np.zeros_like(s_temp)
            index = 4
            s[0, :, index] = s_temp[0, :, index]
            index = 12
            s[0, :, index] = s_temp[0, :, index]
            index = 20
            s[0, :, index] = s_temp[0, :, index]

        if (cfo_tx!=0) or (iqmm_gain!=0) or (iqmm_phase!=0) or (self.clip_percentage_in_tx<100) or conf.run_tdcnn:
            empty_tf_tensor = tf.zeros([0], dtype=tf.float32)
            s_o = s.copy()
            s, _,  = SEDChannel.apply_td_and_impairments(s, False, cfo_tx, self.clip_percentage_in_tx, num_res, n_users, False, empty_tf_tensor, iqmm_gain, iqmm_phase, conf.channel_seed)
            s_clean, _,  = SEDChannel.apply_td_and_impairments(s_o, False, cfo_tx, 100, num_res, n_users, False, empty_tf_tensor, 0, 0, conf.channel_seed)
        else:
            s_clean = None


        # if show_impair:
        #     plt.subplot(2,1,1)
        #     plt.plot( np.abs(s[0,0,:]), linestyle='-', color='b', label='After Clipping')
        #     plt.xlabel('Subcarriers')
        #     plt.ylabel('Before Impair')
        #     plt.grid()
        #     plt.title('Impairment effect')
        #
        # if show_impair:
        #     plt.subplot(2,1,2)
        #     plt.plot( np.abs(s[0,0,:]), linestyle='-', color='b', label='After Clipping')
        #     plt.xlabel('Subcarriers')
        #     plt.ylabel('After Impair')
        #     plt.grid()
        #     plt.show()
        show_impair = False
        if show_impair:
            fig, axs = plt.subplots(3, 1, figsize=(8, 10))

            # --- Real part ---
            axs[0].stem(np.real(s_orig[0, 0, :]), linefmt='b-', markerfmt='bo', basefmt=" ", label='Before CFO')
            axs[0].stem(np.real(s[0, 0, :]), linefmt='r--', markerfmt='ro', basefmt=" ", label='After CFO')
            axs[0].set_ylabel('I')
            axs[0].grid(True)
            axs[0].legend()

            # --- Imag part ---
            axs[1].stem(np.imag(s_orig[0, 0, :]), linefmt='b-', markerfmt='bo', basefmt=" ", label='Before CFO')
            axs[1].stem(np.imag(s[0, 0, :]), linefmt='r--', markerfmt='ro', basefmt=" ", label='After CFO')
            axs[1].set_ylabel('Q')
            axs[1].grid(True)
            axs[1].legend()

            # --- Constellation diagram ---
            axs[2].scatter(np.real(s.flatten()), np.imag(s.flatten()), color='r', alpha=0.5,
                           label='After CFO')
            axs[2].scatter(np.real(s_orig.flatten()), np.imag(s_orig.flatten()), color='b', alpha=0.5,
                           label='Before CFO')
            axs[2].set_xlabel('I')
            axs[2].set_ylabel('Q')
            axs[2].grid(True)
            axs[2].axis('equal')
            axs[2].legend()

            # --- Global title ---
            fig.suptitle('Impairment effect with cfo = ' + str(conf.cfo) + ' scs', fontsize=14)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        # (dim0, dim1, dim2) = s.shape
        # s_real = np.empty((dim0*2, dim1, dim2), dtype=s.real.dtype)
        # s_real[0::2, :, :] = s.real  # Real parts at even indices
        # s_real[1::2, :, :] = s.imag  # Imaginary parts at odd indices

        # pass through channel
        rx, rx_ce = SEDChannel.transmit(s=s, h=h, noise_var=noise_var, num_res=num_res, cfo_and_iqmm_in_rx=self.cfo_and_iqmm_in_rx, n_users=n_users, pilot_length=self._pilot_length)
        if conf.run_tdcnn:
            rx_clean, _ = SEDChannel.transmit(s=s_clean, h=h, noise_var=0, num_res=num_res,
                                            cfo_and_iqmm_in_rx=self.cfo_and_iqmm_in_rx, n_users=n_users,
                                            pilot_length=self._pilot_length)
            rx_clean = np.transpose(rx_clean, (1, 0, 2))
        else:
            rx_clean = None

        rx = np.transpose(rx, (1, 0, 2))
        if not(conf.separate_pilots):
            rx_ce_t = np.zeros((n_users,rx.shape[0],rx.shape[1],rx.shape[2]),dtype=complex)
            for user in range(n_users):
                rx_ce_t[user,:,:,:] = np.transpose(rx_ce[user,:,:,:], (1, 0, 2))
            rx_ce = rx_ce_t
        else:
            rx_ce = np.transpose(rx_ce, (1, 0, 2))

        s_orig = np.transpose(s_orig, (1, 0, 2))

        return tx, rx, rx_ce, s_orig, rx_clean

    def _transmit_and_detect(self, noise_var: float, num_res: int, index: int, n_users: int, mod_data: int, ldpc_k: int, ldpc_n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        h = SEDChannel.calculate_channel(conf.n_ants, n_users, num_res)
        tx, rx, rx_ce, s_orig, rx_clean = self._transmit(h, noise_var,num_res, n_users, mod_data, ldpc_k, ldpc_n)
        return tx, h, rx, rx_ce, s_orig, rx_clean
