from typing import Tuple

import numpy as np
import torch
from numpy.random import default_rng

from python_code import conf
from python_code.channel.mimo_channels.sed_channel import SEDChannel
from python_code.channel.modulator import BPSKModulator
import commpy.modulation as mod
import matplotlib.pyplot as plt
import tensorflow as tf

from python_code.coding.dmrs_pilots import (build_dmrs_tx_symbols, dmrs_known_tx, dmrs_layout,
                                             dmrs_reference_values, estimate_channel_from_dmrs,
                                             interleave_group_symbols)
from python_code.coding.ldpc_wrapper import LDPC5GCodec
from python_code.coding.crc_wrapper import CRC5GCodec
from python_code.coding.pilot_coding import encode_pilots
from python_code.utils.constants import DMRS_NUM_PAYLOAD_SYMB, DMRS_SYMBOL_LOCAL_IDX, NUM_SYMB_PER_SLOT
from python_code.utils.probs_utils import skip_indices





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


    def _transmit(self, h: np.ndarray, noise_var: float, num_res: int, n_users: int, mod_data: int, ldpc_k: int, ldpc_n: int, pilot_data_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

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
            # The 64QAM pilot-mix feature below overwrites pilot bits with 1s,
            # which would destroy codewords — so coded pilots require mix == 0.
            if getattr(conf, 'encode_pilots', False) and conf.make_64QAM_16QAM_percentage == 0:
                tx_pilots = encode_pilots(self._bits_generator, self._pilot_length,
                                          num_res, n_users, codec, crc, ldpc_k, ldpc_n)
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

        if conf.make_64QAM_16QAM_percentage == 50 and conf.mod_pilot == 64:  # 64QAM special case
            # For 64QAM (6 bits per symbol), divide pilot_length into three parts
            pilot_length = tx_pilots.shape[0]
            third_size = pilot_length // 3

            # First third: skip_indices with ratio 3
            indices_first = skip_indices(third_size, 3)
            tx_pilots[indices_first, :, :] = 1

            # Second third: skip_indices with ratio 1.5
            indices_second = skip_indices(third_size, 1.5) + third_size
            tx_pilots[indices_second, :, :] = 1

            # Third part: unchanged (full 64QAM), no modification needed

        elif not(conf.make_64QAM_16QAM_percentage == 50) and conf.make_64QAM_16QAM_percentage>0:
            indices = skip_indices(int(tx_pilots.shape[0] * conf.make_64QAM_16QAM_percentage / 100), pilot_data_ratio)
            tx_pilots[indices, :, :] = 1

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
            # OryEger
            # tx_data[:6, 0, 0] = [0, 1, 0, 1, 1, 1]

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


        if not self.cfo_and_iqmm_in_rx:
            cfo_tx = conf.cfo
            iqmm_gain = conf.iqmm_gain
            iqmm_phase = conf.iqmm_phase
        else:
            cfo_tx = 0
            iqmm_gain = 0
            iqmm_phase = 0

        # When plotting channel: use constant envelope (ones) across all subcarriers
        # so that rx / s_orig gives a clean channel estimate with no data interference
        if conf.plot_channel:
            s = np.ones_like(s)

        # conf.chanestmode == 'dmrs': interleave a standard-layout, comb-2 OCC/FCC in-slot DMRS
        # (2 symbols/slot, matching ekf.py's own embedded-pilot scheme exactly - see
        # coding/dmrs_pilots.py) into the payload before the one transmit() call below, instead of
        # relying on this block's own dedicated pilot region for CE. This must happen on the
        # combined payload `s` (pilot-equivalent + data together) and before it, so DMRS and
        # payload ride through the identical channel draw in a single transmit() call - two
        # separate transmit() calls would not be guaranteed to see the same TDL realization (see
        # sed_channel.py's own external_chan reuse for its internal genie-CE companion pass).
        chanestmode = getattr(conf, 'chanestmode', 'legacy')
        dmrs_known_tx_dict = None
        num_slots_dmrs = None
        if chanestmode == 'dmrs':
            total_payload_symbols = s.shape[1]
            if total_payload_symbols % DMRS_NUM_PAYLOAD_SYMB != 0:
                raise ValueError(
                    f"chanestmode='dmrs' needs pilot_length+data_length to divide evenly into whole "
                    f"slots of {DMRS_NUM_PAYLOAD_SYMB} payload symbols each (got {total_payload_symbols} "
                    f"payload symbols) - round conf.pilot_size/data_size accordingly.")
            num_slots_dmrs = total_payload_symbols // DMRS_NUM_PAYLOAD_SYMB
            dmrs_layout_dict = dmrs_layout(n_users, num_res)
            dmrs_ref_values = dmrs_reference_values(self._bits_generator, num_res)
            dmrs_known_tx_dict = dmrs_known_tx(dmrs_layout_dict, dmrs_ref_values, n_users)
            num_occasions = num_slots_dmrs * len(DMRS_SYMBOL_LOCAL_IDX)
            dmrs_s = build_dmrs_tx_symbols(dmrs_known_tx_dict, n_users, num_res, num_occasions)
            s = interleave_group_symbols(s, dmrs_s, num_slots_dmrs)

        s_orig = np.copy(s)

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

        H_est, noise_var_est = None, None
        if chanestmode == 'dmrs':
            # Genie CFO compensation (same math/ordering as evaluate.py's own inline GENIE_CFO
            # block, which normally runs on the caller side after this function returns) must
            # happen here, on the full interleaved (num_slots_dmrs*NUM_SYMB_PER_SLOT-symbol) grid,
            # BEFORE DMRS rows get stripped below - stripping first would leave payload rows
            # non-contiguous within their own slot, which this vector's per-symbol CP-relative
            # indexing assumes. evaluate.py skips its own copy of this block under
            # chanestmode='dmrs' for exactly this reason (see run_evaluate).
            cfo_comp = genie_cfo_comp_vector(num_slots_dmrs)
            if cfo_comp is not None:
                rx = rx * cfo_comp[:, None, None]
                if not conf.separate_pilots:
                    rx_ce = rx_ce * cfo_comp[None, :, None, None]
                else:
                    rx_ce = rx_ce * cfo_comp[:, None, None]

            # Per-group DMRS-based CE (one estimate_channel_from_dmrs call per conf.slots_per_group
            # consecutive slots, mirroring ekf.py's own grouping exactly), then broadcast each
            # group's H/noise_var out to every payload symbol in that group - H_est/noise_var_est
            # come back already per-symbol (same length as the stripped-down rx/s_orig below), so
            # evaluate.py's per-RE loop can index them exactly like rx/equalized with no
            # group-boundary logic of its own (see lmmse_equalize_with_H's per-symbol-H branch).
            dmrs_set = set(DMRS_SYMBOL_LOCAL_IDX)
            payload_local_idx = [i for i in range(NUM_SYMB_PER_SLOT) if i not in dmrs_set]
            payload_rows, dmrs_rows_by_slot = [], []
            for slot in range(num_slots_dmrs):
                base = slot * NUM_SYMB_PER_SLOT
                payload_rows.extend(base + i for i in payload_local_idx)
                dmrs_rows_by_slot.append([base + i for i in DMRS_SYMBOL_LOCAL_IDX])

            total_payload_symbols = num_slots_dmrs * DMRS_NUM_PAYLOAD_SYMB
            H_est = np.zeros((total_payload_symbols, num_res, conf.n_ants, n_users), dtype=complex)
            noise_var_est = np.zeros(total_payload_symbols, dtype=float)
            estimate_nv = not getattr(conf, 'override_noise_var', False)
            slots_per_group = max(1, int(getattr(conf, 'slots_per_group', 1)))
            payload_pos = 0
            for group_start in range(0, num_slots_dmrs, slots_per_group):
                group_end = min(group_start + slots_per_group, num_slots_dmrs)
                group_dmrs_rows = [row for slot in range(group_start, group_end) for row in dmrs_rows_by_slot[slot]]
                rx_dmrs_group = torch.from_numpy(rx[group_dmrs_rows])
                est = estimate_channel_from_dmrs(rx_dmrs_group, dmrs_known_tx_dict, conf.n_ants, num_res, n_users,
                                                  estimate_noise_var=estimate_nv)
                H_group = est['H'].numpy()
                group_payload_count = (group_end - group_start) * DMRS_NUM_PAYLOAD_SYMB
                H_est[payload_pos:payload_pos + group_payload_count] = H_group[None, :, :, :]
                if estimate_nv:
                    noise_var_est[payload_pos:payload_pos + group_payload_count] = est['noise_var_est']
                payload_pos += group_payload_count
            if not estimate_nv:
                noise_var_est[:] = noise_var

            rx = rx[payload_rows]
            s_orig = s_orig[payload_rows]
            rx_ce = rx_ce[:, payload_rows] if not conf.separate_pilots else rx_ce[payload_rows]

        return tx, rx, rx_ce, s_orig, rx_clean, H_est, noise_var_est

    def _transmit_and_detect(self, noise_var: float, num_res: int, index: int, n_users: int, mod_data: int, ldpc_k: int, ldpc_n: int, pilot_data_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        h = SEDChannel.calculate_channel(conf.n_ants, n_users, num_res)
        tx, rx, rx_ce, s_orig, rx_clean, H_est, noise_var_est = self._transmit(
            h, noise_var, num_res, n_users, mod_data, ldpc_k, ldpc_n, pilot_data_ratio)
        return tx, h, rx, rx_ce, s_orig, rx_clean, H_est, noise_var_est
