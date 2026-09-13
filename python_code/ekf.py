#!/usr/bin/env python
"""
ekf.py -- Streaming per-group channel-drift + EKF/SGD parameter tracking.

A deliberately different loop from evaluate.py's run_evaluate(): no train/val/test
split, and the channel genuinely changes over the course of the run instead of
being fixed per block. Every group of data slots is treated as fully known (a
pilot) for BER scoring.

Channel estimation is in-band, 5G-style: every slot (both the scored data slots in
conf.slots_per_group, and - for weights_track_mode='sgd' only - the separate
conf.calib_slots_per_group training slots) carries its own embedded DMRS: 2 OFDM
symbols (PUSCH mapping-type-A style, comb-2 Type-1, up to 4 CDM-multiplexed ports -
see _dmrs_layout) instead of the whole slot being a known pilot. The other
NUM_SYMB_PER_SLOT-2=12 symbols of each slot carry real CRC+LDPC-coded payload bits
(scored, for the data region; trained on, for the sgd calib region - never both for
the same bits, so there's still no leakage between "what a region's own CE/training
sees" and "what gets judged"). This replaces an earlier design that spent a whole
extra calibration slot per group purely on LMMSE CE.

Unit of work is a *group*: conf.slots_per_group data slots, all sharing one channel
realization (channel_drift_base_index only advances between groups, not within
one). What happens to the data slots - and to whichever params escnn_load_freeze
leaves unfrozen - depends on conf.weights_track_mode:
  'ekf' (default) - unsupervised, syndrome-driven: the data slots get exactly one
      EKF predict() + one sequential update() per slot (see
      ESCNNTrainer.ekf_predict_update, unchanged - handing it a group's worth of
      data makes it resolve num_slots == group size on its own). There is no
      training step and no calib region at all in this mode.
  'sgd' - ordinary supervised training (ESCNNTrainer._online_training, the same code path
      evaluate.py's normal pilot training uses - conf.training_loss picks 'bce'/'tent'/
      'gfmi'/'tsyn' same as there) against a separate conf.calib_slots_per_group-sized
      region's own known, CRC+LDPC-coded payload bits (see encode_pilots below) - a
      ground-truth upper-bound baseline to compare the blind EKF against. The data
      slots are never trained on in this mode either, so both modes score identically
      on data neither has seen. That calib region has the exact same 2-DMRS+12-payload
      structure as the data region, and gets its own CE from its own embedded DMRS too
      (not a whole-slot-known-pilot estimate).
Setting slots_per_group=1 makes every data slot its own group, i.e. a new
channel every slot.

which_augment: DMRS is transmitted (and its CE computed, for the LMMSE baseline
metrics always reported below) regardless of which_augment - only whether that CE
is *fed into ESCNN* as a prior is conditional. 'NO_AUGMENT' and 'AUGMENT_LMMSE' are
supported (see main()); ESCNNTrainer's own NO_AUGMENT branch never reads probs_in,
so an empty tensor is passed there, matching evaluate.py's own convention.
AUGMENT_DEEPSIC/AUGMENT_DEEPRX would need a DeepSIC/DeepRx trainer wired into this
file too (evaluate.py runs a real separate forward pass for those), which isn't
implemented here - main() raises NotImplementedError rather than silently doing the
wrong thing.

Reuses, unmodified: ESCNNTrainer (construction/weight loading/freezing/EKF/
_online_training/_forward), EkfParamTracker, SyndromeLoss, SEDChannel (channel
generation, including the TDL channel_drift_base_index machinery), LmmseDemod,
encode_pilots, the LDPC/CRC codecs. Nothing here duplicates any of that - this file
is only the orchestration loop plus the DMRS pilot mechanism those pieces don't
otherwise support.

Usage:
    python -m python_code.ekf --config path/to/config.yaml

Relevant config keys (see config.yaml for the full list/defaults):
    load_escnn_weights_tag       - required: the pretrained checkpoint to track from
    mod_pilot                     - unsupported with DMRS (must be <= 0): the payload region
                                    carries real coded bits at num_bits_pilot == qm, no padded
                                    channels for a wider network to train/score against.
    escnn_load_freeze            - which params tracking (ekf or sgd) is allowed to move
    weights_track_mode            - 'ekf' (default, unsupervised syndrome EKF) or 'sgd'
                                    (supervised training on a separate calib region, loss
                                    picked by conf.training_loss - see module docstring above)
    calib_slots_per_group         - 'sgd' mode only: slots in the separate training-only region
                                    (2-DMRS+12-payload structure, same as the data region). Not
                                    used at all in 'ekf' mode. Default 1; raise well above 1 for
                                    'sgd' - a single slot's worth of bits is unlikely to move the
                                    weights via gradient descent. training_loss='tsyn' needs it
                                    raised further still: _train_model's train/val split rounds
                                    down to whole slots, so too few calib slots leaves zero slots
                                    on one side of the split
    escnn_ekf_*                  - EKF dynamics/noise/chunking (same knobs as the
                                    block-based evaluate.py path); 'ekf' mode only
    epochs                        - 'sgd' mode only: epochs of full training per group
                                    (same knob evaluate.py's own pilot training uses)
    slots_per_group               - slots per group (= slots per channel realization, and per CFO value)
    channel_drift_base_index     - starting slot offset into the TDL trajectory
    cfo                           - base/starting CFO (scs); constant within a group
    cfo_drift                     - CFO drift rate (scs/sec); advances cfo by cfo_drift *
                                    elapsed-seconds at each group boundary, can be negative
    delay_spread                  - RMS delay spread (s); feeds the DMRS frequency-domain
                                    denoise/interpolation's delay-domain truncation (see
                                    _estimate_channel_from_dmrs) - already set from channel_model
                                    (config_singleton.py), no new key needed
    pilot_size                   - total data budget in bits (this script's own run length,
                                    not the regular pass's pilot_size); truncated down to a
                                    whole number of groups, see main(). Named pilot_size, not
                                    data_size, because every slot here is a pilot - there's no
                                    separate "data" region to speak of
"""
import argparse
import glob
import os
from datetime import datetime
from typing import Tuple

import commpy.modulation as commpy_mod
import h5py
import numpy as np
import pandas as pd
import torch

from python_code import conf
from python_code.channel.mimo_channels.sed_channel import SEDChannel
from python_code.channel.modulator import BPSKModulator
from python_code.coding.crc_wrapper import CRC5GCodec
from python_code.coding.ldpc_wrapper import LDPC5GCodec
from python_code.coding.mcs_table import get_mcs
from python_code.coding.pilot_coding import encode_pilots
from python_code.detectors.escnn.escnn_trainer import ESCNNTrainer
from python_code.detectors.lmmse.lmmse_equalizer import LmmseDemod
from python_code.evaluate import calc_mi_from_ldpc, crc_fail_mask, resolve_auto_escnn_weights_tag
from python_code.utils.constants import (CP, DMRS_NUM_PAYLOAD_SYMB, DMRS_SYMBOL_LOCAL_IDX, FFT_size,
                                          FIRST_CP, GENIE_CFO, NUM_SAMPLES_PER_SLOT,
                                          NUM_SYMB_PER_SLOT, SAMPLING_RATE, SLOT_LENGTH_SEC)
from python_code.utils.probs_utils import relevant_indices

# M-QAM average-energy normalization constants (2*(M-1)/3), same table evaluate.py
# uses to turn an SNR into a noise variance.
CONSTELLATION_FACTOR = {2: 1, 4: 2, 16: 10, 64: 42, 256: 170}

# --- In-slot DMRS pilots (replaces the old dedicated-calibration-slot CE mechanism for the
# scored data region, and - for weights_track_mode='sgd' - for the calib region too; see module
# docstring). PUSCH mapping-type-A style: 2 OFDM symbols/slot, comb-2 Type-1, up to 4
# CDM-multiplexed ports (FD-OCC). These are concrete numbers from the design, not tunables, so
# they're kept as constants (in utils/constants.py, shared with escnn_trainer.py's syndrome/EKF
# code - see that module's DMRS_NUM_PAYLOAD_SYMB) rather than new config.yaml keys.
_DMRS_SYMBOL_LOCAL_IDX = DMRS_SYMBOL_LOCAL_IDX          # slot-local OFDM symbol indices carrying DMRS
_DMRS_NUM_PAYLOAD_SYMB = DMRS_NUM_PAYLOAD_SYMB          # 12
_DMRS_POWER_BOOST_DB = 3.0                   # extra pilot EPRE (dB) at n_users==1 only, where the
                                              # 'odd' comb sits completely unused for DMRS (see
                                              # _dmrs_known_tx/_dmrs_layout) - n_users in (2, 4)
                                              # already occupy every comb RE, so no spare budget to
                                              # redistribute and no boost
_DMRS_DELAY_TRUNC_MARGIN = 12                # L_taps = ceil(margin * delay_spread / delay_bin_width)


def _long_path(p: str) -> str:
    return ("\\\\?\\" + p) if os.name == 'nt' else p


def _genie_cfo_comp_vector(num_slots: int):
    """This group's genie CFO phase-compensation vector (one factor per OFDM symbol, length
    num_slots*NUM_SYMB_PER_SLOT), mirroring evaluate.py's GENIE_CFO path (~line 786): cancels
    the common per-symbol phase rotation using the true conf.cfo (the caller has already set
    this to the group's drifted value before transmitting), leaving the intra-symbol phase
    ramp - i.e. ICI - uncorrected by design, same as evaluate.py. Returns None (no-op) unless
    GENIE_CFO and conf.cfo != 0."""
    if not GENIE_CFO or conf.cfo == 0:
        return None
    n = np.arange(int(num_slots * NUM_SAMPLES_PER_SLOT))
    cfo_phase = -2 * np.pi * conf.cfo * n / FFT_size
    comp = []
    pointer = 0
    cp_length = FIRST_CP
    for _ in range(NUM_SYMB_PER_SLOT):
        pointer += cp_length + FFT_size // 2
        comp.append(np.exp(1j * cfo_phase[pointer]))
        pointer += FFT_size // 2
        cp_length = CP
    return np.tile(np.array(comp), num_slots)


def _fmt_count(n: int) -> str:
    """Compact tag=value form for a large integer count: whole thousands print as e.g. '20k'/'5k'
    instead of '20000'/'5000' (filenames here are already NTFS's 255-char component limit away
    from breaking - see the _bler.csv/_mi.csv write failures this shortening fixes); anything
    else (not a whole thousand, or < 1000) prints as-is."""
    if n >= 1000 and n % 1000 == 0:
        return f'{n // 1000}k'
    return str(n)


def _build_ekf_filename_suffix(chan_text: str, mod_text: str, n_users: int, code_rate) -> str:
    """Simplified analogue of evaluate.py's _build_escnn_filename_suffix: same spirit (readable
    tag=value pairs), but keeping only what this script actually uses. Deliberately drops
    train_samples, pilot_data_ratio, batch_size, block_length_factor, dropout, weight_decay,
    beta_balance, save-weights tag - none of those vary here. learning_rate is included for
    both modes (even though 'ekf' mode's tracking has no optimizer and never reads it - it's
    driven by the escnn_ekf_* noise/dynamics params already included above), so it's visible
    in the filename whenever a batch sweeps it. epochs, training_loss and tw are 'sgd'-only:
    'ekf' mode's tracking goes through ekf_predict_update's own syndrome measurement, never
    _calculate_loss/conf.training_loss, so they're genuinely meaningless there (no
    epoch-per-group or loss-function concept in EKF tracking)."""
    freeze_codes = {'none': 'n', 'scale': 'sc', 'first_conv': 'fc1', 'second_conv': 'fc2', 'last_conv': 'fc3',
                    'scale_only': 'so', 'last_conv_only': 'lco', 'first_conv_only': 'fco',
                    'first_conv_and_scale_only': 'fc1sco', 'all': 'a'}
    corr_map = {'none': 'No', 'low': 'Lo', 'medium': 'Med', 'medium_a': 'MedA', 'high': 'Hi', 'custom': 'Cust'}
    title_string = (f"{chan_text}_sp={conf.speed}_{mod_text}_REs={conf.num_res}_UEs={n_users}"
                     f"_ant={conf.n_ants}_cfo={conf.cfo:.2f}_cfod={getattr(conf, 'cfo_drift', 0.0):.2f}"
                     f"_iqg={getattr(conf, 'iqmm_gain', 0)}_iqp={getattr(conf, 'iqmm_phase', 0)}"
                     f"_Clp={conf.clip_percentage_in_tx}")
    title_string += '_C=' + corr_map.get(getattr(conf, 'spatial_correlation', 'none'), 'No')
    if conf.mcs > -1:
        title_string += f'_R={code_rate:.2f}'
    if conf.load_escnn_weights_tag:
        title_string += '_r=' + conf.load_escnn_weights_tag
    title_string += '_frz=' + freeze_codes.get(conf.escnn_load_freeze, conf.escnn_load_freeze)
    title_string += '_tf=' + str(getattr(conf, 'tsyn_fallback_iters', 0))
    title_string += '_dyn=' + getattr(conf, 'escnn_ekf_dynamics', 'ar1')
    title_string += '_a=' + str(getattr(conf, 'escnn_ekf_alpha', 0.99))
    title_string += '_sp0=' + str(getattr(conf, 'escnn_ekf_sigma_p0', 0.1))
    title_string += '_sq=' + str(getattr(conf, 'escnn_ekf_sigma_q', 0.01))
    title_string += '_sr=' + str(getattr(conf, 'escnn_ekf_sigma_r', 0.5))
    title_string += '_spg=' + str(getattr(conf, 'slots_per_group', 1))
    track_mode = getattr(conf, 'weights_track_mode', 'ekf')
    title_string += '_trk=' + track_mode
    if track_mode == 'sgd':
        # calib_slots_per_group is 'sgd'-training-only now (its old general-CE role is gone -
        # see module docstring) so it's only meaningful, and only printed, in that mode.
        title_string += '_csg=' + str(getattr(conf, 'calib_slots_per_group', 1))
    title_string += '_lr=' + str(getattr(conf, 'learning_rate', 5.0e-3))
    if track_mode == 'sgd':
        title_string += '_ep=' + str(getattr(conf, 'epochs', 100))
        loss_mode = getattr(conf, 'training_loss', 'bce')
        title_string += '_loss=' + loss_mode
        if loss_mode == 'tsyn':
            title_string += '_tw=' + str(getattr(conf, 'tw', 0.5))
    title_string += '_ps=' + _fmt_count(int(getattr(conf, 'pilot_size', -1)))
    zllr = getattr(conf, 'debug_zero_llr_res', [])
    if zllr:
        title_string += '_zllr=' + '-'.join(str(r) for r in zllr)
    title_string += '_' + conf.cur_str
    return title_string.replace(" ", "_")


def modulate_bits(tx_bits: np.ndarray, mod_order: int, n_users: int, num_res: int) -> np.ndarray:
    """tx_bits: (pilot_length, n_users, num_res) bits -> s: (n_users, num_symbols, num_res)
    complex symbols. Mirrors MIMOChannel._transmit's modulation block exactly, just
    factored out so it can be called per group instead of once per whole block."""
    if mod_order == 2:
        return BPSKModulator.modulate(tx_bits.transpose(1, 0, 2))
    pilot_length = tx_bits.shape[0]
    num_symbols = int(pilot_length / np.log2(mod_order))
    s = np.zeros((n_users, num_symbols, num_res), dtype=complex)
    qam = commpy_mod.QAMModem(mod_order)
    for user in range(n_users):
        for re in range(num_res):
            s[user, :, re] = qam.modulate(tx_bits[:, user, re])
    return s


def lmmse_equalize_with_H(H: torch.Tensor, rx_c: torch.Tensor, noise_var: float, re: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same linear-MMSE equalization math as LmmseEqualize (lmmse_equalizer.py lines 61-70),
    but against an H estimated elsewhere (from that region's own embedded DMRS here - see
    _estimate_channel_from_dmrs) instead of re-estimating it from rx_c itself - LmmseEqualize
    always re-estimates H from whatever it's given, which would leak the answer if rx_c were the
    same symbols being scored."""
    n_users = H.shape[1]
    I_users = torch.eye(n_users, dtype=H.dtype, device=H.device)
    W = torch.linalg.inv(H.T.conj() @ H + noise_var * I_users) @ H.T.conj()
    bias = (W @ H).diag().real
    W = W.cpu()
    bias = bias.cpu()
    equalized = torch.zeros(rx_c.shape[0], n_users, dtype=torch.cfloat)
    for i in range(rx_c.shape[0]):
        equalized[i, :] = torch.matmul(W, rx_c[i, :, re]) / bias
    postEqSINR = bias / (1 - bias)
    return equalized, postEqSINR


def load_pretrained_weights(escnn_trainer: ESCNNTrainer):
    """Mirrors evaluate.py's tag -> checkpoint-path lookup (evaluate.py ~line 618)."""
    if not conf.load_escnn_weights_tag:
        raise ValueError("ekf.py needs conf.load_escnn_weights_tag set - the EKF "
                          "tracks drift away from a pretrained checkpoint, it doesn't train "
                          "one from scratch. Train+save one first with evaluate.py "
                          "(save_escnn_weights: True), then point this run at that tag.")
    weights_load_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'Scratchpad', 'weights'))
    all_tag_matches = glob.glob(os.path.join(weights_load_dir, f'*_{conf.load_escnn_weights_tag}.pt'))
    if not all_tag_matches:
        raise FileNotFoundError(f"No saved ESCNN weights found for tag "
                                 f"'{conf.load_escnn_weights_tag}' in {weights_load_dir}")
    snr_override = getattr(conf, 'load_escnn_weights_snr_override', -1)
    desired_snr = snr_override if snr_override >= 0 else conf.snr
    weights_matches = [p for p in all_tag_matches if f'_SNR={desired_snr}_' in os.path.basename(p)]
    if not weights_matches:
        available_snrs = sorted({os.path.basename(p).split('_SNR=')[1].split('_')[0]
                                  for p in all_tag_matches if '_SNR=' in os.path.basename(p)})
        raise FileNotFoundError(f"No saved ESCNN weights for tag '{conf.load_escnn_weights_tag}' "
                                 f"at SNR={desired_snr}. Available SNRs: {available_snrs}. Set "
                                 f"load_escnn_weights_snr_override to pick one explicitly.")
    best_weights_path = max(weights_matches, key=lambda p: os.path.getmtime(_long_path(p)))
    escnn_trainer.load_weights(_long_path(best_weights_path))
    escnn_trainer.set_load_freeze(conf.escnn_load_freeze)
    print(f"[drift] loaded pretrained weights: {best_weights_path}", flush=True)


def transmit_symbols(s: np.ndarray, n_users: int, num_res: int, h: np.ndarray,
                      noise_var: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transmit an already-modulated symbol array (n_users, num_symbols, num_res) through
    SEDChannel (at whatever conf.channel_drift_base_index is currently set to) and return
    (rx, rx_ce, s_orig): complex128, symbol-major. rx_ce (the genie per-user channel-estimation
    array) is unused by the DMRS CE path below - _estimate_channel_from_dmrs works from the
    plain composite rx instead, since DMRS's CDM ports are genuinely superimposed on the same
    REs and need real deocc, not a per-user-interference-free shortcut - kept in the return only
    for signature parity."""
    s_orig = np.copy(s)
    rx, rx_ce = SEDChannel.transmit(s=s, h=h, noise_var=noise_var, num_res=num_res,
                                     cfo_and_iqmm_in_rx=conf.cfo_and_iqmm_in_rx,
                                     n_users=n_users, pilot_length=s.shape[1])
    # SEDChannel.transmit's TDL path returns rx as complex64 (via apply_td_and_impairments's
    # internal np.complex64 buffer) but rx_ce as complex128 (assigned into a dtype=complex
    # container, which upcasts it) - channel_dataset.py normally papers over this by
    # accumulating both into complex128 arrays before use; do the same here explicitly.
    rx = rx.astype(np.complex128)
    rx_ce = rx_ce.astype(np.complex128)
    rx = np.transpose(rx, (1, 0, 2))                        # (symbols, n_ants, num_res)
    s_orig = np.transpose(s_orig, (1, 0, 2))                 # (symbols, n_users, num_res)
    if conf.separate_pilots:
        rx_ce = np.transpose(rx_ce, (1, 0, 2))               # (symbols, n_ants, num_res)
        rx_ce = np.broadcast_to(rx_ce[None, :, :, :], (n_users,) + rx_ce.shape).copy()
    else:
        rx_ce_t = np.zeros((n_users, rx.shape[0], rx.shape[1], rx.shape[2]), dtype=complex)
        for user in range(n_users):
            rx_ce_t[user] = np.transpose(rx_ce[user], (1, 0, 2))
        rx_ce = rx_ce_t
    return rx, rx_ce, s_orig


def _generate_full_re_reference(rng: np.random.Generator, n_users: int, num_res: int) -> np.ndarray:
    """One random unit-energy QPSK value per RE per user - full frequency resolution, no comb
    decimation at all (diagnostic-only, see run_group's save_diag ground-truth pass)."""
    bits = rng.integers(0, 2, size=(2, n_users, num_res))
    return ((1 - 2 * bits[0]) + 1j * (1 - 2 * bits[1])) / np.sqrt(2)


def _ground_truth_channel(rng: np.random.Generator, n_users: int, n_ants: int, num_res: int,
                           h: np.ndarray) -> torch.Tensor:
    """Diagnostic-only (save_diag): a dense, full-resolution per-RE LS channel estimate that
    bypasses the DMRS comb/delay-domain machinery entirely - round-robin blocks of
    NUM_SYMB_PER_SLOT repeated symbols per user (only that user transmits, at the same known
    reference value on every RE for the whole block; everyone else silent, so each user's
    estimate is exactly interference-free), noise_var=0, direct per-RE division averaged over the
    block. A single symbol per user isn't long enough for the TDL/discrete-time channel
    convolution to settle into a valid steady state (confirmed empirically - it produced garbage:
    ~0 power, effectively random phase); a full slot's worth is the same duration every other
    transmission in this file already uses safely. This is the independent ground truth
    _estimate_channel_from_dmrs's output (both truncated and untruncated) should be compared
    against - the truncated-vs-untruncated comparison alone can only show what truncation itself
    changes, never whether either one matches the real channel. CFO compensation isn't applied
    here (this isn't a group-aligned transmission) - only meaningful for conf.cfo == 0 diagnostic
    runs. Returns (num_res, n_ants, n_users) complex128."""
    ref = _generate_full_re_reference(rng, n_users, num_res)
    s = np.zeros((n_users, n_users * NUM_SYMB_PER_SLOT, num_res), dtype=complex)
    for user in range(n_users):
        s[user, user * NUM_SYMB_PER_SLOT:(user + 1) * NUM_SYMB_PER_SLOT, :] = ref[user, :][None, :]
    rx, _, _ = transmit_symbols(s, n_users, num_res, h, 0.0)  # (n_users*NUM_SYMB_PER_SLOT, n_ants, num_res)
    H_gt = np.zeros((num_res, n_ants, n_users), dtype=complex)
    for user in range(n_users):
        block = rx[user * NUM_SYMB_PER_SLOT:(user + 1) * NUM_SYMB_PER_SLOT]  # (NUM_SYMB_PER_SLOT, n_ants, num_res)
        H_gt[:, :, user] = (block / ref[user, :][None, None, :]).mean(axis=0).T
    return torch.from_numpy(H_gt)


def _dmrs_layout(n_users: int, num_res: int) -> dict:
    """Per-user DMRS RE/OCC assignment (comb-2 Type-1 FD-OCC, up to 4 CDM-multiplexed ports -
    see module docstring). Returns {user: {'comb': 'even'|'odd', 'pairs': [(re_a, re_b_or_None), ...],
    'sign': (s_a, s_b_or_None), 'occ_active': bool}}. Pairing (for FD-OCC and for the shared
    reference sequence) groups consecutive same-comb REs two at a time; a trailing unpaired RE
    (only when a comb has an odd count) becomes a trivial single-port pair. 'occ_active' is True
    only when two users genuinely share a comb (n_users==4) - it tells
    _estimate_channel_from_dmrs whether Step 1 must deocc-combine each RE-pair into one estimate
    (spacing 4) or can treat every comb RE as its own independent pilot point (spacing 2,
    n_users in (1,2))."""
    if n_users not in (1, 2, 4):
        raise NotImplementedError(f"DMRS layout only supports n_users in (1, 2, 4); got {n_users}.")
    evens, odds = list(range(0, num_res, 2)), list(range(1, num_res, 2))

    def _pairs(comb_res):
        pairs = [(comb_res[i], comb_res[i + 1]) for i in range(0, len(comb_res) - 1, 2)]
        if len(comb_res) % 2 == 1:
            pairs.append((comb_res[-1], None))
        return pairs

    even_pairs, odd_pairs = _pairs(evens), _pairs(odds)
    layout = {}
    for user in range(n_users):
        if n_users in (1, 2):
            comb, pairs, sign = (('even', even_pairs, (1, 1)) if user == 0
                                  else ('odd', odd_pairs, (1, 1)))
        else:  # n_users == 4
            comb, pairs = ('even', even_pairs) if user in (0, 2) else ('odd', odd_pairs)
            sign = (1, 1) if user in (0, 1) else (1, -1)
        layout[user] = {'comb': comb, 'pairs': pairs, 'sign': sign, 'occ_active': n_users == 4}
    return layout


def _dmrs_reference_values(rng: np.random.Generator, num_res: int) -> dict:
    """One random unit-average-energy QPSK value per comb RE-pair - shared by both members of a
    pair (required for FD-OCC deocc to work; harmless for the non-OCC n_users in (1,2) case,
    where it just means both REs of a pair happen to carry the same known value). Independent of
    qm/mcs - closer to real DMRS (always QPSK regardless of the data's own modulation), and it
    means DMRS carries no codeword structure."""
    evens, odds = list(range(0, num_res, 2)), list(range(1, num_res, 2))
    n_even_pairs, n_odd_pairs = (len(evens) + 1) // 2, (len(odds) + 1) // 2

    def _qpsk(n):
        bits = rng.integers(0, 2, size=(2, n))
        return ((1 - 2 * bits[0]) + 1j * (1 - 2 * bits[1])) / np.sqrt(2)

    return {'even': _qpsk(n_even_pairs), 'odd': _qpsk(n_odd_pairs)}


def _dmrs_known_tx(layout: dict, ref_values: dict, n_users: int) -> dict:
    """Per user: {'occ_active': bool, 'entries': [(re_a, re_b_or_None, val_a, val_b_or_None), ...]}
    - the exact known complex symbol each of that user's DMRS REs carries (comb reference value
    x OCC sign x amplitude). DMRS is always QPSK, independent of whatever modulation the payload
    (mod_data) uses (module docstring), so amplitude is fixed at QPSK's own nominal per-RE energy
    (CONSTELLATION_FACTOR[4] - the same Es convention noise_var is derived from, just always at
    QPSK's value rather than the payload's) - plus the +3dB (linear sqrt(2) amplitude) boost only
    at n_users==1, where the 'odd' comb sits completely unused for DMRS (see _dmrs_layout) so the
    freed budget goes entirely into the one active port; at n_users==2 both combs are already
    occupied (one user each, no spare to redistribute), same as n_users==4's OCC-shared case, so
    neither gets a boost. Shared by _build_dmrs_tx_symbols (what gets transmitted) and
    _estimate_channel_from_dmrs (what the LS/deocc division uses)."""
    amp = np.sqrt(CONSTELLATION_FACTOR[4])  # DMRS's own (QPSK) reference energy, not mod_data's
    if n_users == 1:
        amp *= 10 ** (_DMRS_POWER_BOOST_DB / 20.0)  # dB -> linear amplitude factor
    out = {}
    for user, info in layout.items():
        ref = ref_values[info['comb']]
        sign_a, sign_b = info['sign']
        entries = []
        for pair_idx, (re_a, re_b) in enumerate(info['pairs']):
            val = amp * ref[pair_idx]
            val_a = sign_a * val
            val_b = (sign_b * val) if re_b is not None else None
            entries.append((re_a, re_b, val_a, val_b))
        out[user] = {'occ_active': info['occ_active'], 'entries': entries}
    return out


def _build_dmrs_tx_symbols(known_tx: dict, n_users: int, num_res: int, num_occasions: int) -> np.ndarray:
    """(n_users, num_occasions, num_res) complex DMRS tx symbols from _dmrs_known_tx's per-user
    known values - constant across every occasion (only the channel/noise differ between
    occasions), so Step 1's time-averaging in _estimate_channel_from_dmrs is coherent. No data is
    multiplexed onto any DMRS RE, including off-comb ones (module docstring) - everything else
    in the array stays zero."""
    s = np.zeros((n_users, num_occasions, num_res), dtype=complex)
    for user, info in known_tx.items():
        for re_a, re_b, val_a, val_b in info['entries']:
            s[user, :, re_a] = val_a
            if re_b is not None:
                s[user, :, re_b] = val_b
    return s


def _interleave_group_symbols(payload_s: np.ndarray, dmrs_s: np.ndarray, num_slots: int) -> np.ndarray:
    """Interleave a region's payload symbols (n_users, num_slots*_DMRS_NUM_PAYLOAD_SYMB, num_res)
    and DMRS symbols (n_users, num_slots*len(_DMRS_SYMBOL_LOCAL_IDX), num_res) into
    (n_users, num_slots*NUM_SYMB_PER_SLOT, num_res): DMRS at _DMRS_SYMBOL_LOCAL_IDX within every
    slot, payload at the other slot-local positions, in order - so the combined array is a
    genuine contiguous per-slot symbol sequence, letting the existing CP/CFO machinery
    (_genie_cfo_comp_vector) treat it exactly like any other slot-based transmission with no
    changes needed there."""
    n_users, _, num_res = payload_s.shape
    s = np.zeros((n_users, num_slots * NUM_SYMB_PER_SLOT, num_res), dtype=complex)
    dmrs_set = set(_DMRS_SYMBOL_LOCAL_IDX)
    payload_local_idx = [i for i in range(NUM_SYMB_PER_SLOT) if i not in dmrs_set]
    for slot in range(num_slots):
        base = slot * NUM_SYMB_PER_SLOT
        for j, local_idx in enumerate(payload_local_idx):
            s[:, base + local_idx, :] = payload_s[:, slot * _DMRS_NUM_PAYLOAD_SYMB + j, :]
        for j, local_idx in enumerate(_DMRS_SYMBOL_LOCAL_IDX):
            s[:, base + local_idx, :] = dmrs_s[:, slot * len(_DMRS_SYMBOL_LOCAL_IDX) + j, :]
    return s


def _edge_taper(length: int) -> np.ndarray:
    """Length-`length` array of 1s except the last quarter (min 1 tap), which raised-cosine
    tapers down to 0 - used to fight Gibbs ringing at the *discarded* edge of a kept delay-domain
    segment, without attenuating the (presumably larger) energy nearer delay 0."""
    w = np.ones(length)
    edge = max(1, length // 4)
    w[length - edge:] = 0.5 * (1 + np.cos(np.pi * np.arange(edge) / edge))
    return w


def _fold_delay_taps(h_alias: np.ndarray, num_res: int, causal_len: int, anticausal_len: int,
                      taper: bool) -> np.ndarray:
    """Place an M-point aliased delay-domain sequence into a full num_res-length array, split
    between its causal (near-zero, non-negative-delay) front and its *wrapped* tail - IFFT/DFT
    periodicity puts negative delays at the far end of h_alias (index M-1 = delay -1, M-2 = delay
    -2, ...; this is where a pulse-shaping filter's pre-cursor taps - e.g. TLD_channel.py's
    negative l_min - show up). Zero-padding by naively appending zeros after index
    causal_len+anticausal_len-1 would jam a hard discontinuity right against any real wrapped
    content, ringing across the whole spectrum on FFT - this is what the untruncated diagnostic
    surfaced (see _estimate_channel_from_dmrs). Correct placement: causal part goes at the front,
    anticausal part goes at the *end* of the full-length array, zeros in between.

    taper=True raised-cosine-tapers the *discarded* edge of each kept segment - the end farthest
    from delay 0 (only meaningful when causal_len/anticausal_len are a truncation, not the full M;
    the untruncated caller passes taper=False since nothing is being discarded)."""
    M = h_alias.shape[0]
    h_out = np.zeros((num_res,) + h_alias.shape[1:], dtype=complex)
    if causal_len > 0:
        w = _edge_taper(causal_len) if taper else np.ones(causal_len)
        h_out[:causal_len] = h_alias[:causal_len] * w[:, None]
    if anticausal_len > 0:
        w = _edge_taper(anticausal_len)[::-1] if taper else np.ones(anticausal_len)
        h_out[num_res - anticausal_len:] = h_alias[M - anticausal_len:] * w[:, None]
    return h_out


def _estimate_channel_from_dmrs(rx_dmrs: torch.Tensor, known_tx: dict, n_ants: int, num_res: int,
                                 n_users: int, return_untruncated: bool = False,
                                 estimate_noise_var: bool = False):
    """Per user, per antenna: LS+deocc at each DMRS pilot position (averaged over every DMRS
    occasion in the region - the two in-slot symbols x every slot in it), then an IFFT
    delay-domain denoise/interpolate to fill every RE. Returns (num_res, n_ants, n_users)
    complex128 (matching rx's own dtype - see transmit_symbols - so lmmse_equalize_with_H's
    matmul against rx_c doesn't hit a complex64/complex128 dtype mismatch), replacing what a
    dedicated-calibration-slot ChannelEstimate used to produce.

    Step 1 (LS + deocc + time-average): for occ_active users (n_users==4), each RE-pair is
    deocc-combined into one estimate (spacing-4 resolution across num_res - "3 missing REs out
    of 4"); otherwise (n_users in (1,2)) every comb RE gets its own independent estimate
    (spacing-2 - "every other RE") since there's no second port sharing it to separate out.

    Step 2 (delay domain): IFFT the user's M uniformly-spaced pilot estimates -> an aliased
    delay-domain estimate; white noise spreads evenly across all M delay bins while true channel
    energy concentrates within the delay spread, so truncating to L_taps (from conf.delay_spread)
    and zeroing the rest is a large, SNR-independent noise reduction. L_taps is split evenly
    between causal and anticausal (wrapped/pre-cursor - see _fold_delay_taps), the same ratio the
    untruncated path uses (there, an even split isn't a policy choice, it's the only correct one -
    see below) - an earlier draft here reserved only 1/4 of the budget for the anticausal side on
    the (real but overstated) assumption that pre-cursor content is much smaller than the main
    response; the ground-truth-vs-truncated diagnostic showed that assumption starving one side of
    the seam of resolution while the other side (which happened to get more of the budget) matched
    well - an artifact of the ratio, not of L_taps overall. Zero-pad back to num_res (correctly
    split, not just appended - _fold_delay_taps) and FFT -> full-resolution H, including at the
    original pilot REs (denoised the same as everywhere else, not left as their raw single-shot LS
    value).

    return_untruncated (diagnostic only - see run_group's save_diag path): also returns a second
    (num_res, n_ants, n_users) estimate from the *same* h_alias with no truncation/taper at all
    (L_taps=M, i.e. plain DFT interpolation of the raw pilot estimates, no denoising assumption) -
    split exactly at the Nyquist point M//2 (the only correct split when nothing is discarded,
    unlike L_taps's causal/anticausal ratio above, which is a truncation policy choice). Comparing
    the two on a noise_var=0 pass isolates exactly what the L_taps truncation choice is doing to
    the estimate, with noise out of the picture entirely.

    estimate_noise_var: also returns a receiver-side noise_var estimate (see noise_var_terms
    below) - the DMRS-pilot analog of what LmmseEqualize computes inline from its own LS
    estimate, for conf.override_noise_var=False to use instead of the genie theoretical
    noise_var (see run_group). Meaningless (and not requested) on the noise_var=0 save_diag
    pass, so this and return_untruncated are never both True in practice.

    Returns a dict: {'H': ..., and optionally 'H_untrunc'/'noise_var_est'} - a dict rather than
    a positional tuple since which optional fields are present depends on which of the two
    independent flags above is set."""
    delta_f = SAMPLING_RATE / FFT_size  # real subcarrier spacing (Hz)
    delay_bin = 1.0 / (num_res * delta_f)
    rx_np = rx_dmrs.cpu().numpy()  # (num_occasions, n_ants, num_res)
    H = np.zeros((num_res, n_ants, n_users), dtype=complex)
    H_untrunc = np.zeros((num_res, n_ants, n_users), dtype=complex) if return_untruncated else None
    # noise_var_terms: per-(user, pilot RE) residual of the raw, undivided per-occasion rx sample
    # around its own across-occasion mean (see estimate_noise_var branch below for why undivided) -
    # the same idea LmmseEqualize (lmmse_equalizer.py:74/80) uses for its noise_var, adapted to
    # this estimator's pilot layout. Averaged over ~num_res/2 pilot REs (all entries, all users)
    # below, not just the 2 DMRS occasions in isolation - see run_group's noise_var discussion for
    # why that matters (the per-RE residual alone is a noisy 1-2 degree-of-freedom estimate, but
    # pooling across REs meaningfully reduces the final noise_var_est's variance).
    noise_var_terms = [] if estimate_noise_var else None
    for user, info in known_tx.items():
        pair_est = []
        for re_a, re_b, val_a, val_b in info['entries']:
            if info['occ_active']:
                if re_b is not None:
                    est = 0.5 * (rx_np[:, :, re_a] / val_a + rx_np[:, :, re_b] / val_b)
                else:
                    est = rx_np[:, :, re_a] / val_a
                pair_est.append(est.mean(axis=0))
            else:
                est_a = rx_np[:, :, re_a] / val_a
                pair_est.append(est_a.mean(axis=0))
                if re_b is not None:
                    est_b = rx_np[:, :, re_b] / val_b
                    pair_est.append(est_b.mean(axis=0))
            if estimate_noise_var:
                # Residual on the RAW (undivided, uncombined) rx samples, not on est/est_a/est_b
                # above - dividing by val_a/val_b first would scale the residual by the DMRS
                # reference amplitude (which includes _DMRS_POWER_BOOST_DB and
                # CONSTELLATION_FACTOR[4]), silently deflating the noise_var estimate by that
                # factor squared. h_true*val is constant across occasions either way (block
                # fading + val doesn't vary per occasion - see _build_dmrs_tx_symbols), so
                # raw - mean(raw) equals noise - mean(noise) exactly regardless of amplitude or
                # FD-OCC combining, with no rescaling needed.
                for re_i in (re_a, re_b):
                    if re_i is not None:
                        raw = rx_np[:, :, re_i]
                        noise_var_terms.append(np.mean(np.abs(raw - raw.mean(axis=0, keepdims=True)) ** 2))
        pair_est = np.stack(pair_est, axis=0)              # (M, n_ants)
        M = pair_est.shape[0]

        h_alias = np.fft.ifft(pair_est, axis=0)             # (M, n_ants), aliased delay-domain estimate

        L_taps = int(np.clip(np.ceil(_DMRS_DELAY_TRUNC_MARGIN * conf.delay_spread / delay_bin), 1, M))
        anticausal_trunc = L_taps // 2
        causal_trunc = L_taps - anticausal_trunc
        h_trunc = _fold_delay_taps(h_alias, num_res, causal_trunc, anticausal_trunc, taper=True)
        H[:, :, user] = np.fft.fft(h_trunc, axis=0)          # (num_res, n_ants), full-resolution

        if return_untruncated:
            anticausal_full = M // 2
            causal_full = M - anticausal_full
            h_full = _fold_delay_taps(h_alias, num_res, causal_full, anticausal_full, taper=False)
            H_untrunc[:, :, user] = np.fft.fft(h_full, axis=0)

    result = {'H': torch.from_numpy(H)}
    if return_untruncated:
        result['H_untrunc'] = torch.from_numpy(H_untrunc)
    if estimate_noise_var:
        result['noise_var_est'] = float(np.mean(noise_var_terms)) if noise_var_terms else 0.0
    return result


def _generate_region_content(rng: np.random.Generator, num_slots: int, n_users: int, num_res: int,
                              qm: int, mod_data: int, layout: dict, codec: LDPC5GCodec,
                              crc: CRC5GCodec, ldpc_k: int, ldpc_n: int) -> dict:
    """Generate one region's worth of known content - num_slots slots of
    _DMRS_NUM_PAYLOAD_SYMB CRC+LDPC-coded payload symbols plus _DMRS_SYMBOL_LOCAL_IDX DMRS
    symbols per slot (see module docstring) - without transmitting it yet, so the exact same
    content can be sent twice (noisy, then noise_var=0 for save_diag) without re-drawing rng."""
    num_occasions = num_slots * len(_DMRS_SYMBOL_LOCAL_IDX)
    payload_bit_length = num_slots * _DMRS_NUM_PAYLOAD_SYMB * qm
    tx_bits = encode_pilots(rng, payload_bit_length, num_res, n_users, codec, crc, ldpc_k, ldpc_n)
    payload_s = modulate_bits(tx_bits, mod_data, n_users, num_res)
    ref_values = _dmrs_reference_values(rng, num_res)
    known_tx = _dmrs_known_tx(layout, ref_values, n_users)
    dmrs_s = _build_dmrs_tx_symbols(known_tx, n_users, num_res, num_occasions)
    s = _interleave_group_symbols(payload_s, dmrs_s, num_slots)
    return {'tx_bits': tx_bits, 'known_tx': known_tx, 's': s, 'num_slots': num_slots}


def _transmit_and_estimate(content: dict, n_ants: int, num_res: int, n_users: int, h: np.ndarray,
                            noise_var: float, return_untruncated: bool = False,
                            estimate_noise_var: bool = False) -> dict:
    """Transmit one region's content (from _generate_region_content) through SEDChannel, split
    the returned samples into payload rows and DMRS-occasion rows by slot-local symbol index,
    and estimate that region's own channel from its own DMRS (_estimate_channel_from_dmrs) -
    replaces both today's calib-block transmit+ChannelEstimate and the old data-block transmit.

    return_untruncated: diagnostic-only passthrough to _estimate_channel_from_dmrs - when True,
    the result dict also carries 'H_est_untrunc' (see run_group's save_diag path).
    estimate_noise_var: passthrough to _estimate_channel_from_dmrs - when True, the result dict
    also carries 'noise_var_est' (see run_group's conf.override_noise_var handling). Never
    combined with return_untruncated (see _estimate_channel_from_dmrs's docstring)."""
    num_slots = content['num_slots']
    rx, _, _ = transmit_symbols(content['s'], n_users, num_res, h, noise_var)
    cfo_comp = _genie_cfo_comp_vector(num_slots)
    if cfo_comp is not None:
        rx = rx * cfo_comp[:, None, None]
    dmrs_set = set(_DMRS_SYMBOL_LOCAL_IDX)
    payload_local_idx = [i for i in range(NUM_SYMB_PER_SLOT) if i not in dmrs_set]
    payload_rows, dmrs_rows = [], []
    for slot in range(num_slots):
        base = slot * NUM_SYMB_PER_SLOT
        payload_rows.extend(base + i for i in payload_local_idx)
        dmrs_rows.extend(base + i for i in _DMRS_SYMBOL_LOCAL_IDX)
    rx_payload = rx[payload_rows]
    rx_dmrs = torch.from_numpy(rx[dmrs_rows])
    result = {'tx_bits': content['tx_bits'], 'rx_payload': rx_payload}
    est = _estimate_channel_from_dmrs(rx_dmrs, content['known_tx'], n_ants, num_res, n_users,
                                       return_untruncated=return_untruncated,
                                       estimate_noise_var=estimate_noise_var)
    result['H_est'] = est['H']
    if return_untruncated:
        result['H_est_untrunc'] = est['H_untrunc']
    if estimate_noise_var:
        result['noise_var_est'] = est['noise_var_est']
    return result


def _llr_diag_line(snr, group_idx: int, slot: int, user: int, llr_scored: np.ndarray,
                    hard_user: np.ndarray, tx_user: np.ndarray, crc_fail: bool) -> str:
    """LMMSE-only diagnostic: recomputes the gmi estimator's term by hand (same formula
    _mi_from_flat's 'gmi' branch uses) over every RE this slot/user scored, to see whether a
    slot's MI hit is a broad shift or driven by a small tail of wrong-but-overconfident LLRs
    (the postEqSINR=bias/(1-bias) blow-up hypothesis). 'wrong' comes from the hard decision the
    code already computes, not from re-deriving the LLR's sign convention - self-consistent by
    construction since both are thresholded from the same real/imag part. worst_re is whichever
    RE has the single largest wrong-and-confident |LLR| in this slot/user - deliberately not
    assumed to be RE=0, since the goal is the general case across all REs. crc_fail is that same
    slot's actual LDPC decode outcome (crc_fail_mask) - printed here (not inferred separately)
    so these lines can be grepped/filtered to check whether outlier wrong-and-overconfident LLRs
    actually line up with real BLER failures, not just correlate with them in aggregate.

    n_wrong_res/max_per_re/top_re_share test a second, independent hypothesis once magnitude
    alone (wrong_term_share/worst_re) turned out not to cleanly separate crc=FAIL from crc=OK at
    matched wrong_frac: whether failures are driven by wrong bits *concentrated* on a handful of
    REs (a burst/trapping-set signature the LDPC's local redundancy can't outvote) rather than
    the same total error count spread diffusely across most REs (which ordinary belief
    propagation handles fine). max_per_re/top_re_share single out whichever RE carries the most
    wrong bits BY COUNT (not by |LLR|, unlike worst_re above) - a burst on one RE looks like
    top_re_share close to 1 even when wrong_frac is unremarkable."""
    abs_l = np.abs(llr_scored)
    wrong = (hard_user != tx_user)
    sign_agree = np.where(wrong, -1.0, 1.0)
    term = np.logaddexp(0.0, -sign_agree * abs_l) / np.log(2)
    gmi_diag = float(np.clip(1.0 - term.mean(), 0.0, 1.0))
    wrong_frac = float(wrong.mean())
    num_res = wrong.shape[-1]
    if wrong.any():
        wrong_l = abs_l[wrong]
        wrong_l_mean, wrong_l_max = float(wrong_l.mean()), float(wrong_l.max())
        wrong_term_share = float(term[wrong].sum() / max(term.sum(), 1e-300))
        wrong_l_per_re = np.where(wrong, abs_l, 0.0).max(axis=(0, 1))
        worst_re = int(wrong_l_per_re.argmax())
        worst_re_val = float(wrong_l_per_re[worst_re])
        wrong_count_per_re = wrong.sum(axis=(0, 1))
        total_wrong = int(wrong_count_per_re.sum())
        n_wrong_res = int((wrong_count_per_re > 0).sum())
        top_wrong_re = int(wrong_count_per_re.argmax())
        max_per_re = int(wrong_count_per_re[top_wrong_re])
        top_re_share = float(max_per_re / total_wrong)
    else:
        wrong_l_mean = wrong_l_max = wrong_term_share = worst_re_val = 0.0
        worst_re = -1
        n_wrong_res = max_per_re = top_wrong_re = 0
        top_re_share = 0.0
    return (f"[llr-diag] det=lmmse SNR={snr} group={group_idx} slot={slot} user={user} "
            f"crc={'FAIL' if crc_fail else 'OK'} gmi={gmi_diag:.4f} "
            f"wrong_frac={wrong_frac:.4e} |L|_mean={float(abs_l.mean()):.2f} "
            f"|L|_max={float(abs_l.max()):.2f} wrong|L|_mean={wrong_l_mean:.2f} "
            f"wrong|L|_max={wrong_l_max:.2f} wrong_term_share={wrong_term_share:.3f} "
            f"worst_re={worst_re}(|L|={worst_re_val:.2f}) "
            f"n_wrong_res={n_wrong_res}/{num_res} max_per_re={max_per_re} "
            f"top_re_share={top_re_share:.3f}(re={top_wrong_re})")


def run_group(escnn_trainer: ESCNNTrainer, codec: LDPC5GCodec, crc: CRC5GCodec, rng: np.random.Generator,
              qm: int, num_bits_pilot: int, mod_data: int, n_users: int, n_ants: int, num_res: int,
              ldpc_k: int, ldpc_n: int, noise_var: float, h: np.ndarray, group_size_slots: int,
              calib_slots: int, layout: dict, group_idx: int, base_index: int, base_cfo: float = 0.0,
              cfo_drift: float = 0.0, save_diag: bool = False) -> dict:
    """Generate, transmit, LMMSE-estimate/equalize, EKF/SGD-update, and score one group of
    group_size_slots consecutive slots sharing a single channel realization. Each slot carries
    its own embedded DMRS used to estimate H for that same slot's payload - no separate
    calibration slot in 'ekf' mode any more (see module docstring). 'sgd' mode still builds a
    separate calib_slots-sized region purely as supervised training data (it can't train and be
    scored on the same bits), built the exact same DMRS+payload way, with its own CE from its
    own embedded DMRS - not a whole-slot-known-pilot estimate like before.

    qm is the real data modulation (mcs); num_bits_pilot equals qm - DMRS's payload-only region
    needs real coded bits at the network's own bit-width, so the mod_pilot padding path older
    versions of this file (and evaluate.py) support isn't compatible here (see main()'s guard).

    save_diag gates the noise-free reference-channel diagnostics (re-transmits the data region's
    exact same content at noise_var=0), only worth paying for at the handful of SNRs in
    conf.save_loss_plot_snr - see main().

    base_index/base_cfo are passed in explicitly (not re-read from conf) because
    conf.channel_drift_base_index/conf.cfo are also the attributes TLD_channel.py/SEDChannel
    read at transmit time, which this function overwrites every call - re-reading them here
    instead of using the caller's fixed starting values would make each group's index/CFO
    accumulate on top of the previous group's, not the original base.

    cfo_drift (scs/sec) advances conf.cfo the same way base_index advances
    channel_drift_base_index: constant for every slot within this group, stepping by
    cfo_drift * elapsed-time-in-seconds at each group boundary."""
    elapsed_slots = group_idx * group_size_slots
    conf.set_value('channel_drift_base_index', base_index + elapsed_slots)
    conf.set_value('cfo', base_cfo + cfo_drift * elapsed_slots * SLOT_LENGTH_SEC)

    weights_track_mode = getattr(conf, 'weights_track_mode', 'ekf')
    which_augment = getattr(conf, 'which_augment', 'AUGMENT_LMMSE')

    data_content = _generate_region_content(rng, group_size_slots, n_users, num_res, qm, mod_data,
                                             layout, codec, crc, ldpc_k, ldpc_n)
    data_result = _transmit_and_estimate(data_content, n_ants, num_res, n_users, h, noise_var,
                                          estimate_noise_var=True)
    tx_bits, rx_data, H_data = data_result['tx_bits'], data_result['rx_payload'], data_result['H_est']
    rx_data_t = torch.from_numpy(rx_data)
    noise_var_est = data_result['noise_var_est']
    # Mirrors LmmseEqualize's own override_noise_var switch (lmmse_equalizer.py:83): True means
    # "trust the given (genie theoretical) noise_var", False means "use the receiver's own
    # pilot-residual estimate instead" - evaluate.py's LMMSE already behaves this way under the
    # same config, so ekf.py's should too (see module-level noise_var docstring note this fixes).
    lmmse_noise_var = noise_var if getattr(conf, 'override_noise_var', True) else noise_var_est

    num_symbols = rx_data.shape[0]
    pilot_data_ratio = 1.0  # mod_pilot padding unsupported with DMRS - see main()'s guard
    real_bit_idx = relevant_indices(num_bits_pilot, pilot_data_ratio)
    detected_word_lmmse = np.zeros((num_symbols * num_bits_pilot, n_users, num_res))
    llrs_mat_lmmse = np.zeros((num_symbols, num_bits_pilot * n_users, num_res, 1))
    h_abs_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    h_angle_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    h_abs_true_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    h_angle_true_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    h_abs_true_untrunc_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    h_angle_true_untrunc_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    h_abs_ground_truth_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    h_angle_ground_truth_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    sinr_per_re = np.zeros((num_res, n_users), dtype=np.float32)
    # (symbols, n_users, num_res) complex - the actual per-RE equalized samples LmmseDemod turns
    # into llrs_mat_lmmse, kept here only so run_group's caller can dump the whole LLR-generation
    # chain (equalized sample -> postEqSINR -> LLR) to disk for offline debugging - see main()'s
    # save_diag H5 write.
    equalized_lmmse = np.zeros((num_symbols, n_users, num_res), dtype=np.complex64)
    for re in range(num_res):
        equalized, postEqSINR = lmmse_equalize_with_H(H_data[re], rx_data_t, lmmse_noise_var, re)
        LmmseDemod(equalized, postEqSINR, qm, re, llrs_mat_lmmse, detected_word_lmmse, pilot_data_ratio)
        h_abs_per_re[re] = H_data[re].abs().cpu().numpy()
        h_angle_per_re[re] = H_data[re].angle().cpu().numpy()
        sinr_per_re[re] = postEqSINR.cpu().numpy()
        equalized_lmmse[:, :, re] = equalized.cpu().numpy()

    if save_diag:
        # Same content (tx_bits/DMRS reference values unchanged), noise_var=0 - a genuinely
        # noise-free reference H for comparison, mirroring the old rx_clean idiom.
        # return_untruncated=True also gets an L_taps=M (no truncation/taper) estimate from the
        # exact same noiseless pass - comparing it against H_true isolates what the truncation
        # choice itself is doing, with noise out of the picture entirely (see
        # _estimate_channel_from_dmrs's docstring - this is the truncated-vs-untruncated
        # diagnostic, not the noisy-vs-true one, which can't tell truncation bias from noise
        # since both noisy and true pass through the same truncation).
        data_result_true = _transmit_and_estimate(data_content, n_ants, num_res, n_users, h, 0.0,
                                                    return_untruncated=True)
        H_true, H_true_untrunc = data_result_true['H_est'], data_result_true['H_est_untrunc']
        # Independent ground truth (bypasses the DMRS comb/delay-domain machinery entirely - see
        # _ground_truth_channel's docstring): what the truncated-vs-untruncated comparison alone
        # can't tell us is whether either one matches the real channel, only what truncation
        # itself changes between them.
        H_gt = _ground_truth_channel(rng, n_users, n_ants, num_res, h)
        for re in range(num_res):
            h_abs_true_per_re[re] = H_true[re].abs().cpu().numpy()
            h_angle_true_per_re[re] = H_true[re].angle().cpu().numpy()
            h_abs_true_untrunc_per_re[re] = H_true_untrunc[re].abs().cpu().numpy()
            h_angle_true_untrunc_per_re[re] = H_true_untrunc[re].angle().cpu().numpy()
            h_abs_ground_truth_per_re[re] = H_gt[re].abs().cpu().numpy()
            h_angle_ground_truth_per_re[re] = H_gt[re].angle().cpu().numpy()

    # Diagnostic only: zero out LMMSE's LLRs at the given RE indices (e.g. RE 0, suspected of an
    # anomalous |H| - see plot_channel_diag.py) before they're used for anything downstream -
    # LDPC decoding (lmmse_stream, below) and the AUGMENT_LMMSE prior fed to ESCNN
    # (probs_for_aug, also below, since it's sigmoid(llrs_mat_lmmse)). Zeroing (not removing the
    # RE) keeps ldpc_n/the code rate unchanged - a zeroed LLR just tells the decoder "no
    # information here" for that RE's bits instead of the possibly-corrupted value it had.
    # conf.debug_zero_llr_res defaults to [] (no-op) - only set it to test this hypothesis.
    debug_zero_llr_res = getattr(conf, 'debug_zero_llr_res', [])
    if debug_zero_llr_res:
        if group_idx == 0:
            nonzero_before = int(np.count_nonzero(llrs_mat_lmmse[:, :, debug_zero_llr_res, :]))
        llrs_mat_lmmse[:, :, debug_zero_llr_res, :] = 0.0
        if group_idx == 0:
            print(f"[ekf] debug_zero_llr_res={debug_zero_llr_res}: zeroed {nonzero_before} "
                  f"nonzero LLR entries at those REs (group 0, out of "
                  f"{llrs_mat_lmmse[:, :, debug_zero_llr_res, :].size} total); "
                  f"all-zero after={bool(np.all(llrs_mat_lmmse[:, :, debug_zero_llr_res, :] == 0))}",
                  flush=True)

    rx_real = np.empty((num_symbols, n_ants * 2, num_res), dtype=np.float32)
    rx_real[:, 0::2, :] = rx_data.real.astype(np.float32)
    rx_real[:, 1::2, :] = rx_data.imag.astype(np.float32)
    rx_real_t = torch.from_numpy(rx_real)

    # DMRS is transmitted (and its CE computed, for the LMMSE baseline scoring below) regardless
    # of which_augment - only whether it's fed into ESCNN as a prior is conditional.
    # which_augment == 'NO_AUGMENT': ESCNNTrainer self-inits a flat-0.5 prior and never reads
    # probs_in in that branch (escnn_trainer.py's _forward/ekf_predict_update/_online_training),
    # so an empty tensor here (matching evaluate.py's own NO_AUGMENT convention) is safe.
    if which_augment == 'AUGMENT_LMMSE':
        probs_for_aug = torch.sigmoid(torch.tensor(llrs_mat_lmmse, dtype=torch.float32))
    else:
        probs_for_aug = torch.tensor([], dtype=torch.float32)

    if weights_track_mode == 'sgd':
        # escnn_frozen mirrors evaluate.py's own guard before calling _online_training (Adam
        # raises on an empty param list) - same "run the loaded weights statically" fallback
        # ekf_predict_update uses when escnn_load_freeze leaves nothing trainable.
        escnn_frozen = not any(p.requires_grad for nets in escnn_trainer.detector for net in nets
                                for p in net.parameters())
        if escnn_frozen:
            escnn_trainer._tsyn_warn_once('sgd_all_frozen', "escnn_load_freeze leaves nothing "
                                           "trainable - skipping SGD update (running loaded "
                                           "weights statically).", tag='sgd')
        else:
            # Separate calib_slots-sized region, built the same DMRS+payload way as the scored
            # data above - never scored, only trained on (see module docstring).
            calib_content = _generate_region_content(rng, calib_slots, n_users, num_res, qm,
                                                       mod_data, layout, codec, crc, ldpc_k, ldpc_n)
            calib_result = _transmit_and_estimate(calib_content, n_ants, num_res, n_users, h, noise_var,
                                                   estimate_noise_var=True)
            calib_tx_bits = calib_result['tx_bits']
            rx_calib, H_calib = calib_result['rx_payload'], calib_result['H_est']
            lmmse_noise_var_calib = (noise_var if getattr(conf, 'override_noise_var', True)
                                      else calib_result['noise_var_est'])
            num_calib_symbols = rx_calib.shape[0]
            rx_calib_t = torch.from_numpy(rx_calib)
            detected_word_lmmse_calib = np.zeros((num_calib_symbols * num_bits_pilot, n_users, num_res))
            llrs_mat_lmmse_calib = np.zeros((num_calib_symbols, num_bits_pilot * n_users, num_res, 1))
            for re in range(num_res):
                equalized_c, postEqSINR_c = lmmse_equalize_with_H(H_calib[re], rx_calib_t, lmmse_noise_var_calib, re)
                LmmseDemod(equalized_c, postEqSINR_c, qm, re, llrs_mat_lmmse_calib,
                           detected_word_lmmse_calib, pilot_data_ratio)
            rx_calib_real = np.empty((num_calib_symbols, n_ants * 2, num_res), dtype=np.float32)
            rx_calib_real[:, 0::2, :] = rx_calib.real.astype(np.float32)
            rx_calib_real[:, 1::2, :] = rx_calib.imag.astype(np.float32)
            rx_calib_real_t = torch.from_numpy(rx_calib_real)
            probs_for_aug_calib = torch.sigmoid(torch.tensor(llrs_mat_lmmse_calib, dtype=torch.float32))
            tx_calib_t = torch.from_numpy(calib_tx_bits.astype(np.float32))
            escnn_trainer._online_training(tx_calib_t, rx_calib_real_t, num_bits_pilot, n_users,
                                            conf.iterations, conf.epochs, False, probs_for_aug_calib)
    else:
        escnn_trainer.ekf_predict_update(rx_real_t, num_bits_pilot, n_users, conf.iterations, probs_for_aug,
                                          payload_symbols_per_slot=_DMRS_NUM_PAYLOAD_SYMB)
    _, llrs_mat_list = escnn_trainer._forward(rx_real_t, num_bits_pilot, n_users, conf.iterations, probs_for_aug)
    escnn_llrs = llrs_mat_list[-1].squeeze(-1).cpu().numpy()   # (symbols, num_bits_pilot*n_users, num_res)

    # BER (hard decisions) + build the LDPC-decoder-input LLR streams BLER/MI need below.
    # Stream layout matches SyndromeLoss/_syndrome_component's convention (symbol-major, then
    # bit-in-symbol, then RE), which is also encode_pilots' own tx_bits layout - so a plain
    # reshape(-1) lines both up with no reordering needed.
    ber_escnn_num = ber_escnn_den = ber_lmmse_num = ber_lmmse_den = 0
    ber_escnn_user, ber_lmmse_user = [], []
    escnn_stream = np.zeros((n_users, group_size_slots * ldpc_n))
    lmmse_stream = np.zeros((n_users, group_size_slots * ldpc_n))
    tx_stream = np.zeros((n_users, group_size_slots * ldpc_n))
    lmmse_llr_scored_by_user = [None] * n_users
    lmmse_user_by_user = [None] * n_users
    tx_user_by_user = [None] * n_users
    for user in range(n_users):
        tx_user = tx_bits[:, user, :].reshape(num_symbols, qm, num_res)

        escnn_user_llr = escnn_llrs[:, user * num_bits_pilot:(user + 1) * num_bits_pilot, :][:, real_bit_idx, :]
        escnn_user = (escnn_user_llr > 0).astype(int)
        n_err_escnn = int((escnn_user != tx_user).sum())
        ber_escnn_num += n_err_escnn
        ber_escnn_den += tx_user.size
        ber_escnn_user.append(n_err_escnn / tx_user.size)

        lmmse_user = detected_word_lmmse[:, user, :].reshape(num_symbols, num_bits_pilot, num_res)[:, real_bit_idx, :]
        n_err_lmmse = int((lmmse_user != tx_user).sum())
        ber_lmmse_num += n_err_lmmse
        ber_lmmse_den += tx_user.size
        ber_lmmse_user.append(n_err_lmmse / tx_user.size)

        escnn_stream[user] = escnn_user_llr.reshape(-1)
        lmmse_llr_full = llrs_mat_lmmse[:, user * num_bits_pilot:(user + 1) * num_bits_pilot, :, 0]
        lmmse_llr_scored = lmmse_llr_full[:, real_bit_idx, :]
        lmmse_stream[user] = lmmse_llr_scored.reshape(-1)
        tx_stream[user] = tx_user.reshape(-1)

        lmmse_llr_scored_by_user[user] = lmmse_llr_scored
        lmmse_user_by_user[user] = lmmse_user
        tx_user_by_user[user] = tx_user

    # BLER: per-slot LDPC decode + CRC check, same mechanism evaluate.py uses
    # (codec.decode -> crc.decode -> crc_fail_mask), over the data slots only. Also drives
    # [llr-diag] (LMMSE only - see _llr_diag_line), printed per (slot, user) here rather than
    # once per whole group so each line can be tagged with whether THIS slot's LDPC decode
    # actually failed CRC.
    symbols_per_slot = num_symbols // group_size_slots
    bler_escnn_fail = np.zeros(n_users, dtype=int)
    bler_lmmse_fail = np.zeros(n_users, dtype=int)
    for slot in range(group_size_slots):
        win = slice(slot * ldpc_n, (slot + 1) * ldpc_n)
        decoded_escnn = codec.decode(escnn_stream[:, win])
        bler_escnn_fail += crc_fail_mask(decoded_escnn, crc.decode(decoded_escnn)).astype(int)
        decoded_lmmse = codec.decode(lmmse_stream[:, win])
        lmmse_fail = crc_fail_mask(decoded_lmmse, crc.decode(decoded_lmmse))
        bler_lmmse_fail += lmmse_fail.astype(int)

        sym_win = slice(slot * symbols_per_slot, (slot + 1) * symbols_per_slot)
        for user in range(n_users):
            print(_llr_diag_line(conf.snr, group_idx, slot, user,
                                  lmmse_llr_scored_by_user[user][sym_win],
                                  lmmse_user_by_user[user][sym_win],
                                  tx_user_by_user[user][sym_win],
                                  bool(lmmse_fail[user])),
                  flush=True)
    bler_escnn_user = (bler_escnn_fail / group_size_slots).tolist()
    bler_lmmse_user = (bler_lmmse_fail / group_size_slots).tolist()

    # MI: genie-aided (ground-truth tx bits available since every slot is a known pilot),
    # from the same LDPC-decoder-input LLR/tx streams BLER used.
    mi_escnn_user = [calc_mi_from_ldpc(tx_stream, escnn_stream, user_idx=u) for u in range(n_users)]
    mi_lmmse_user = [calc_mi_from_ldpc(tx_stream, lmmse_stream, user_idx=u) for u in range(n_users)]

    return {
        'ber_escnn': ber_escnn_num / ber_escnn_den, 'ber_lmmse': ber_lmmse_num / ber_lmmse_den,
        'ber_escnn_user': ber_escnn_user, 'ber_lmmse_user': ber_lmmse_user,
        'bler_escnn': float(bler_escnn_fail.sum() / (group_size_slots * n_users)),
        'bler_lmmse': float(bler_lmmse_fail.sum() / (group_size_slots * n_users)),
        'bler_escnn_user': bler_escnn_user, 'bler_lmmse_user': bler_lmmse_user,
        'mi_escnn': float(calc_mi_from_ldpc(tx_stream, escnn_stream)),
        'mi_lmmse': float(calc_mi_from_ldpc(tx_stream, lmmse_stream)),
        'mi_escnn_user': mi_escnn_user, 'mi_lmmse_user': mi_lmmse_user,
        'num_symbols': num_symbols,
        'h_abs_per_re': h_abs_per_re, 'h_abs_true_per_re': h_abs_true_per_re,
        'h_angle_per_re': h_angle_per_re, 'h_angle_true_per_re': h_angle_true_per_re,
        'h_abs_true_untrunc_per_re': h_abs_true_untrunc_per_re,
        'h_angle_true_untrunc_per_re': h_angle_true_untrunc_per_re,
        'h_abs_ground_truth_per_re': h_abs_ground_truth_per_re,
        'h_angle_ground_truth_per_re': h_angle_ground_truth_per_re,
        'sinr_per_re': sinr_per_re,
        # Full LLR-generation chain for this group (LMMSE only), so the raw arrays behind
        # everything [llr-diag] summarizes can be dumped to disk instead of re-deriving more
        # scalar stats one at a time - see main()'s save_diag H5 write.
        'tx_bits': tx_bits,
        'llrs_mat_lmmse': llrs_mat_lmmse,
        'detected_word_lmmse': detected_word_lmmse,
        'equalized_lmmse': equalized_lmmse,
        'H_data': H_data.cpu().numpy() if hasattr(H_data, 'cpu') else H_data,
        # noise_var actually fed into this group's LMMSE math (conf.override_noise_var switches
        # between the genie theoretical noise_var and noise_var_est below - see this function's
        # override_noise_var handling above) plus the raw pilot-residual estimate itself, so a
        # run with override_noise_var=True can still see what the estimate WOULD have been.
        'noise_var_est': noise_var_est,
        'lmmse_noise_var': lmmse_noise_var,
    }


def main():
    parser = argparse.ArgumentParser(description='Streaming channel-drift + EKF/SGD parameter tracking')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    args = parser.parse_args()

    conf.reload_config(args.config)
    resolve_auto_escnn_weights_tag()
    # DMRS is transmitted (and its CE computed for the always-reported LMMSE baseline) regardless
    # of which_augment - only whether ESCNN gets fed that CE as a prior depends on the mode (see
    # module docstring/run_group). Only NO_AUGMENT/AUGMENT_LMMSE are wired up here.
    which_augment = getattr(conf, 'which_augment', 'AUGMENT_LMMSE')
    if which_augment not in ('NO_AUGMENT', 'AUGMENT_LMMSE'):
        raise NotImplementedError(f"ekf.py's DMRS-based CE only supports which_augment in "
                                   f"('NO_AUGMENT', 'AUGMENT_LMMSE') so far - {which_augment!r} "
                                   f"would need a DeepSIC/DeepRx trainer wired into this file too "
                                   f"(evaluate.py runs a real separate forward pass for those), "
                                   f"which isn't implemented here yet.")

    n_users, n_ants, num_res = conf.n_users, conf.n_ants, conf.num_res
    qm, code_rate = get_mcs(conf.mcs)
    qm = int(qm)
    mod_data = int(2 ** qm)
    # Unified codeword sizing: every region (scored data, and - 'sgd' only - the separate calib
    # region) now has the same 2-DMRS + _DMRS_NUM_PAYLOAD_SYMB-payload per-slot structure, so one
    # (ldpc_n, ldpc_k) covers both - no second codec needed.
    ldpc_n = int(num_res * _DMRS_NUM_PAYLOAD_SYMB * qm)
    ldpc_k = int(ldpc_n * code_rate)
    crc_length = 24 if ldpc_k > 3824 else 16
    codec = LDPC5GCodec(k=ldpc_k + crc_length, n=ldpc_n)
    crc = CRC5GCodec(crc_length)
    rng = np.random.default_rng(seed=conf.seed)

    # num_bits_pilot: DMRS's payload REs carry real coded bits at the network's own bit-width, so
    # (unlike evaluate.py's AUGMENT_LMMSE path) there's no padded-channel mechanism to size a
    # wider network against a smaller real modulation - mod_pilot isn't supported here.
    if getattr(conf, 'mod_pilot', -1) > 0:
        raise NotImplementedError("ekf.py's DMRS-based CE needs num_bits_pilot == qm (payload REs "
                                   "carry real coded bits, not padded LLR=0/prob=0.5 channels) - "
                                   "mod_pilot padding isn't supported. Set mod_pilot <= 0.")
    num_bits_pilot = qm

    layout = _dmrs_layout(n_users, num_res)

    noise_var = 10 ** (-0.1 * conf.snr) * CONSTELLATION_FACTOR[mod_data]
    h = SEDChannel.calculate_channel(n_ants, n_users, num_res)
    # Same SNR whitelist evaluate.py uses to gate its (also expensive) per-SNR loss/LLR plots
    # (evaluate.py:2341) - the per-RE channel/SINR diagnostics are only worth the extra
    # transmit+estimate and the H5 file at those SNRs, not every SNR in a sweep.
    save_diag = conf.snr in getattr(conf, 'save_loss_plot_snr', [])

    escnn_trainer = ESCNNTrainer(num_bits_pilot, n_users, n_ants)
    escnn_trainer._initialize_detector(num_bits_pilot, n_users, n_ants)
    load_pretrained_weights(escnn_trainer)

    group_size_slots = max(1, int(getattr(conf, 'slots_per_group', 1)))
    base_index = int(getattr(conf, 'channel_drift_base_index', 0))
    base_cfo = float(conf.cfo)
    cfo_drift = float(getattr(conf, 'cfo_drift', 0.0))

    # Printed unconditionally (not just when non-default) so a run where this was meant to be set
    # but wasn't (stale config, unsynced code) is visible in the log rather than silently absent.
    print(f"[ekf] debug_zero_llr_res={getattr(conf, 'debug_zero_llr_res', [])}", flush=True)

    # pilot_size (bits) -> OFDM symbols (// qm) -> whole groups
    # (// (NUM_SYMB_PER_SLOT * group_size_slots)). Deliberately conf.pilot_size, not
    # conf.data_size: every slot in this script is fully known (a pilot) - there's no "data"
    # region at all - so pilot_size is the config key that actually names what this run
    # length is. evaluate.py's data_size is untouched by this and keeps its own meaning there
    # (its own pilot_size, plus data_size either explicit or derived from
    # pilot_size*(block_length_factor-1)). And unlike evaluate.py's get_next_divisible (which
    # rounds the bit count UP so nothing is lost), this truncates DOWN: pilot_size here is a
    # budget, and running past it isn't wanted, so any leftover data that doesn't fill a
    # complete group is simply discarded. Every physical OFDM symbol still counts here (even
    # though 2/14 of each slot's symbols are DMRS, not payload) - pilot_size is the run's
    # wall-clock/slot budget, not a payload-bit budget.
    pilot_size_bits = int(getattr(conf, 'pilot_size', -1))
    if pilot_size_bits <= 0:
        raise ValueError(f"pilot_size={pilot_size_bits} - ekf.py needs pilot_size set > 0 "
                          f"explicitly (bits) - it's this script's whole data budget, since "
                          f"every slot here is a pilot.")
    symbols_per_group = NUM_SYMB_PER_SLOT * group_size_slots
    num_symbols_total = pilot_size_bits // qm
    num_groups = num_symbols_total // symbols_per_group
    used_symbols = num_groups * symbols_per_group
    if used_symbols * qm < pilot_size_bits:
        print(f"[drift] pilot_size={pilot_size_bits} bits -> {num_symbols_total} symbols -> "
              f"{num_groups} whole group(s) of {symbols_per_group} symbols each; discarding "
              f"{pilot_size_bits - used_symbols * qm} leftover bits that don't fill a full group.",
              flush=True)
    if num_groups == 0:
        raise ValueError(f"pilot_size={pilot_size_bits} bits ({num_symbols_total} symbols) isn't "
                          f"enough for even one group of {symbols_per_group} symbols "
                          f"(slots_per_group={group_size_slots} slots); raise pilot_size "
                          f"or lower slots_per_group.")

    weights_track_mode = getattr(conf, 'weights_track_mode', 'ekf')
    calib_slots_per_group = max(1, int(getattr(conf, 'calib_slots_per_group', 1)))
    calib_note = (f", {calib_slots_per_group} calib slot(s)/group for sgd training"
                  if weights_track_mode == 'sgd' else "")
    print(f"[drift] {num_groups} groups x {group_size_slots} slot(s)/group "
          f"({_DMRS_NUM_PAYLOAD_SYMB}/{NUM_SYMB_PER_SLOT} payload symbols/slot, "
          f"{len(_DMRS_SYMBOL_LOCAL_IDX)} DMRS){calib_note}, track_mode={weights_track_mode}, "
          f"which_augment={which_augment}, starting at "
          f"channel_drift_base_index={base_index}, cfo={base_cfo}{'' if cfo_drift == 0 else f' (drift={cfo_drift} scs/sec)'}, "
          f"SNR={conf.snr}dB, mcs={conf.mcs}",
          flush=True)

    results = []
    for g in range(num_groups):
        stats = run_group(escnn_trainer, codec, crc, rng, qm, num_bits_pilot, mod_data, n_users, n_ants, num_res,
                           ldpc_k, ldpc_n, noise_var, h, group_size_slots, calib_slots_per_group, layout,
                           g, base_index, base_cfo=base_cfo, cfo_drift=cfo_drift, save_diag=save_diag)
        slot_lo = base_index + g * group_size_slots
        slot_hi = slot_lo + group_size_slots - 1
        stats['channel_drift_base_index'] = slot_lo
        results.append(stats)
        # SINR summary prints every group regardless of save_diag - it's a cheap byproduct of
        # the (always-computed) noisy DMRS-based H, unlike h_abs_true_per_re below, which
        # needs its own extra transmit and stays gated to save_loss_plot_snr.
        sinr_db_re = 10 * np.log10(stats['sinr_per_re'])
        print(f"[drift] group {g}/{num_groups} slots={slot_lo}-{slot_hi} "
              f"ber_escnn={stats['ber_escnn']:.4e} ber_lmmse={stats['ber_lmmse']:.4e} "
              f"bler_escnn={stats['bler_escnn']:.4e} bler_lmmse={stats['bler_lmmse']:.4e} "
              f"mi_escnn={stats['mi_escnn']:.4f} mi_lmmse={stats['mi_lmmse']:.4f} "
              f"sinr_db(mean/min/max over REs+users)={sinr_db_re.mean():.1f}/"
              f"{sinr_db_re.min():.1f}/{sinr_db_re.max():.1f} "
              f"noise_var_est={stats['noise_var_est']:.4e} "
              f"lmmse_noise_var={stats['lmmse_noise_var']:.4e}", flush=True)

    if mod_data == 2:
        mod_text = 'BPSK'
    elif mod_data == 4:
        mod_text = 'QPSK'
    else:
        mod_text = str(mod_data) + 'Q'
    if conf.channel_model[0] == 'N':
        chan_text = 'Flat'
    elif conf.channel_model[0] in ('A', 'B', 'C'):
        # 'T' + model letter (e.g. 'TA', 'TC') - delay spread dropped, kept only to distinguish
        # TDL from the 'Flat'/env-name cases below, not to encode every channel parameter.
        chan_text = 'T' + conf.channel_model
    else:
        chan_text = conf.channel_model
    title_string = _build_ekf_filename_suffix(chan_text, mod_text, n_users, code_rate)
    title_string += '_s=' + str(conf.channel_seed) + '_SNR=' + str(conf.snr)
    title_string = datetime.now().strftime("%Y%m%d_%H%M_") + title_string
    output_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'Scratchpad'))
    os.makedirs(_long_path(output_dir), exist_ok=True)

    # Column A holds the per-group channel_drift_base_index (not an SNR sweep), named "cdi"
    # accordingly. That index is otherwise fully recoverable as (row number - 1) under the
    # default base_index=0/slots_per_group=1, so it isn't kept as a second, redundant column.
    idx = [r['channel_drift_base_index'] for r in results]
    # Column names/order deliberately match evaluate.py's data/data_bler/data_mi dicts
    # exactly, including its quirks: every file (BER, BLER, *and* MI) uses the "total_ber_"
    # prefix (evaluate.py's own BLER/MI CSVs do too, never "total_bler_"/"total_mi_"), and
    # ESCNN's columns carry a "_1" (an iteration number in evaluate.py, from
    # f"total_ber_{i+1}"/f"total_ber_user{u}_{i+1}") even though this script has no
    # per-iteration concept to report - it's meaningless here, kept only so a shared
    # column-name parser doesn't need to special-case which script produced the file. Order:
    # LMMSE (aggregate, then every user) first, ESCNN (aggregate, then every user) last -
    # never interleaved.
    data = {'cdi': idx,
            'total_ber_lmmse': [r['ber_lmmse'] for r in results],
            'total_ber_1': [r['ber_escnn'] for r in results]}
    data_bler = {'cdi': idx,
                 'total_ber_lmmse': [r['bler_lmmse'] for r in results],
                 'total_ber_1': [r['bler_escnn'] for r in results]}
    data_mi = {'cdi': idx,
               'total_ber_lmmse': [r['mi_lmmse'] for r in results],
               'total_ber_1': [r['mi_escnn'] for r in results]}
    for u in range(n_users):
        data[f'total_ber_lmmse_user{u}'] = [r['ber_lmmse_user'][u] for r in results]
        data_bler[f'total_ber_lmmse_user{u}'] = [r['bler_lmmse_user'][u] for r in results]
        data_mi[f'total_ber_lmmse_user{u}'] = [r['mi_lmmse_user'][u] for r in results]
    for u in range(n_users):
        data[f'total_ber_user{u}_1'] = [r['ber_escnn_user'][u] for r in results]
        data_bler[f'total_ber_user{u}_1'] = [r['bler_escnn_user'][u] for r in results]
        data_mi[f'total_ber_user{u}_1'] = [r['mi_escnn_user'][u] for r in results]

    file_path = os.path.abspath(os.path.join(output_dir, title_string) + ".csv")
    pd.DataFrame(data).to_csv(_long_path(file_path), index=False)
    print(f"[CSV] wrote {file_path}", flush=True)
    file_path_bler = os.path.abspath(os.path.join(output_dir, title_string) + "_bler.csv")
    pd.DataFrame(data_bler).to_csv(_long_path(file_path_bler), index=False)
    print(f"[CSV] wrote {file_path_bler}", flush=True)
    file_path_mi = os.path.abspath(os.path.join(output_dir, title_string) + "_mi.csv")
    pd.DataFrame(data_mi).to_csv(_long_path(file_path_mi), index=False)
    print(f"[CSV] wrote {file_path_mi}", flush=True)

    # Per-RE diagnostics (|H|, post-eq SINR): one (num_res, n_ants, n_users)/(num_res, n_users)
    # array per group. Kept out of the CSVs (those are scalar-per-group by design, shared with
    # evaluate.py's column layout) and out of the console log (too large to print in full per
    # group - only a mean/min/max summary goes there) - HDF5 instead, mirroring evaluate.py's
    # save_llrs convention (float16 + gzip), grouped by cdi so it lines up with the CSV rows.
    # Only written when save_diag (conf.snr in conf.save_loss_plot_snr) - same whitelist
    # run_group() used to skip the extra noise-free transmit in the first place.
    if save_diag:
        file_path_diag = os.path.abspath(os.path.join(output_dir, title_string) + "_diag.h5")
        with h5py.File(_long_path(file_path_diag), "w") as diag_h5:
            # File-level attrs so the axes are self-describing - the arrays themselves carry no
            # labels, and re/ant/user are otherwise just positional indices with no other record
            # of which physical RE/antenna/user each one is.
            diag_h5.attrs["h_abs_per_re_dims"] = "RE, ant, user"
            diag_h5.attrs["h_abs_true_per_re_dims"] = "RE, ant, user"
            diag_h5.attrs["h_angle_per_re_dims"] = "RE, ant, user"
            diag_h5.attrs["h_angle_true_per_re_dims"] = "RE, ant, user"
            diag_h5.attrs["h_abs_true_untrunc_per_re_dims"] = "RE, ant, user"
            diag_h5.attrs["h_angle_true_untrunc_per_re_dims"] = "RE, ant, user"
            diag_h5.attrs["h_abs_ground_truth_per_re_dims"] = "RE, ant, user"
            diag_h5.attrs["h_angle_ground_truth_per_re_dims"] = "RE, ant, user"
            diag_h5.attrs["sinr_per_re_dims"] = "RE, user"
            diag_h5.attrs["num_res"] = num_res
            diag_h5.attrs["n_ants"] = n_ants
            diag_h5.attrs["n_users"] = n_users
            diag_h5.attrs["h_abs_per_re_note"] = "DMRS-based LS+IFFT-denoise estimate - what LMMSE/EKF actually see"
            diag_h5.attrs["h_abs_true_per_re_note"] = "same estimator, noise_var=0 - noise-free reference channel"
            diag_h5.attrs["h_angle_per_re_note"] = "angle(H), noisy DMRS-based estimate, radians, not unwrapped"
            diag_h5.attrs["h_angle_true_per_re_note"] = "angle(H), noise-free reference, radians, not unwrapped"
            diag_h5.attrs["h_abs_true_untrunc_per_re_note"] = ("same noise_var=0 pass as h_abs_true_per_re, but "
                                                                 "with the L_taps delay-domain truncation/taper "
                                                                 "skipped entirely (plain DFT interpolation of the "
                                                                 "raw pilot estimates) - compare against "
                                                                 "h_abs_true_per_re to isolate what truncation "
                                                                 "itself is doing, with noise out of the picture")
            diag_h5.attrs["h_angle_true_untrunc_per_re_note"] = "angle(H) for h_abs_true_untrunc_per_re, radians, not unwrapped"
            diag_h5.attrs["h_abs_ground_truth_per_re_note"] = ("independent ground truth (_ground_truth_channel) - "
                                                                 "a dense, full-resolution, noise_var=0 per-RE LS "
                                                                 "estimate that bypasses the DMRS comb/delay-domain "
                                                                 "machinery entirely; compare h_abs_true_per_re and "
                                                                 "h_abs_true_untrunc_per_re against this, not just "
                                                                 "against each other, to tell whether either DMRS "
                                                                 "reconstruction matches the real channel")
            diag_h5.attrs["h_angle_ground_truth_per_re_note"] = "angle(H) for h_abs_ground_truth_per_re, radians, not unwrapped"
            diag_h5.attrs["noise_var"] = noise_var
            diag_h5.attrs["noise_var_note"] = "constant for the whole run - 10**(-0.1*conf.snr)*CONSTELLATION_FACTOR[mod_data]"
            diag_h5.attrs["override_noise_var"] = bool(getattr(conf, 'override_noise_var', True))
            diag_h5.attrs["lmmse_noise_var_note"] = ("per-group attr (cdi_*.attrs) - noise_var actually fed "
                                                       "into this group's LMMSE math: equals the file-level "
                                                       "noise_var when override_noise_var=True, else that "
                                                       "group's own noise_var_est")
            diag_h5.attrs["noise_var_est_note"] = ("per-group attr (cdi_*.attrs) - receiver-side pilot-residual "
                                                     "noise_var estimate from that group's own DMRS occasions "
                                                     "(_estimate_channel_from_dmrs's estimate_noise_var path), "
                                                     "computed regardless of override_noise_var so it's visible "
                                                     "even when not the one actually used")
            diag_h5.attrs["tx_bits_dims"] = "bit (symbol-major, bit-in-symbol), user, RE"
            diag_h5.attrs["llrs_mat_lmmse_dims"] = "symbol, bit, RE"
            diag_h5.attrs["detected_word_lmmse_dims"] = "bit (symbol-major, bit-in-symbol), user, RE"
            diag_h5.attrs["equalized_lmmse_dims"] = "symbol, user, RE"
            diag_h5.attrs["H_data_dims"] = "RE, ant, user"
            diag_h5.attrs["llr_chain_note"] = ("full LMMSE LLR-generation chain for offline debugging: "
                                                "H_data (channel estimate) + noise_var -> equalized_lmmse "
                                                "(lmmse_equalize_with_H's output) + sinr_per_re (postEqSINR) "
                                                "-> llrs_mat_lmmse (LmmseDemod's output, what [llr-diag] scores "
                                                "against tx_bits). detected_word_lmmse is llrs_mat_lmmse's hard "
                                                "decision, already thresholded.")
            for r in results:
                grp = diag_h5.create_group(f"cdi_{r['channel_drift_base_index']}")
                grp.create_dataset("h_abs_per_re", data=r['h_abs_per_re'].astype(np.float16),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("h_abs_true_per_re", data=r['h_abs_true_per_re'].astype(np.float16),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("h_angle_per_re", data=r['h_angle_per_re'].astype(np.float16),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("h_angle_true_per_re", data=r['h_angle_true_per_re'].astype(np.float16),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("h_abs_true_untrunc_per_re", data=r['h_abs_true_untrunc_per_re'].astype(np.float16),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("h_angle_true_untrunc_per_re", data=r['h_angle_true_untrunc_per_re'].astype(np.float16),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("h_abs_ground_truth_per_re", data=r['h_abs_ground_truth_per_re'].astype(np.float16),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("h_angle_ground_truth_per_re", data=r['h_angle_ground_truth_per_re'].astype(np.float16),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("sinr_per_re", data=r['sinr_per_re'].astype(np.float16),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("tx_bits", data=r['tx_bits'].astype(np.uint8),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("llrs_mat_lmmse", data=r['llrs_mat_lmmse'].squeeze(-1).astype(np.float16),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("detected_word_lmmse", data=r['detected_word_lmmse'].astype(np.uint8),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("equalized_lmmse", data=r['equalized_lmmse'].astype(np.complex64),
                                    compression="gzip", compression_opts=4)
                grp.create_dataset("H_data", data=r['H_data'].astype(np.complex64),
                                    compression="gzip", compression_opts=4)
                grp.attrs["noise_var_est"] = r['noise_var_est']
                grp.attrs["lmmse_noise_var"] = r['lmmse_noise_var']
        print(f"[H5] wrote {file_path_diag}", flush=True)


if __name__ == '__main__':
    main()
