#!/usr/bin/env python
"""
ekf.py -- Streaming per-group channel-drift + unsupervised EKF tracking.

A deliberately different loop from evaluate.py's run_evaluate(): no train/val/test
split, no supervised training, and the channel genuinely changes over the course of
the run instead of being fixed per block. Every group of data slots is treated as
fully known (a pilot) for BER scoring, but LMMSE's channel estimate (and the soft
probs fed to ESCNN via AUGMENT_LMMSE) comes from a *separate*, dedicated calibration
slot at the same channel realization - not from the slots being scored - so there's
no leakage between "what LMMSE/ESCNN get to see" and "what gets judged."

Unit of work is a *group*: conf.slots_per_group data slots plus one
calibration slot, all sharing one channel realization (channel_drift_base_index only
advances between groups, not within one). The data slots get exactly one EKF
predict() + one sequential update() per slot (see ESCNNTrainer.ekf_predict_update,
unchanged - handing it a group's worth of data makes it resolve num_slots == group
size on its own); the calibration slot never enters the EKF. Setting
slots_per_group=1 makes every data slot its own group, i.e. a new
channel every slot (plus its own calibration slot).

Reuses, unmodified: ESCNNTrainer (construction/weight loading/freezing/EKF/_forward),
EkfParamTracker, SyndromeLoss, SEDChannel (channel generation, including the TDL
channel_drift_base_index machinery), ChannelEstimate/LmmseDemod, encode_pilots, the
LDPC/CRC codecs. Nothing here duplicates any of that - this file is only the
orchestration loop for a shape those pieces don't otherwise support (streaming,
calibration-slot LMMSE, no training).

Usage:
    python -m python_code.ekf --config path/to/config.yaml

Relevant config keys (see config.yaml for the full list/defaults):
    load_escnn_weights_tag       - required: the pretrained checkpoint to track from
    escnn_load_freeze            - which params the EKF is allowed to move
    escnn_ekf_*                  - EKF dynamics/noise/chunking (same knobs as the
                                    block-based evaluate.py path)
    slots_per_group               - slots per group (= slots per channel realization, and per CFO value)
    channel_drift_base_index     - starting slot offset into the TDL trajectory
    cfo                           - base/starting CFO (scs); constant within a group
    cfo_drift                     - CFO drift rate (scs/sec); advances cfo by cfo_drift *
                                    elapsed-seconds at each group boundary, can be negative
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
from python_code.detectors.lmmse.lmmse_equalizer import ChannelEstimate, LmmseDemod
from python_code.evaluate import calc_mi_from_ldpc, crc_fail_mask, resolve_auto_escnn_weights_tag
from python_code.utils.constants import (CP, FFT_size, FIRST_CP, GENIE_CFO, NUM_SAMPLES_PER_SLOT,
                                          NUM_SYMB_PER_SLOT, SLOT_LENGTH_SEC)

# M-QAM average-energy normalization constants (2*(M-1)/3), same table evaluate.py
# uses to turn an SNR into a noise variance.
CONSTELLATION_FACTOR = {2: 1, 4: 2, 16: 10, 64: 42, 256: 170}


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


def _build_ekf_filename_suffix(chan_text: str, mod_text: str, n_users: int, code_rate) -> str:
    """Simplified analogue of evaluate.py's _build_escnn_filename_suffix: same spirit (readable
    tag=value pairs), but keeping only what this script actually uses. Deliberately drops
    everything training-specific (epochs, train_samples, pilot_data_ratio, batch_size,
    block_length_factor, learning_rate, dropout, weight_decay, training_loss/beta_balance/tw,
    save-weights tag) since no training happens here - the EKF replaces it entirely."""
    freeze_codes = {'none': 'n', 'scale': 'sc', 'first_conv': 'fc1', 'second_conv': 'fc2', 'last_conv': 'fc3',
                    'scale_only': 'so', 'last_conv_only': 'lco', 'first_conv_only': 'fco',
                    'first_conv_and_scale_only': 'fc1sco', 'all': 'a'}
    corr_map = {'none': 'No', 'low': 'Lo', 'medium': 'Med', 'medium_a': 'MedA', 'high': 'Hi', 'custom': 'Cust'}
    title_string = (f"{chan_text}_sp={conf.speed}_{mod_text}_REs={conf.num_res}_UEs={n_users}"
                     f"_ant={conf.n_ants}_cfo={conf.cfo}_kr={conf.kernel_size}"
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
    title_string += '_ps=' + str(getattr(conf, 'pilot_size', -1))
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
    but against an H estimated elsewhere (a dedicated calibration slot here) instead of
    re-estimating it from rx_c itself - LmmseEqualize always re-estimates H from whatever it's
    given, which would leak the answer if rx_c were the same symbols being scored (see the
    calibration-slot discussion this replaces)."""
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


def transmit_and_prep(bits: np.ndarray, mod_data: int, n_users: int, num_res: int, h: np.ndarray,
                       noise_var: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modulate+transmit bits through SEDChannel (at whatever conf.channel_drift_base_index is
    currently set to) and return (rx, rx_ce, s_orig) ready for ChannelEstimate /
    lmmse_equalize_with_H: complex128, symbol-major, and (for conf.separate_pilots) with the
    leading user axis on rx_ce."""
    s = modulate_bits(bits, mod_data, n_users, num_res)
    s_orig = np.copy(s)
    rx, rx_ce = SEDChannel.transmit(s=s, h=h, noise_var=noise_var, num_res=num_res,
                                     cfo_and_iqmm_in_rx=conf.cfo_and_iqmm_in_rx,
                                     n_users=n_users, pilot_length=bits.shape[0])
    # SEDChannel.transmit's TDL path returns rx as complex64 (via apply_td_and_impairments's
    # internal np.complex64 buffer) but rx_ce as complex128 (assigned into a dtype=complex
    # container, which upcasts it) - channel_dataset.py normally papers over this by
    # accumulating both into complex128 arrays before use; do the same here explicitly.
    rx = rx.astype(np.complex128)
    rx_ce = rx_ce.astype(np.complex128)
    rx = np.transpose(rx, (1, 0, 2))                        # (symbols, n_ants, num_res)
    s_orig = np.transpose(s_orig, (1, 0, 2))                 # (symbols, n_users, num_res)
    if conf.separate_pilots:
        # SEDChannel.transmit returns one combined (n_ants, symbols, num_res) array for
        # separate_pilots=True (each user's own samples already live at their own
        # round-robin positions within it); ChannelEstimate/lmmse_equalize_with_H still
        # index a leading user axis, so replicate it - matches what channel_dataset.py's
        # broadcasting assignment into its 4D rx_ce_full array does implicitly.
        rx_ce = np.transpose(rx_ce, (1, 0, 2))               # (symbols, n_ants, num_res)
        rx_ce = np.broadcast_to(rx_ce[None, :, :, :], (n_users,) + rx_ce.shape).copy()
    else:
        rx_ce_t = np.zeros((n_users, rx.shape[0], rx.shape[1], rx.shape[2]), dtype=complex)
        for user in range(n_users):
            rx_ce_t[user] = np.transpose(rx_ce[user], (1, 0, 2))
        rx_ce = rx_ce_t
    return rx, rx_ce, s_orig


def run_group(escnn_trainer: ESCNNTrainer, codec: LDPC5GCodec, crc: CRC5GCodec, rng: np.random.Generator,
              qm: int, mod_data: int, n_users: int, n_ants: int, num_res: int, ldpc_k: int, ldpc_n: int,
              noise_var: float, h: np.ndarray, group_size_slots: int, group_idx: int, base_index: int,
              base_cfo: float = 0.0, cfo_drift: float = 0.0, save_diag: bool = False) -> dict:
    """Generate, transmit, LMMSE-estimate/equalize, EKF-update, and score one group of
    group_size_slots consecutive slots sharing a single channel realization - plus one extra
    calibration slot at the same channel_drift_base_index, used only to estimate H for LMMSE (and,
    via AUGMENT_LMMSE, to feed ESCNN's auxiliary input). The calibration slot is never scored
    and never enters the EKF, so LMMSE's estimate can't leak the answer for the slots being
    judged (see the augmentation-leakage discussion this replaces).

    save_diag gates the noise-free reference-channel diagnostics (an extra transmit + per-RE
    LS estimate, only worth paying for at the handful of SNRs in conf.save_loss_plot_snr - see
    main()) - off by default so a plain run_group() call stays as cheap as before this existed.

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

    # Calibration slot: plain random bits (never decoded/scored, so no need for LDPC coding).
    calib_bits = rng.integers(0, 2, size=(NUM_SYMB_PER_SLOT * qm, n_users, num_res))
    _, rx_ce_calib, s_orig_calib = transmit_and_prep(calib_bits, mod_data, n_users, num_res, h, noise_var)
    calib_cfo_comp = _genie_cfo_comp_vector(1)
    if calib_cfo_comp is not None:
        rx_ce_calib = rx_ce_calib * calib_cfo_comp[None, :, None, None]
    rx_ce_calib_t = torch.from_numpy(rx_ce_calib)
    s_orig_calib_t = torch.from_numpy(s_orig_calib)

    # Noise-free reference channel: same calib_bits (so same symbols) and same channel
    # realization (conf.channel_seed/channel_drift_base_index unchanged since the call above),
    # just noise_var=0 - mirrors the existing rx_clean idiom in mimo_channel_dataset.py
    # (run_tdcnn). Feeding this through the same ChannelEstimate() LS estimator as the noisy
    # path gives a genuinely noise-free |H| per RE for free, without re-deriving it from
    # Sionna's internal CIR tensor (which would mean re-implementing ApplyTimeChannel's own
    # delay-alignment/CP-removal logic by hand just to get the same answer). Only worth the
    # extra transmit when save_diag is set.
    if save_diag:
        _, rx_ce_calib_true, s_orig_calib_true = transmit_and_prep(calib_bits, mod_data, n_users, num_res, h, 0.0)
        rx_ce_calib_true_t = torch.from_numpy(rx_ce_calib_true)
        s_orig_calib_true_t = torch.from_numpy(s_orig_calib_true)

    pilot_length = group_size_slots * NUM_SYMB_PER_SLOT * qm
    tx_bits = encode_pilots(rng, pilot_length, num_res, n_users, codec, crc, ldpc_k, ldpc_n)
    rx, _, _ = transmit_and_prep(tx_bits, mod_data, n_users, num_res, h, noise_var)
    data_cfo_comp = _genie_cfo_comp_vector(group_size_slots)
    if data_cfo_comp is not None:
        rx = rx * data_cfo_comp[:, None, None]
    # lmmse_equalize_with_H creates its output on CPU unconditionally, so inputs need to be
    # CPU too (mirrors evaluate.py's own rx_c = rx.cpu()).
    rx_c = torch.from_numpy(rx)

    num_symbols = rx.shape[0]
    detected_word_lmmse = np.zeros((num_symbols * qm, n_users, num_res))
    llrs_mat_lmmse = np.zeros((num_symbols, qm * n_users, num_res, 1))
    # Per-RE diagnostics for the EKF-divergence investigation: |H| and angle(H) (calibration-slot
    # channel estimate, noisy and noise-free) and post-equalization SINR, all independent of the
    # ESCNN/syndrome measurement the EKF actually tracks - a candidate "trust signal" for
    # gating that update (see ekf_tracker.py's update()) without the circularity of using the
    # syndrome innovation itself. The _per_re (no "true") arrays are what LMMSE/EKF actually see
    # (noise and all); the _true_per_re arrays are the noise_var=0 reference, so channel dips (or
    # a shift in the channel's frequency-domain phase structure vs. training) can be told apart
    # from estimation noise when comparing across SNRs at the same cdi.
    h_abs_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    h_abs_true_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    h_angle_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    h_angle_true_per_re = np.zeros((num_res, n_ants, n_users), dtype=np.float32)
    sinr_per_re = np.zeros((num_res, n_users), dtype=np.float32)
    for re in range(num_res):
        H = ChannelEstimate(rx_ce_calib_t, s_orig_calib_t, NUM_SYMB_PER_SLOT, re)
        equalized, postEqSINR = lmmse_equalize_with_H(H, rx_c, noise_var, re)
        LmmseDemod(equalized, postEqSINR, qm, re, llrs_mat_lmmse, detected_word_lmmse, 1)
        h_abs_per_re[re] = H.abs().cpu().numpy()
        h_angle_per_re[re] = H.angle().cpu().numpy()
        sinr_per_re[re] = postEqSINR.cpu().numpy()
        if save_diag:
            H_true = ChannelEstimate(rx_ce_calib_true_t, s_orig_calib_true_t, NUM_SYMB_PER_SLOT, re)
            h_abs_true_per_re[re] = H_true.abs().cpu().numpy()
            h_angle_true_per_re[re] = H_true.angle().cpu().numpy()

    # Diagnostic only: zero out LMMSE's LLRs at the given RE indices (e.g. RE 0, suspected of an
    # anomalous |H| - see plot_channel_diag.py) before they're used for anything downstream -
    # LDPC decoding (lmmse_stream, below) and the AUGMENT_LMMSE prior fed to ESCNN
    # (probs_for_aug, also below, since it's sigmoid(llrs_mat_lmmse)). Zeroing (not removing the
    # RE) keeps ldpc_n/the code rate unchanged; a zeroed LLR just tells the decoder "no
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
    rx_real[:, 0::2, :] = rx.real.astype(np.float32)
    rx_real[:, 1::2, :] = rx.imag.astype(np.float32)
    rx_real_t = torch.from_numpy(rx_real)

    # AUGMENT_LMMSE: safe here specifically because llrs_mat_lmmse came from the calibration
    # slot's own channel estimate, never from the data slots being scored below - no leakage.
    probs_for_aug = torch.sigmoid(torch.tensor(llrs_mat_lmmse, dtype=torch.float32))

    escnn_trainer.ekf_predict_update(rx_real_t, qm, n_users, conf.iterations, probs_for_aug)
    _, llrs_mat_list = escnn_trainer._forward(rx_real_t, qm, n_users, conf.iterations, probs_for_aug)
    escnn_llrs = llrs_mat_list[-1].squeeze(-1).cpu().numpy()   # (symbols, qm*n_users, num_res)

    # BER (hard decisions) + build the LDPC-decoder-input LLR streams BLER/MI need below.
    # Stream layout matches SyndromeLoss/_syndrome_component's convention (symbol-major, then
    # bit-in-symbol, then RE), which is also encode_pilots' own tx_bits layout - so a plain
    # reshape(-1) lines both up with no reordering needed.
    ber_escnn_num = ber_escnn_den = ber_lmmse_num = ber_lmmse_den = 0
    ber_escnn_user, ber_lmmse_user = [], []
    escnn_stream = np.zeros((n_users, group_size_slots * ldpc_n))
    lmmse_stream = np.zeros((n_users, group_size_slots * ldpc_n))
    tx_stream = np.zeros((n_users, group_size_slots * ldpc_n))
    for user in range(n_users):
        tx_user = tx_bits[:, user, :].reshape(num_symbols, qm, num_res)
        escnn_user = (escnn_llrs[:, user * qm:(user + 1) * qm, :] > 0).astype(int)
        n_err_escnn = int((escnn_user != tx_user).sum())
        ber_escnn_num += n_err_escnn
        ber_escnn_den += tx_user.size
        ber_escnn_user.append(n_err_escnn / tx_user.size)

        lmmse_user = detected_word_lmmse[:, user, :].reshape(num_symbols, qm, num_res)
        n_err_lmmse = int((lmmse_user != tx_user).sum())
        ber_lmmse_num += n_err_lmmse
        ber_lmmse_den += tx_user.size
        ber_lmmse_user.append(n_err_lmmse / tx_user.size)

        escnn_stream[user] = escnn_llrs[:, user * qm:(user + 1) * qm, :].reshape(-1)
        lmmse_stream[user] = llrs_mat_lmmse[:, user * qm:(user + 1) * qm, :, 0].reshape(-1)
        tx_stream[user] = tx_user.reshape(-1)

    # BLER: per-slot LDPC decode + CRC check, same mechanism evaluate.py uses
    # (codec.decode -> crc.decode -> crc_fail_mask), over the data slots only - the
    # calibration slot is never LDPC-coded and never enters this.
    bler_escnn_fail = np.zeros(n_users, dtype=int)
    bler_lmmse_fail = np.zeros(n_users, dtype=int)
    for slot in range(group_size_slots):
        win = slice(slot * ldpc_n, (slot + 1) * ldpc_n)
        decoded_escnn = codec.decode(escnn_stream[:, win])
        bler_escnn_fail += crc_fail_mask(decoded_escnn, crc.decode(decoded_escnn)).astype(int)
        decoded_lmmse = codec.decode(lmmse_stream[:, win])
        bler_lmmse_fail += crc_fail_mask(decoded_lmmse, crc.decode(decoded_lmmse)).astype(int)
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
        'sinr_per_re': sinr_per_re,
    }


def main():
    parser = argparse.ArgumentParser(description='Streaming channel-drift + unsupervised EKF tracking')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    args = parser.parse_args()

    conf.reload_config(args.config)
    resolve_auto_escnn_weights_tag()
    # Always AUGMENT_LMMSE here, regardless of what's in the config: run_group() builds its
    # own leakage-free probs_for_aug from a dedicated calibration slot every call, so this is
    # never actually reading LMMSE priors computed elsewhere.
    conf.set_value('which_augment', 'AUGMENT_LMMSE')

    n_users, n_ants, num_res = conf.n_users, conf.n_ants, conf.num_res
    qm, code_rate = get_mcs(conf.mcs)
    qm = int(qm)
    mod_data = int(2 ** qm)
    ldpc_n = int(num_res * NUM_SYMB_PER_SLOT * qm)
    ldpc_k = int(ldpc_n * code_rate)
    crc_length = 24 if ldpc_k > 3824 else 16
    codec = LDPC5GCodec(k=ldpc_k + crc_length, n=ldpc_n)
    crc = CRC5GCodec(crc_length)
    rng = np.random.default_rng(seed=conf.seed)

    noise_var = 10 ** (-0.1 * conf.snr) * CONSTELLATION_FACTOR[mod_data]
    h = SEDChannel.calculate_channel(n_ants, n_users, num_res)
    # Same SNR whitelist evaluate.py uses to gate its (also expensive) per-SNR loss/LLR plots
    # (evaluate.py:2341) - the per-RE channel/SINR diagnostics are only worth the extra
    # transmit+estimate and the H5 file at those SNRs, not every SNR in a sweep.
    save_diag = conf.snr in getattr(conf, 'save_loss_plot_snr', [])

    escnn_trainer = ESCNNTrainer(qm, n_users, n_ants)
    escnn_trainer._initialize_detector(qm, n_users, n_ants)
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
    # complete group is simply discarded.
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

    print(f"[drift] {num_groups} groups x {group_size_slots} slot(s)/group, starting at "
          f"channel_drift_base_index={base_index}, cfo={base_cfo}{'' if cfo_drift == 0 else f' (drift={cfo_drift} scs/sec)'}, "
          f"SNR={conf.snr}dB, mcs={conf.mcs}", flush=True)

    results = []
    for g in range(num_groups):
        stats = run_group(escnn_trainer, codec, crc, rng, qm, mod_data, n_users, n_ants, num_res,
                           ldpc_k, ldpc_n, noise_var, h, group_size_slots, g, base_index,
                           base_cfo=base_cfo, cfo_drift=cfo_drift, save_diag=save_diag)
        slot_lo = base_index + g * group_size_slots
        slot_hi = slot_lo + group_size_slots - 1
        stats['channel_drift_base_index'] = slot_lo
        results.append(stats)
        # SINR summary prints every group regardless of save_diag - it's a cheap byproduct of
        # the (always-computed) noisy calibration-slot H, unlike h_abs_true_per_re below, which
        # needs its own extra transmit and stays gated to save_loss_plot_snr.
        sinr_db_re = 10 * np.log10(stats['sinr_per_re'])
        print(f"[drift] group {g}/{num_groups} slots={slot_lo}-{slot_hi} "
              f"ber_escnn={stats['ber_escnn']:.4e} ber_lmmse={stats['ber_lmmse']:.4e} "
              f"bler_escnn={stats['bler_escnn']:.4e} bler_lmmse={stats['bler_lmmse']:.4e} "
              f"mi_escnn={stats['mi_escnn']:.4f} mi_lmmse={stats['mi_lmmse']:.4f} "
              f"sinr_db(mean/min/max over REs+users)={sinr_db_re.mean():.1f}/"
              f"{sinr_db_re.min():.1f}/{sinr_db_re.max():.1f}", flush=True)

    if mod_data == 2:
        mod_text = 'BPSK'
    elif mod_data == 4:
        mod_text = 'QPSK'
    else:
        mod_text = str(mod_data) + 'Q'
    if conf.channel_model[0] == 'N':
        chan_text = 'Flat'
    elif conf.channel_model[0] in ('A', 'B', 'C'):
        chan_text = 'TDL-' + conf.channel_model + '-' + str(int(round(float(conf.delay_spread) * 1e9)))
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
            diag_h5.attrs["sinr_per_re_dims"] = "RE, user"
            diag_h5.attrs["num_res"] = num_res
            diag_h5.attrs["n_ants"] = n_ants
            diag_h5.attrs["n_users"] = n_users
            diag_h5.attrs["h_abs_per_re_note"] = "LS estimate from the noisy calibration slot - what LMMSE/EKF actually see"
            diag_h5.attrs["h_abs_true_per_re_note"] = "same LS estimator, noise_var=0 - noise-free reference channel"
            diag_h5.attrs["h_angle_per_re_note"] = "angle(H), noisy calibration-slot estimate, radians, not unwrapped"
            diag_h5.attrs["h_angle_true_per_re_note"] = "angle(H), noise-free reference, radians, not unwrapped"
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
                grp.create_dataset("sinr_per_re", data=r['sinr_per_re'].astype(np.float16),
                                    compression="gzip", compression_opts=4)
        print(f"[H5] wrote {file_path_diag}", flush=True)


if __name__ == '__main__':
    main()
