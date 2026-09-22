#!/usr/bin/env python
"""
dmrs_pilots.py -- In-slot DMRS pilot generation and delay-domain channel estimation.

Extracted from ekf.py (which introduced this in-band, 5G-style DMRS scheme: PUSCH
mapping-type-A style, 2 OFDM symbols/slot, comb-2 Type-1, up to 4 CDM-multiplexed
ports via FD-OCC - see dmrs_layout) so evaluate.py's training pipeline can reuse the
exact same pilot layout and channel-estimation math (conf.chanestmode) instead of
reimplementing it - a second copy would inevitably drift from ekf.py's own DMRS
handling, silently reintroducing the train/inference channel-estimate mismatch this
module exists to close. ekf.py imports everything here unchanged; nothing about its
own behavior changes by virtue of this extraction.

Kept out of ekf.py itself (rather than the reverse - evaluate.py importing from
ekf.py) because ekf.py already imports several names from evaluate.py at module
scope; a shared module avoids the resulting circular import.
"""
from typing import Tuple

import numpy as np
import torch

from python_code import conf
from python_code.utils.constants import (CP, DMRS_NUM_PAYLOAD_SYMB, DMRS_SYMBOL_LOCAL_IDX, FFT_size,
                                          FIRST_CP, GENIE_CFO, NUM_SAMPLES_PER_SLOT, NUM_SYMB_PER_SLOT,
                                          SAMPLING_RATE)

# M-QAM average-energy normalization constants (2*(M-1)/3), same table evaluate.py
# uses to turn an SNR into a noise variance.
CONSTELLATION_FACTOR = {2: 1, 4: 2, 16: 10, 64: 42, 256: 170}

# These are concrete numbers from the DMRS design, not tunables, so they're kept as
# constants rather than new config.yaml keys.
DMRS_POWER_BOOST_DB = 3.0     # extra pilot EPRE (dB) at n_users==1 only, where the
                               # 'odd' comb sits completely unused for DMRS (see
                               # dmrs_known_tx/dmrs_layout) - n_users in (2, 4)
                               # already occupy every comb RE, so no spare budget to
                               # redistribute and no boost
DMRS_DELAY_TRUNC_MARGIN = 12  # L_taps = ceil(margin * delay_spread / delay_bin_width)


def dmrs_layout(n_users: int, num_res: int) -> dict:
    """Per-user DMRS RE/OCC assignment (comb-2 Type-1 FD-OCC, up to 4 CDM-multiplexed
    ports). Returns {user: {'comb': 'even'|'odd', 'pairs': [(re_a, re_b_or_None), ...],
    'sign': (s_a, s_b_or_None), 'occ_active': bool}}. Pairing (for FD-OCC and for the shared
    reference sequence) groups consecutive same-comb REs two at a time; a trailing unpaired RE
    (only when a comb has an odd count) becomes a trivial single-port pair. 'occ_active' is True
    only when two users genuinely share a comb (n_users==4) - it tells
    estimate_channel_from_dmrs whether Step 1 must deocc-combine each RE-pair into one estimate
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


def dmrs_reference_values(rng: np.random.Generator, num_res: int) -> dict:
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


def dmrs_known_tx(layout: dict, ref_values: dict, n_users: int) -> dict:
    """Per user: {'occ_active': bool, 'entries': [(re_a, re_b_or_None, val_a, val_b_or_None), ...]}
    - the exact known complex symbol each of that user's DMRS REs carries (comb reference value
    x OCC sign x amplitude). DMRS is always QPSK, independent of whatever modulation the payload
    (mod_data) uses, so amplitude is fixed at QPSK's own nominal per-RE energy
    (CONSTELLATION_FACTOR[4] - the same Es convention noise_var is derived from, just always at
    QPSK's value rather than the payload's) - plus the +3dB (linear sqrt(2) amplitude) boost only
    at n_users==1, where the 'odd' comb sits completely unused for DMRS (see dmrs_layout) so the
    freed budget goes entirely into the one active port; at n_users==2 both combs are already
    occupied (one user each, no spare to redistribute), same as n_users==4's OCC-shared case, so
    neither gets a boost. Shared by build_dmrs_tx_symbols (what gets transmitted) and
    estimate_channel_from_dmrs (what the LS/deocc division uses)."""
    amp = np.sqrt(CONSTELLATION_FACTOR[4])  # DMRS's own (QPSK) reference energy, not mod_data's
    if n_users == 1:
        amp *= 10 ** (DMRS_POWER_BOOST_DB / 20.0)  # dB -> linear amplitude factor
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


def build_dmrs_tx_symbols(known_tx: dict, n_users: int, num_res: int, num_occasions: int) -> np.ndarray:
    """(n_users, num_occasions, num_res) complex DMRS tx symbols from dmrs_known_tx's per-user
    known values - constant across every occasion (only the channel/noise differ between
    occasions), so Step 1's time-averaging in estimate_channel_from_dmrs is coherent. No data is
    multiplexed onto any DMRS RE, including off-comb ones - everything else in the array stays
    zero."""
    s = np.zeros((n_users, num_occasions, num_res), dtype=complex)
    for user, info in known_tx.items():
        for re_a, re_b, val_a, val_b in info['entries']:
            s[user, :, re_a] = val_a
            if re_b is not None:
                s[user, :, re_b] = val_b
    return s


def interleave_group_symbols(payload_s: np.ndarray, dmrs_s: np.ndarray, num_slots: int) -> np.ndarray:
    """Interleave a region's payload symbols (n_users, num_slots*DMRS_NUM_PAYLOAD_SYMB, num_res)
    and DMRS symbols (n_users, num_slots*len(DMRS_SYMBOL_LOCAL_IDX), num_res) into
    (n_users, num_slots*NUM_SYMB_PER_SLOT, num_res): DMRS at DMRS_SYMBOL_LOCAL_IDX within every
    slot, payload at the other slot-local positions, in order - so the combined array is a
    genuine contiguous per-slot symbol sequence, letting ordinary slot-based CP/CFO machinery
    treat it exactly like any other slot-based transmission with no changes needed there."""
    n_users, _, num_res = payload_s.shape
    s = np.zeros((n_users, num_slots * NUM_SYMB_PER_SLOT, num_res), dtype=complex)
    dmrs_set = set(DMRS_SYMBOL_LOCAL_IDX)
    payload_local_idx = [i for i in range(NUM_SYMB_PER_SLOT) if i not in dmrs_set]
    for slot in range(num_slots):
        base = slot * NUM_SYMB_PER_SLOT
        for j, local_idx in enumerate(payload_local_idx):
            s[:, base + local_idx, :] = payload_s[:, slot * DMRS_NUM_PAYLOAD_SYMB + j, :]
        for j, local_idx in enumerate(DMRS_SYMBOL_LOCAL_IDX):
            s[:, base + local_idx, :] = dmrs_s[:, slot * len(DMRS_SYMBOL_LOCAL_IDX) + j, :]
    return s


def genie_cfo_comp_vector(num_slots: int):
    """This region's genie CFO phase-compensation vector (one factor per OFDM symbol, length
    num_slots*NUM_SYMB_PER_SLOT), mirroring evaluate.py's own inline GENIE_CFO path: cancels the
    common per-symbol phase rotation using the true conf.cfo, leaving the intra-symbol phase ramp
    - i.e. ICI - uncorrected by design, same as evaluate.py. Must be applied before any DMRS rows
    are stripped out (interleave_group_symbols's row indices assume the full, contiguous
    NUM_SYMB_PER_SLOT-per-slot grid this vector is built against - see mimo_channel_dataset.py's
    conf.chanestmode='dmrs' path). Returns None (no-op) unless GENIE_CFO and conf.cfo != 0."""
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


def edge_taper(length: int) -> np.ndarray:
    """Length-`length` array of 1s except the last quarter (min 1 tap), which raised-cosine
    tapers down to 0 - used to fight Gibbs ringing at the *discarded* edge of a kept delay-domain
    segment, without attenuating the (presumably larger) energy nearer delay 0."""
    w = np.ones(length)
    edge = max(1, length // 4)
    w[length - edge:] = 0.5 * (1 + np.cos(np.pi * np.arange(edge) / edge))
    return w


def fold_delay_taps(h_alias: np.ndarray, num_res: int, causal_len: int, anticausal_len: int,
                     taper: bool) -> np.ndarray:
    """Place an M-point aliased delay-domain sequence into a full num_res-length array, split
    between its causal (near-zero, non-negative-delay) front and its *wrapped* tail - IFFT/DFT
    periodicity puts negative delays at the far end of h_alias (index M-1 = delay -1, M-2 = delay
    -2, ...; this is where a pulse-shaping filter's pre-cursor taps show up). Zero-padding by
    naively appending zeros after index causal_len+anticausal_len-1 would jam a hard discontinuity
    right against any real wrapped content, ringing across the whole spectrum on FFT. Correct
    placement: causal part goes at the front, anticausal part goes at the *end* of the full-length
    array, zeros in between.

    taper=True raised-cosine-tapers the *discarded* edge of each kept segment - the end farthest
    from delay 0 (only meaningful when causal_len/anticausal_len are a truncation, not the full M;
    the untruncated caller passes taper=False since nothing is being discarded)."""
    M = h_alias.shape[0]
    h_out = np.zeros((num_res,) + h_alias.shape[1:], dtype=complex)
    if causal_len > 0:
        w = edge_taper(causal_len) if taper else np.ones(causal_len)
        h_out[:causal_len] = h_alias[:causal_len] * w[:, None]
    if anticausal_len > 0:
        w = edge_taper(anticausal_len)[::-1] if taper else np.ones(anticausal_len)
        h_out[num_res - anticausal_len:] = h_alias[M - anticausal_len:] * w[:, None]
    return h_out


def estimate_channel_from_dmrs(rx_dmrs: torch.Tensor, known_tx: dict, n_ants: int, num_res: int,
                                n_users: int, return_untruncated: bool = False,
                                estimate_noise_var: bool = False):
    """Per user, per antenna: LS+deocc at each DMRS pilot position (averaged over every DMRS
    occasion in the region - e.g. the two in-slot symbols x every slot in a group), then an IFFT
    delay-domain denoise/interpolate to fill every RE. Returns (num_res, n_ants, n_users)
    complex128, replacing what a dedicated-calibration-slot ChannelEstimate used to produce.

    Step 1 (LS + deocc + time-average): for occ_active users (n_users==4), each RE-pair is
    deocc-combined into one estimate (spacing-4 resolution across num_res - "3 missing REs out
    of 4"); otherwise (n_users in (1,2)) every comb RE gets its own independent estimate
    (spacing-2 - "every other RE") since there's no second port sharing it to separate out.

    Step 2 (delay domain): IFFT the user's M uniformly-spaced pilot estimates -> an aliased
    delay-domain estimate; white noise spreads evenly across all M delay bins while true channel
    energy concentrates within the delay spread, so truncating to L_taps (from conf.delay_spread)
    and zeroing the rest is a large, SNR-independent noise reduction. L_taps is split evenly
    between causal and anticausal (wrapped/pre-cursor - see fold_delay_taps), the same ratio the
    untruncated path uses (there, an even split isn't a policy choice, it's the only correct one).
    Zero-pad back to num_res (correctly split, not just appended - fold_delay_taps) and FFT ->
    full-resolution H, including at the original pilot REs (denoised the same as everywhere else,
    not left as their raw single-shot LS value) and at every off-comb RE (filled in purely by this
    interpolation - see fold_delay_taps/edge_taper).

    return_untruncated (diagnostic only): also returns a second (num_res, n_ants, n_users)
    estimate from the *same* h_alias with no truncation/taper at all (L_taps=M, i.e. plain DFT
    interpolation of the raw pilot estimates, no denoising assumption) - split exactly at the
    Nyquist point M//2 (the only correct split when nothing is discarded, unlike L_taps's
    causal/anticausal ratio above, which is a truncation policy choice). Comparing the two on a
    noise_var=0 pass isolates exactly what the L_taps truncation choice is doing to the estimate,
    with noise out of the picture entirely.

    estimate_noise_var: also returns a receiver-side noise_var estimate (see noise_var_terms
    below) - the DMRS-pilot analog of what LmmseEqualize computes inline from its own LS
    estimate. Meaningless (and not requested) on the noise_var=0 save_diag pass, so this and
    return_untruncated are never both True in practice.

    Returns a dict: {'H': ..., and optionally 'H_untrunc'/'noise_var_est'} - a dict rather than
    a positional tuple since which optional fields are present depends on which of the two
    independent flags above is set."""
    delta_f = SAMPLING_RATE / FFT_size  # real subcarrier spacing (Hz)
    delay_bin = 1.0 / (num_res * delta_f)
    rx_np = rx_dmrs.cpu().numpy()  # (num_occasions, n_ants, num_res)
    H = np.zeros((num_res, n_ants, n_users), dtype=complex)
    H_untrunc = np.zeros((num_res, n_ants, n_users), dtype=complex) if return_untruncated else None
    # noise_var_terms: per-(user, pilot RE) residual of the raw, undivided per-occasion rx sample
    # around its own across-occasion mean - the same idea LmmseEqualize (lmmse_equalizer.py:74/80)
    # uses for its noise_var, adapted to this estimator's pilot layout. Averaged over ~num_res/2
    # pilot REs (all entries, all users) below, not just the DMRS occasions in isolation - pooling
    # across REs meaningfully reduces the final noise_var_est's variance.
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
                # reference amplitude (which includes DMRS_POWER_BOOST_DB and
                # CONSTELLATION_FACTOR[4]), silently deflating the noise_var estimate by that
                # factor squared. h_true*val is constant across occasions either way (block
                # fading + val doesn't vary per occasion), so raw - mean(raw) equals
                # noise - mean(noise) exactly regardless of amplitude or FD-OCC combining, with
                # no rescaling needed.
                for re_i in (re_a, re_b):
                    if re_i is not None:
                        raw = rx_np[:, :, re_i]
                        noise_var_terms.append(np.mean(np.abs(raw - raw.mean(axis=0, keepdims=True)) ** 2))
        pair_est = np.stack(pair_est, axis=0)              # (M, n_ants)
        M = pair_est.shape[0]

        h_alias = np.fft.ifft(pair_est, axis=0)             # (M, n_ants), aliased delay-domain estimate

        L_taps = int(np.clip(np.ceil(DMRS_DELAY_TRUNC_MARGIN * conf.delay_spread / delay_bin), 1, M))
        anticausal_trunc = L_taps // 2
        causal_trunc = L_taps - anticausal_trunc
        h_trunc = fold_delay_taps(h_alias, num_res, causal_trunc, anticausal_trunc, taper=True)
        H[:, :, user] = np.fft.fft(h_trunc, axis=0)          # (num_res, n_ants), full-resolution

        if return_untruncated:
            anticausal_full = M // 2
            causal_full = M - anticausal_full
            h_full = fold_delay_taps(h_alias, num_res, causal_full, anticausal_full, taper=False)
            H_untrunc[:, :, user] = np.fft.fft(h_full, axis=0)

    result = {'H': torch.from_numpy(H)}
    if return_untruncated:
        result['H_untrunc'] = torch.from_numpy(H_untrunc)
    if estimate_noise_var:
        result['noise_var_est'] = float(np.mean(noise_var_terms)) if noise_var_terms else 0.0
    return result


def lmmse_equalize_with_H(H: torch.Tensor, rx_c: torch.Tensor, noise_var, re: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same linear-MMSE equalization math as LmmseEqualize (lmmse_equalizer.py lines 61-70),
    but against an H estimated elsewhere (from that region's own embedded DMRS here - see
    estimate_channel_from_dmrs) instead of re-estimating it from rx_c itself - LmmseEqualize
    always re-estimates H from whatever it's given, which would leak the answer if rx_c were the
    same symbols being scored.

    H is either a single (n_ants, n_users) matrix - ekf.py's usage, one fixed H per group,
    applied uniformly to every symbol in rx_c - or a per-symbol (rx_c.shape[0], n_ants, n_users)
    array - conf.chanestmode's usage, H already broadcast out to per-symbol shape by the caller
    (evaluate.py) so it can index it exactly like rx_c/equalized/postEqSINR with no group-boundary
    logic of its own. noise_var follows the same either/or (scalar, or a (rx_c.shape[0],) array).
    postEqSINR comes back shaped to match: a single (n_users,) vector in the single-H case, or a
    (rx_c.shape[0], n_users) array in the per-symbol case - callers (LmmseDemod) branch on its
    ndim the same way."""
    per_symbol = H.dim() == 3
    n_users = H.shape[-1]
    I_users = torch.eye(n_users, dtype=H.dtype, device=H.device)
    equalized = torch.zeros(rx_c.shape[0], n_users, dtype=torch.cfloat)
    if not per_symbol:
        W = torch.linalg.inv(H.T.conj() @ H + noise_var * I_users) @ H.T.conj()
        bias = (W @ H).diag().real
        W = W.cpu()
        bias = bias.cpu()
        for i in range(rx_c.shape[0]):
            equalized[i, :] = torch.matmul(W, rx_c[i, :, re]) / bias
        postEqSINR = bias / (1 - bias)
    else:
        postEqSINR = torch.zeros(rx_c.shape[0], n_users)
        for i in range(rx_c.shape[0]):
            H_i = H[i]
            nv_i = noise_var[i] if torch.is_tensor(noise_var) and noise_var.dim() > 0 else noise_var
            W_i = torch.linalg.inv(H_i.T.conj() @ H_i + nv_i * I_users) @ H_i.T.conj()
            bias_i = (W_i @ H_i).diag().real
            equalized[i, :] = torch.matmul(W_i.cpu(), rx_c[i, :, re]) / bias_i.cpu()
            postEqSINR[i, :] = (bias_i / (1 - bias_i)).cpu()
    return equalized, postEqSINR
