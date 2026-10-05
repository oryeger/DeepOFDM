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
DMRS_DELAY_TRUNC_MARGIN = 12  # fallback window for non-TDL models: margin * delay_spread (see _dd_fallback_extent)

# 3GPP 38.901 Tables 7.7.2-1..5: last-path delay of TDL-A..E in units of the RMS delay spread.
# Used only for the fallback delay-domain window (no noise estimate available - see
# estimate_channel_from_dmrs); the normal path sizes the window from the data.
TDL_LAST_PATH_NORM_DELAY = {'A': 9.6586, 'B': 4.7834, 'C': 8.6523, 'D': 12.5254, 'E': 20.6519}
# Sionna's discrete-time channel filter spills ~6 samples on each side of every path
# (time_lag_discrete_time_channel's l_min=-6 / l_max=...+6), and every channel model aligns the
# receiver to the strongest tap (TA=argmax), so the window must cover at least this much on
# both sides of delay 0.
DD_FILTER_SPILL_SAMPLES = 6


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


def genie_ici_noise_var(H, mod_data: int) -> float:
    """TEMPORARY (always on): genie-aided ICI variance to add on top of the
    thermal noise_var fed to LMMSE, so its postEqSINR/LLRs account for the residual ICI that
    genie_cfo_comp_vector leaves in by design. Uses the true conf.cfo (genie) - meant only to
    confirm the LMMSE BLER/MI turnaround at high SNR is LLR overconfidence from unmodeled ICI;
    to be replaced by a receiver-side (non-genie) estimate.

    Fully loaded OFDM symbol, CFO eps (in scs): desired-subcarrier gain sinc(eps), ICI power
    (1 - sinc^2(eps)) * Es * |h|^2. H here is the DMRS estimate, which already absorbs the
    sinc(eps) gain, so |h|^2 = |H|^2 / sinc^2(eps). Per receive antenna, interference sums over
    users; averaged over REs and antennas. Returns 0.0 when cfo == 0."""
    if conf.cfo == 0:
        return 0.0
    H_np = H.cpu().numpy() if torch.is_tensor(H) else np.asarray(H)  # (num_res, n_ants, n_users)
    s2 = float(np.sinc(conf.cfo)) ** 2                                # np.sinc(x) = sin(pi x)/(pi x)
    h_pwr = float(np.mean(np.sum(np.abs(H_np) ** 2, axis=-1)))       # sum over users, mean over RE/ant
    return (1.0 - s2) / s2 * CONSTELLATION_FACTOR[mod_data] * h_pwr


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


def _dd_fallback_extent() -> float:
    """Delay extent (s, |delay| from the strongest tap) used when the window can't be sized from
    the data (a single DMRS occasion -> no noise estimate). TDL-A..E: the model's last-path delay
    (TDL_LAST_PATH_NORM_DELAY x delay_spread); anything else (QuaDRiGa/Sionna UMa/UMi/RMa, whose
    conf.delay_spread is only a 300 ns placeholder): DMRS_DELAY_TRUNC_MARGIN x delay_spread.
    Plus the filter spill either way."""
    model = str(getattr(conf, 'channel_model', 'C'))
    spill = DD_FILTER_SPILL_SAMPLES / SAMPLING_RATE
    if model[:1] in TDL_LAST_PATH_NORM_DELAY and model not in ('UMa', 'UMi', 'RMa'):
        return TDL_LAST_PATH_NORM_DELAY[model[:1]] * float(conf.delay_spread) + spill
    return DMRS_DELAY_TRUNC_MARGIN * float(conf.delay_spread) + spill


def _genie_pilot_ici_var(known_tx: dict, user: int, num_res: int, h_pwr: float) -> float:
    """TEMPORARY (genie, same conf.cfo as genie_cfo_comp_vector/genie_ici_noise_var): ICI variance
    on this user's LS pilot estimates (channel units). Unlike thermal noise it is identical in every
    DMRS occasion (same pilots, frozen channel), so the across-occasion residual can't see it and
    occasion averaging doesn't reduce it - but across pilot REs it is effectively white (random
    QPSK neighbours), so for the delay-domain window it is noise. Exact ICI coefficients for an
    FFT_size-point OFDM symbol, |c_l|^2 = sin^2(pi*eps) / (N^2 sin^2(pi*(l+eps)/N)), summed over
    every RE that carries DMRS energy on the DMRS symbols (all users), with mean |H|^2 = h_pwr.
    Returns 0.0 when cfo == 0."""
    eps = float(getattr(conf, 'cfo', 0.0))
    if eps == 0:
        return 0.0
    tx_pwr = np.zeros(num_res)
    for info in known_tx.values():
        for re_a, re_b, val_a, val_b in info['entries']:
            tx_pwr[re_a] += abs(val_a) ** 2
            if re_b is not None:
                tx_pwr[re_b] += abs(val_b) ** 2
    Nf = FFT_size
    l = np.arange(-(num_res - 1), num_res)
    c2 = (np.sin(np.pi * eps) / (Nf * np.sin(np.pi * (l + eps) / Nf))) ** 2
    c2[l == 0] = 0.0
    own = []
    for re_a, re_b, val_a, val_b in known_tx[user]['entries']:
        for re_k, v in ((re_a, val_a), (re_b, val_b)):
            if re_k is not None:
                interf = np.sum(c2[np.arange(num_res) - re_k + (num_res - 1)] * tx_pwr)   # rx-domain, per |H|^2
                own.append(interf / abs(v) ** 2)
    scale = 0.5 if known_tx[user]['occ_active'] else 1.0      # deocc averages two REs
    return scale * float(np.mean(own)) * h_pwr


def _mirror_dd_interpolate(p: np.ndarray, pos: np.ndarray, num_res: int, sigma2_p: float,
                           delta_f: float, full: bool = False):
    """Delay-domain denoise + interpolate one user's pilot estimates, all antennas at once.

    p: (M, n_ants) LS pilot estimates at uniformly spaced RE positions pos (spacing S, possibly a
    fractional start - FD-OCC pair midpoints). sigma2_p: noise variance of each pilot estimate
    (after occasion averaging/deocc), or None if unknown.

    1) Mirror extension: [p, p[::-1]] (length N=2M) is continuous across both the band edges and
       the period wrap, so its IFFT is compact - no Gibbs leakage from the jump between the last
       and first RE that a plain M-point IFFT sees (that jump was the -13.5 dB noise-free floor of
       the old estimator, worst at the band edges). Delay bins are 1/(N*S*delta_f) wide; a path at
       delay d (relative to the strongest tap) shows up at bin +d and its mirror image at -d, so
       the kept window is symmetric |n| <= c and covers paths both after AND before the strongest
       tap (QuaDRiGa UMa has significant energy >1 us before it).
    2) Window c from the data: per-bin power averaged over antennas, folded (n, -n), and c chosen
       to minimise estimated MSE = (signal energy dropped beyond c) + (noise kept inside c), with
       per-bin noise sigma2_p/N and signal energy estimated as max(power - noise, 0). Lower bound:
       the filter spill; upper bound: everything. Low SNR -> tight window (denoising), high SNR ->
       window opens up (accuracy). If sigma2_p is None, fall back to _dd_fallback_extent().
    3) Evaluate the windowed delay-domain estimate at every RE's (fractional) pilot-grid position
       x = (re - pos[0]) / S directly (exact for any offset - odd comb, OCC midpoints).

    Returns (H (num_res, n_ants), c_bins, bin_width_s)."""
    M = p.shape[0]
    S = float(pos[1] - pos[0]) if M > 1 else 1.0
    N = 2 * M
    bin_w = 1.0 / (N * S * delta_f)
    h = np.fft.ifft(np.concatenate([p, p[::-1]], axis=0), axis=0)          # (N, n_ants)
    n_signed = np.fft.fftfreq(N) * N                                         # 0..M-1, -M..-1

    c_min = int(np.ceil(DD_FILTER_SPILL_SAMPLES / SAMPLING_RATE / bin_w)) + 1
    c_max = M - 1
    if full:
        c = c_max
    elif sigma2_p is not None and sigma2_p > 0:
        pw = np.mean(np.abs(h) ** 2, axis=1)                                 # (N,), antenna-averaged
        fold = np.empty(M)
        fold[0] = pw[0]
        fold[1:] = pw[1:M] + pw[N - np.arange(1, M)]                         # |n| = 1..M-1
        noise_bin = sigma2_p / N
        noise_fold = np.full(M, 2.0 * noise_bin)
        noise_fold[0] = noise_bin
        excess = np.maximum(fold - noise_fold, 0.0)
        dropped = np.concatenate([np.cumsum(excess[::-1])[::-1][1:], [0.0]])  # energy beyond c
        kept_noise = np.cumsum(noise_fold)                                   # noise inside |n|<=c
        cost = dropped + kept_noise
        cand = np.arange(c_min, c_max + 1) if c_min <= c_max else np.array([c_max])
        c = int(cand[np.argmin(cost[cand])])
    else:
        c = int(np.clip(np.ceil(_dd_fallback_extent() / bin_w), c_min, c_max))

    keep = np.abs(n_signed) <= c
    x = (np.arange(num_res) - pos[0]) / S                                    # (num_res,)
    E = np.exp(-2j * np.pi * np.outer(x, n_signed[keep]) / N)                # (num_res, n_kept)
    return E @ h[keep], c, bin_w


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

    Step 2 (delay domain, _mirror_dd_interpolate): mirror-extend the user's M pilot estimates to
    2M (removes the band-edge wrap discontinuity that limited the old plain-IFFT estimator to a
    ~-13.5 dB noise-free NMSE on TDL-C), IFFT, keep the delay window |n| <= c, and evaluate at
    every RE's exact (fractional) pilot-grid position. c is chosen per user from the data (MSE-
    minimising cut on the antenna-averaged power-delay profile against the pilot noise level), so
    it adapts to the actual channel - TDL-A..E at any delay spread, and QuaDRiGa UMa/UMi/RMa whose
    conf.delay_spread is only a placeholder and whose per-seed spreads vary by >10x. Only with a
    single DMRS occasion (no noise estimate) does it fall back to a model-based extent
    (_dd_fallback_extent). result['dd_window_s'] = {user: window half-width in seconds}.

    return_untruncated (diagnostic only): also returns a second (num_res, n_ants, n_users)
    estimate from the same mirrored delay-domain sequence with nothing discarded (c = M-1) - on a
    noise_var=0 pass, comparing the two isolates what the window choice is doing.

    estimate_noise_var: also returns a receiver-side noise_var estimate (see noise_var_terms
    below) - the DMRS-pilot analog of what LmmseEqualize computes inline from its own LS
    estimate. Meaningless (and not requested) on the noise_var=0 save_diag pass, so this and
    return_untruncated are never both True in practice.

    Returns a dict: {'H': ..., and optionally 'H_untrunc'/'noise_var_est'} - a dict rather than
    a positional tuple since which optional fields are present depends on which of the two
    independent flags above is set."""
    delta_f = SAMPLING_RATE / FFT_size  # real subcarrier spacing (Hz)
    rx_np = rx_dmrs.cpu().numpy()  # (num_occasions, n_ants, num_res)
    num_occ = rx_np.shape[0]
    # Residual around the mean of num_occ occasions has expectation sigma^2*(num_occ-1)/num_occ -
    # undo that bias (2 occasions/slot -> x2). With a single occasion the residual is identically
    # 0: no noise estimate, and the delay-domain window falls back to _dd_fallback_extent().
    dof_corr = num_occ / (num_occ - 1) if num_occ > 1 else None
    H = np.zeros((num_res, n_ants, n_users), dtype=complex)
    H_untrunc = np.zeros((num_res, n_ants, n_users), dtype=complex) if return_untruncated else None
    # noise_var_terms: per-(user, pilot RE) residual of the raw, undivided per-occasion rx sample
    # around its own across-occasion mean - the same idea LmmseEqualize (lmmse_equalizer.py:74/80)
    # uses for its noise_var, adapted to this estimator's pilot layout. Pooled over all pilot REs
    # and users for noise_var_est; always computed (cheap) because it also sizes the delay-domain
    # window below, whether or not the caller asked for noise_var_est.
    #
    # Residual on the RAW (undivided, uncombined) rx samples, not on the LS estimates - dividing
    # by val_a/val_b first would scale the residual by the DMRS reference amplitude (which includes
    # DMRS_POWER_BOOST_DB and CONSTELLATION_FACTOR[4]). h_true*val is constant across occasions
    # (block fading + val doesn't vary per occasion), so raw - mean(raw) equals
    # noise - mean(noise) regardless of amplitude or FD-OCC combining.
    noise_var_terms = []
    dd_window_s = {}
    for user, info in known_tx.items():
        pair_est, pair_pos, inv_val_pwr = [], [], []
        for re_a, re_b, val_a, val_b in info['entries']:
            if info['occ_active']:
                if re_b is not None:
                    est = 0.5 * (rx_np[:, :, re_a] / val_a + rx_np[:, :, re_b] / val_b)
                    pair_pos.append(0.5 * (re_a + re_b))       # deocc estimate sits at the pair midpoint
                    inv_val_pwr.append(0.25 * (1 / abs(val_a) ** 2 + 1 / abs(val_b) ** 2))
                else:
                    est = rx_np[:, :, re_a] / val_a
                    pair_pos.append(float(re_a))
                    inv_val_pwr.append(1 / abs(val_a) ** 2)
                pair_est.append(est.mean(axis=0))
            else:
                pair_est.append((rx_np[:, :, re_a] / val_a).mean(axis=0))
                pair_pos.append(float(re_a))
                inv_val_pwr.append(1 / abs(val_a) ** 2)
                if re_b is not None:
                    pair_est.append((rx_np[:, :, re_b] / val_b).mean(axis=0))
                    pair_pos.append(float(re_b))
                    inv_val_pwr.append(1 / abs(val_b) ** 2)
            if dof_corr is not None:
                for re_i in (re_a, re_b):
                    if re_i is not None:
                        raw = rx_np[:, :, re_i]
                        noise_var_terms.append(np.mean(np.abs(raw - raw.mean(axis=0, keepdims=True)) ** 2))
        pair_est = np.stack(pair_est, axis=0)              # (M, n_ants)
        pair_pos = np.asarray(pair_pos)
        # Mirror/DFT interpolation needs uniform spacing; a trailing unpaired RE (odd comb count)
        # breaks it - drop such irregular trailing points from the fit (their REs still get
        # interpolated like every other RE).
        if pair_pos.size > 2:
            step = pair_pos[1] - pair_pos[0]
            regular = np.concatenate([[True], np.isclose(np.diff(pair_pos), step)])
            n_reg = int(np.argmin(regular)) if not regular.all() else regular.size
            pair_est, pair_pos = pair_est[:n_reg], pair_pos[:n_reg]
            inv_val_pwr = inv_val_pwr[:n_reg]

        # Per-pilot-estimate error variance (sizes the window): thermal = rx noise variance /
        # |val|^2 per occasion (deocc halves it), averaged over num_occ occasions; plus the pilot
        # ICI, which is identical in every occasion (invisible to the residual, not reduced by
        # averaging) but white across pilots - without it the window opens up to "fit" the ICI
        # (measured: ~1 dB worse than the old estimator at cfo=0.2, SNR>=20 dB).
        if dof_corr is not None and noise_var_terms:
            nv_rx = float(np.mean(noise_var_terms)) * dof_corr
            sigma2_p = nv_rx * float(np.mean(inv_val_pwr)) / num_occ
            sigma2_p += _genie_pilot_ici_var(known_tx, user, num_res, float(np.mean(np.abs(pair_est) ** 2)))
        else:
            sigma2_p = None

        H[:, :, user], c_bins, bin_w = _mirror_dd_interpolate(pair_est, pair_pos, num_res, sigma2_p, delta_f)
        dd_window_s[user] = c_bins * bin_w

        if return_untruncated:
            H_untrunc[:, :, user], _, _ = _mirror_dd_interpolate(pair_est, pair_pos, num_res, None, delta_f,
                                                                  full=True)

    result = {'H': torch.from_numpy(H), 'dd_window_s': dd_window_s}
    if return_untruncated:
        result['H_untrunc'] = torch.from_numpy(H_untrunc)
    if estimate_noise_var:
        result['noise_var_est'] = (float(np.mean(noise_var_terms)) * dof_corr
                                   if (noise_var_terms and dof_corr is not None) else 0.0)
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
