#!/usr/bin/env python
"""
dmrs_ce_diag.py -- Isolate the DMRS channel-estimation loss of LMMSE (chanestmode='dmrs')
against a genie and against the legacy whole-block CE, in the frequency domain only.

Why: fig34 (dmrs CE) put LMMSE BLER@10% at ~16 dB, fileg (legacy CE, same config) at ~9 dB.
This script calls the REAL estimator (coding/dmrs_pilots.py: dmrs_layout, dmrs_reference_values,
dmrs_known_tx, estimate_channel_from_dmrs) on a frequency-domain y = H s + w model, so the only
thing under test is the estimator itself - no CP/ISI, no Sionna filter spill, no CFO (cfo forced 0),
static channel (Mult.yaml speed=0), one slot per group (Mult.yaml slots_per_group=-1 -> 1 in
mimo_channel_dataset) -> 2 DMRS occasions.

Channel: TDL-C from 3GPP 38.901 Table 7.7.2-3 (independent Rayleigh taps, i.i.d. across rx/tx
antennas, same "users = tx antennas of one TDL" convention as TLD_channel.py), timing-aligned to the
strongest tap (TA=argmax, as the real pipeline does). Built here directly because Sionna does not
import in the local Windows env (mitsuba/LLVM).

Units follow the codebase exactly: data on the commpy odd-integer grid (16QAM Es=CONSTELLATION_FACTOR=10),
noise_var = 10^(-snr/10) * CONSTELLATION_FACTOR[mod_data] (evaluate.py), DMRS amplitude from
dmrs_known_tx.

Detectors (LMMSE, same math as lmmse_equalize_with_H):
    genie        true H, sigma^2/Es (MMSE-correct)
    genie_rawnv  true H, sigma^2    (the old, over-regularized convention - for reference)
    dmrs         estimate_channel_from_dmrs + its noise_var_est/Es (= lmmse_equalize_mmse, i.e.
                 exactly what evaluate.py / ekf.py now do: DMRS EPRE = data Es + 3 dB, FD-OCC SIC)
    legacy       whole-block LS on known data symbols (LmmseEqualize as-is)

Reported per (n_users, SNR): channel NMSE [dB], LMMSE post-eq SINR (Es/MSE) [dB] and uncoded SER,
plus SNR at SER=1e-2 (loss vs genie) and per-RE NMSE of 'dmrs'.
Old-estimator reference (QPSK-energy DMRS, pairwise deocc, sigma^2 regularizer, 24 RE, 200 trials):
loss vs genie at SER=1e-2 = +1.9 / +4.8 / +7.9 dB at n_users 1 / 2 / 4 (current code: +0.5 / +0.9 /
+1.8 dB). To re-measure the old estimator, run this script against the pre-fix dmrs_pilots.py from git.

Run (from C:\\Projects\\DeepOFDM, KMP_DUPLICATE_LIB_OK=TRUE in this Windows env):
    python -m python_code.utils.dmrs_ce_diag
    python -m python_code.utils.dmrs_ce_diag --num_res 96 --users 4 --sic_iters 3
Output: <Scratchpad>\\Analysis\\dmrsdiag[_<tag>]\\dmrs_ce_diag_*.csv / .png / _perRE.png
"""
import argparse
import csv
import os

import numpy as np
import torch

from python_code import conf
from python_code.coding import dmrs_pilots
from python_code.coding.dmrs_pilots import (CONSTELLATION_FACTOR, dmrs_known_tx, dmrs_layout,
                                            dmrs_reference_values, estimate_channel_from_dmrs)
from python_code.utils.constants import DMRS_SYMBOL_LOCAL_IDX, FFT_size, SAMPLING_RATE

# 3GPP 38.901 Table 7.7.2-3 TDL-C: (normalized delay, power dB)
TDL_C = np.array([
    [0.0000, -4.4], [0.2099, -1.2], [0.2219, -3.5], [0.2329, -5.2], [0.2176, -2.5], [0.6366, 0.0],
    [0.6448, -2.2], [0.6560, -3.9], [0.6584, -7.4], [0.7935, -7.1], [0.8213, -10.7], [0.9336, -11.1],
    [1.2285, -5.1], [1.3083, -6.8], [2.1704, -8.7], [2.7105, -13.2], [4.2589, -13.9], [4.6003, -13.9],
    [5.4902, -15.8], [5.6077, -17.1], [6.3065, -16.0], [6.6374, -15.7], [7.0427, -21.6], [8.6523, -22.8]])

N_LEGACY = 1000  # legacy LS: known 16QAM symbols per user per RE (whole block / n_users, separate_pilots)

DETECTORS = ['genie', 'genie_rawnv', 'dmrs', 'legacy']
COLORS = {'genie': 'k', 'genie_rawnv': 'gray', 'dmrs': 'b', 'legacy': 'g'}


def _qam_grid(mod: int) -> np.ndarray:
    m = int(round(np.sqrt(mod)))
    lv = np.arange(-(m - 1), m, 2)
    return (lv[:, None] + 1j * lv[None, :]).ravel()


def tdl_c_freq_response(rng, trials, num_res, n_ants, n_users, delay_spread, delta_f):
    """(trials, num_res, n_ants, n_users) complex, E|h|^2 = 1, aligned to the strongest tap."""
    tau = TDL_C[:, 0] * delay_spread
    pw = 10 ** (TDL_C[:, 1] / 10)
    pw = pw / pw.sum()
    P = tau.size
    a = (rng.standard_normal((trials, P, n_ants, n_users)) +
         1j * rng.standard_normal((trials, P, n_ants, n_users))) * np.sqrt(pw / 2)[None, :, None, None]
    strongest = np.argmax(np.sum(np.abs(a) ** 2, axis=(2, 3)), axis=1)          # (trials,)
    tau_rel = tau[None, :] - tau[strongest][:, None]                            # (trials, P)
    f = np.arange(num_res) * delta_f
    phase = np.exp(-2j * np.pi * f[None, :, None] * tau_rel[:, None, :])        # (trials, R, P)
    return np.einsum('trp,tpau->trau', phase, a)


def _legacy_estimate(rng, H, sigma2, grid):
    """LmmseEqualize (separate_pilots) on N_LEGACY known data symbols of a user transmitting alone:
    H_ls = H + mean_k(w_k / s_k), noise_var = mean_k |w_k/s_k - mean|^2. Given the symbols, the error
    is CN(0, sigma^2 * mean(1/|s_k|^2) / N); drawn directly (exact in distribution) instead of
    materializing N_LEGACY x trials x RE x ants noise samples. noise_var uses E[1/|s|^2] the same way."""
    inv_e = np.mean(1.0 / np.abs(grid) ** 2)                                     # E[1/|s|^2] (uniform)
    err_var = sigma2 * inv_e / N_LEGACY
    err = (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape)) * np.sqrt(err_var / 2)
    return H + err, sigma2 * inv_e * (1 - 1.0 / N_LEGACY)


def _dmrs_estimate(rng, H, sigma2, known_tx, n_ants, num_res, n_users, num_occ):
    """Real estimate_channel_from_dmrs on every trial.
    Returns H_est (T,R,A,U), nv_est (T,) [rx-domain, as estimate_channel_from_dmrs returns it]."""
    T = H.shape[0]
    s = np.zeros((n_users, num_res), dtype=complex)
    for u, info in known_tx.items():
        for re_a, re_b, va, vb in info['entries']:
            s[u, re_a] = va
            if re_b is not None:
                s[u, re_b] = vb
    clean = np.einsum('trau,ur->tra', H, s)                                      # (T, R, A)
    H_est = np.zeros_like(H)
    nv = np.zeros(T)
    for t in range(T):
        w = (rng.standard_normal((num_occ, n_ants, num_res)) +
             1j * rng.standard_normal((num_occ, n_ants, num_res))) * np.sqrt(sigma2 / 2)
        rx = clean[t].T[None, :, :] + w                                          # (occ, A, R)
        est = estimate_channel_from_dmrs(torch.from_numpy(rx), known_tx, n_ants, num_res, n_users,
                                         estimate_noise_var=True)
        H_est[t] = est['H'].numpy()
        nv[t] = est['noise_var_est']
    return H_est, nv


def _lmmse(rng, H_true, H_used, nv_used, sigma2, grid, n_data):
    """LMMSE (same math as LmmseEqualize / lmmse_equalize_with_H: W=(H^H H+nv I)^-1 H^H, bias
    normalization) on n_data fresh symbols per RE through H_true. Returns (SINR_eff_dB, SER)."""
    T, R, A, U = H_true.shape
    Es = np.mean(np.abs(grid) ** 2)
    s = grid[rng.integers(0, grid.size, size=(T, R, U, n_data))]
    w = (rng.standard_normal((T, R, A, n_data)) + 1j * rng.standard_normal((T, R, A, n_data))) * np.sqrt(sigma2 / 2)
    y = H_true @ s + w                                                           # (T, R, A, Nd)
    Hh = np.conj(np.swapaxes(H_used, -1, -2))                                    # (T, R, U, A)
    nv = np.asarray(nv_used, dtype=float).reshape(-1, 1, 1, 1) if np.ndim(nv_used) else nv_used
    G = Hh @ H_used + nv * np.eye(U)[None, None]
    W = np.linalg.solve(G, Hh)                                                   # (T, R, U, A)
    bias = np.real(np.einsum('truu->tru', W @ H_used))[..., None]               # (T, R, U, 1)
    s_hat = (W @ y) / bias
    mse = np.mean(np.abs(s_hat - s) ** 2)
    m = int(round(np.sqrt(grid.size)))
    q = lambda x: np.clip(2 * np.floor(x / 2) + 1, -(m - 1), m - 1)
    ser = np.mean((q(s_hat.real) != s.real) | (q(s_hat.imag) != s.imag))
    return 10 * np.log10(Es / mse), ser


def _nmse_db(H_est, H):
    return 10 * np.log10(np.mean(np.abs(H_est - H) ** 2) / np.mean(np.abs(H) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', default='1,2,4')
    ap.add_argument('--snrs', default='0,40,2', help='start,stop,step (inclusive)')
    ap.add_argument('--num_res', type=int, default=24)
    ap.add_argument('--n_ants', type=int, default=8)
    ap.add_argument('--mod', type=int, default=16)
    ap.add_argument('--trials', type=int, default=200, help='channel realizations per point')
    ap.add_argument('--n_data', type=int, default=100, help='data symbols per RE per trial')
    ap.add_argument('--slots_per_group', type=int, default=1)
    ap.add_argument('--sic_iters', type=int, default=None, help='override dmrs_pilots.DMRS_OCC_SIC_ITERS')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    conf.cfo = 0                      # no CFO/ICI terms in the estimator
    conf.channel_model = 'C'
    if args.sic_iters is not None:
        dmrs_pilots.DMRS_OCC_SIC_ITERS = args.sic_iters
    delay_spread = float(conf.delay_spread)
    delta_f = SAMPLING_RATE / FFT_size
    users = [int(u) for u in args.users.split(',')]
    a0, a1, st = (float(x) for x in args.snrs.split(','))
    snrs = np.arange(a0, a1 + 1e-9, st)
    grid = _qam_grid(args.mod)
    Es = CONSTELLATION_FACTOR[args.mod]
    assert abs(np.mean(np.abs(grid) ** 2) - Es) < 1e-9
    num_occ = len(DMRS_SYMBOL_LOCAL_IDX) * args.slots_per_group

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(os.path.dirname(root), 'Scratchpad', 'Analysis',
                           'dmrsdiag' + (f'_{args.tag}' if args.tag else ''))
    os.makedirs(out_dir, exist_ok=True)
    stem = (f'dmrs_ce_diag_R{args.num_res}_A{args.n_ants}_{args.mod}Q_spg{args.slots_per_group}'
            f'_sic{dmrs_pilots.DMRS_OCC_SIC_ITERS}')
    print(f'[dmrsdiag] TDL-C ds={delay_spread * 1e9:.0f}ns R={args.num_res} A={args.n_ants} '
          f'{args.mod}QAM Es={Es} occ={num_occ} trials={args.trials} '
          f'sic_iters={dmrs_pilots.DMRS_OCC_SIC_ITERS} -> {out_dir}', flush=True)

    rows, per_re = [], []
    for U in users:
        rng = np.random.default_rng(args.seed + U)
        H = tdl_c_freq_response(rng, args.trials, args.num_res, args.n_ants, U, delay_spread, delta_f)
        layout = dmrs_layout(U, args.num_res)
        ref = dmrs_reference_values(rng, args.num_res)
        known = dmrs_known_tx(layout, ref, U, mod_data=args.mod)
        print(f"[dmrsdiag] n_users={U}: DMRS EPRE vs data Es = "
              f"{10 * np.log10(abs(known[0]['entries'][0][2]) ** 2 / Es):+.1f} dB, "
              f"occ_active={layout[0]['occ_active']}", flush=True)
        for snr in snrs:
            sigma2 = 10 ** (-snr / 10) * Es
            row = {'n_users': U, 'snr': snr, 'nmse_noise_ref': 10 * np.log10(sigma2 / Es)}
            pairs = {'genie': (H, sigma2 / Es), 'genie_rawnv': (H, sigma2)}
            H_d, nv_d = _dmrs_estimate(rng, H, sigma2, known, args.n_ants, args.num_res, U, num_occ)
            row['nmse_dmrs'] = _nmse_db(H_d, H)
            row['nv_ratio_dmrs'] = 10 * np.log10(np.mean(nv_d) / sigma2)   # ~0 dB if the estimate is right
            pairs['dmrs'] = (H_d, nv_d / Es)                                 # = lmmse_equalize_mmse
            if snr in (10, 20, 30):
                e = np.mean(np.abs(H_d - H) ** 2, axis=(0, 2, 3)) / np.mean(np.abs(H) ** 2)
                per_re.append((U, 'dmrs', snr, 10 * np.log10(e)))
            H_l, nv_l = _legacy_estimate(rng, H, sigma2, grid)
            row['nmse_legacy'] = _nmse_db(H_l, H)
            pairs['legacy'] = (H_l, nv_l)
            for name in DETECTORS:
                Hu, nvu = pairs[name]
                row[f'sinr_{name}'], row[f'ser_{name}'] = _lmmse(rng, H, Hu, nvu, sigma2, grid, args.n_data)
            rows.append(row)
            print(f"  U={U} SNR={snr:5.1f}  NMSE dmrs={row['nmse_dmrs']:6.1f} leg={row['nmse_legacy']:6.1f} "
                  f"(noise ref {row['nmse_noise_ref']:6.1f})  nv_est/sigma^2={row['nv_ratio_dmrs']:+.1f} dB  "
                  f"SINR genie={row['sinr_genie']:5.1f} dmrs={row['sinr_dmrs']:5.1f} "
                  f"legacy={row['sinr_legacy']:5.1f}", flush=True)

    csv_path = os.path.join(out_dir, stem + '.csv')
    with open(csv_path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print('\n[dmrsdiag] SNR [dB] at uncoded SER=1e-2 (loss vs genie):')
    for U in users:
        rr = [r for r in rows if r['n_users'] == U]
        x = np.array([r['snr'] for r in rr])
        ref_snr = None
        line = f'  n_users={U}:'
        for name in DETECTORS:
            y = np.log10(np.maximum([r[f'ser_{name}'] for r in rr], 1e-12))
            idx = np.where(y <= -2)[0]
            if idx.size == 0 or idx[0] == 0:
                line += f'  {name}=n/a'
                continue
            i = idx[0]
            snr_at = x[i - 1] + (-2 - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
            if name == 'genie':
                ref_snr = snr_at
                line += f'  {name}={snr_at:.1f}'
            else:
                line += f'  {name}={snr_at:.1f}' + (f'({snr_at - ref_snr:+.1f})' if ref_snr is not None else '')
        print(line)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(users), 3, figsize=(20, 5 * len(users)), squeeze=False)
    for k, U in enumerate(users):
        rr = [r for r in rows if r['n_users'] == U]
        x = [r['snr'] for r in rr]
        a = ax[k, 0]
        for name in ['dmrs', 'legacy']:
            a.plot(x, [r[f'nmse_{name}'] for r in rr], 'o-', color=COLORS[name], label=name, ms=3)
        a.plot(x, [r['nmse_noise_ref'] for r in rr], 'k--', label='sigma^2/Es (data noise)')
        a.set_title(f'n_users={U}: channel NMSE'); a.set_ylabel('dB')
        a = ax[k, 1]
        for name in DETECTORS:
            a.plot(x, [r[f'sinr_{name}'] for r in rr], 'o-', color=COLORS[name], label=name, ms=3)
        a.set_title('LMMSE post-eq SINR (Es/MSE)'); a.set_ylabel('dB')
        a = ax[k, 2]
        for name in DETECTORS:
            a.semilogy(x, np.maximum([r[f'ser_{name}'] for r in rr], 1e-6), 'o-', color=COLORS[name],
                       label=name, ms=3)
        a.set_title('uncoded SER'); a.set_ylim(1e-5, 1)
        for a in ax[k]:
            a.grid(True); a.set_xlabel('SNR (dB)'); a.legend(fontsize=8)
    fig.suptitle(f'DMRS CE diagnostic - TDL-C {delay_spread * 1e9:.0f}ns, #RE={args.num_res}, '
                 f'#an={args.n_ants}, {args.mod}QAM, {num_occ} DMRS occasions/group, {args.trials} trials, '
                 f'SIC iters={dmrs_pilots.DMRS_OCC_SIC_ITERS}')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, stem + '.png'), dpi=110)

    if per_re:
        fig2, ax2 = plt.subplots(1, len(users), figsize=(7 * len(users), 4.5), squeeze=False)
        for k, U in enumerate(users):
            for (u, name, snr, e) in per_re:
                if u == U:
                    ax2[0, k].plot(np.arange(args.num_res), e, 'o-', label=f'SNR={snr:.0f}')
            ax2[0, k].set_title(f'n_users={U}: NMSE per RE'); ax2[0, k].set_xlabel('RE')
            ax2[0, k].set_ylabel('dB'); ax2[0, k].grid(True); ax2[0, k].legend(fontsize=7)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, stem + '_perRE.png'), dpi=110)
    print(f'[dmrsdiag] wrote {csv_path} (+ .png, _perRE.png)')


if __name__ == '__main__':
    main()
