"""Offline analysis of ekf.py's --save_loss_plot_snr _diag.h5 dump.

Run as a module so the python_code package imports resolve (needs Sionna/TensorFlow for
the LDPC decode - same env as ekf.py itself, so run this on the cluster, not a plain
Windows dev box):

    python -m python_code.utils.analyze_diag_h5 <path-to-..._diag.h5>

--mcs/--num-res only need overriding if the run used something other than ekf.py's
defaults (conf.mcs / conf.num_res) - get_mcs(conf.mcs) and num_res both feed directly
into the LDPC (k, n) sizing below, so a wrong value desyncs the decode.

Why this exists: the H5 (see ekf.py's run_group()/main()) carries the full LMMSE
LLR-generation chain per group (H_data, equalized_lmmse, llrs_mat_lmmse, tx_bits, ...)
but NOT each slot's pass/fail BLER outcome - that only lives in the run's _bler.csv,
and even there only at (cdi, user) granularity, one row per *group*, not per slot within
a multi-slot group. Rather than requiring the CSV or a third cluster run to add the flag
directly to the H5, this script re-derives crc pass/fail itself, by replaying the *exact*
LDPC decode run_group() does - same LDPC5GCodec/CRC5GCodec construction, same real_bit_idx
selection (a no-op here since ekf.py always calls it with pilot_data_ratio=1.0), same
per-slot stream windowing - so it works standalone from just the H5, and at the same
(group, slot, user) granularity the original [llr-diag] log lines used.

It then buckets (group, slot, user) rows into crc=FAIL vs crc=OK and compares the spatial
distribution of wrong bits between the two: the burst-vs-diffuse hypothesis from the
[llr-diag] investigation (whether FAIL rows have their wrong bits piled onto a handful of
REs - a trapping-set-style signature belief propagation struggles with - vs. OK rows with
a similar total error count spread thinly across most REs, which ordinary BP handles
fine). Earlier magnitude-only stats (wrong_term_share, single worst |LLR|) didn't cleanly
separate FAIL from OK at matched wrong_frac; n_wrong_res/top_re_share test a different,
count-based notion of concentration instead.

It also checks channel-estimation error against the same per-RE wrong-bit counts. Note
noise_var (the scalar fed into ekf.py's LMMSE math, diag_h5.attrs["noise_var"]) is NOT
computed per group/RE - it's one theoretical constant for the whole run
(10**(-0.1*conf.snr)*CONSTELLATION_FACTOR[mod_data], set once in main() before any group
runs - see ekf.py's lmmse_equalize_with_H, which takes it as a plain float argument with
no re-estimation), so it's identical across every row here and can't explain FAIL-vs-OK
differences. What DOES vary per RE/group and feeds the same postEqSINR/bias formula is the
DMRS channel estimate H_data itself - h_abs_per_re/h_angle_per_re (the noisy estimate
LMMSE actually equalizes with) vs h_abs_ground_truth_per_re/h_angle_ground_truth_per_re
(an independent, DMRS-machinery-bypassing ground truth) are both already in the H5, so
channel_err_* below reuses them rather than needing a rerun.
"""
import argparse

import h5py
import numpy as np

from python_code.coding.crc_wrapper import CRC5GCodec
from python_code.coding.ldpc_wrapper import LDPC5GCodec
from python_code.coding.mcs_table import get_mcs
from python_code.evaluate import crc_fail_mask
from python_code.utils.probs_utils import relevant_indices

# See ekf.py's constants.DMRS_NUM_PAYLOAD_SYMB. Kept as a literal here (rather than
# importing python_code.utils.constants) so this script needs no ekf.py-adjacent module
# beyond the coding/eval pieces actually used for the decode replay.
_DMRS_NUM_PAYLOAD_SYMB = 12


def _build_codec(mcs: int, num_res: int):
    """Mirrors ekf.py main()'s 'Unified codeword sizing' block exactly. Must match
    bit-for-bit or the LDPC decode below is silently wrong (different (k, n) = a
    different code, so crc pass/fail read back would be meaningless)."""
    qm, code_rate = get_mcs(mcs)
    qm = int(qm)
    ldpc_n = int(num_res * _DMRS_NUM_PAYLOAD_SYMB * qm)
    ldpc_k = int(ldpc_n * code_rate)
    crc_length = 24 if ldpc_k > 3824 else 16
    codec = LDPC5GCodec(k=ldpc_k + crc_length, n=ldpc_n)
    crc = CRC5GCodec(crc_length)
    return codec, crc, qm, ldpc_n


def analyze(h5_path: str, mcs: int, num_res: int) -> tuple[list[dict], dict]:
    codec, crc, qm, ldpc_n = _build_codec(mcs, num_res)
    real_bit_idx = relevant_indices(qm, 1.0)  # identity here - see module docstring

    rows = []
    with h5py.File(h5_path, "r") as f:
        n_users = int(f.attrs["n_users"])
        meta = {
            "noise_var": float(f.attrs.get("noise_var", float("nan"))),
            "override_noise_var": bool(f.attrs.get("override_noise_var", True)),
        }
        cdi_keys = sorted((k for k in f.keys() if k.startswith("cdi_")),
                           key=lambda k: int(k.split("_")[1]))
        for key in cdi_keys:
            grp = f[key]
            cdi = int(key.split("_")[1])
            # tx_bits: (symbol-major bit-in-symbol, user, RE) - see ekf.py's dims attr.
            tx_bits = grp["tx_bits"][()].astype(np.int64)
            # llrs_mat_lmmse: (symbol, qm*n_users, RE) after the H5 write's squeeze(-1).
            llrs = grp["llrs_mat_lmmse"][()].astype(np.float64)
            num_symbols, _num_bits_total, num_res_here = llrs.shape

            # Channel-estimation error per RE (see module docstring): H_est is what LMMSE
            # actually equalized with; H_gt is the independent ground truth. One value per
            # (RE, ant, user) for the whole group - block fading means this doesn't change
            # within the group, so it applies identically to every slot below.
            noise_var_est = float(grp.attrs.get("noise_var_est", float("nan")))
            lmmse_noise_var = float(grp.attrs.get("lmmse_noise_var", float("nan")))

            h_abs_est = grp["h_abs_per_re"][()].astype(np.float64)
            h_angle_est = grp["h_angle_per_re"][()].astype(np.float64)
            h_abs_gt = grp["h_abs_ground_truth_per_re"][()].astype(np.float64)
            h_angle_gt = grp["h_angle_ground_truth_per_re"][()].astype(np.float64)
            H_est = h_abs_est * np.exp(1j * h_angle_est)          # (RE, ant, user)
            H_gt = h_abs_gt * np.exp(1j * h_angle_gt)
            chan_err = np.abs(H_est - H_gt).mean(axis=1)           # (RE, user) - mean over ant
            chan_rel_err = chan_err / (np.abs(H_gt).mean(axis=1) + 1e-12)

            group_size_slots = num_symbols // _DMRS_NUM_PAYLOAD_SYMB
            symbols_per_slot = _DMRS_NUM_PAYLOAD_SYMB

            lmmse_stream = np.zeros((n_users, group_size_slots * ldpc_n))
            lmmse_llr_scored_by_user = []
            tx_user_by_user = []
            for user in range(n_users):
                tx_user = tx_bits[:, user, :].reshape(num_symbols, qm, num_res_here)
                lmmse_llr_full = llrs[:, user * qm:(user + 1) * qm, :]
                lmmse_llr_scored = lmmse_llr_full[:, real_bit_idx, :]
                lmmse_stream[user] = lmmse_llr_scored.reshape(-1)
                lmmse_llr_scored_by_user.append(lmmse_llr_scored)
                tx_user_by_user.append(tx_user)

            for slot in range(group_size_slots):
                win = slice(slot * ldpc_n, (slot + 1) * ldpc_n)
                decoded = codec.decode(lmmse_stream[:, win])
                crc_out = crc.decode(decoded)
                fail_per_user = crc_fail_mask(decoded, crc_out)

                sym_win = slice(slot * symbols_per_slot, (slot + 1) * symbols_per_slot)
                for user in range(n_users):
                    llr_s = lmmse_llr_scored_by_user[user][sym_win]   # (12, qm, RE)
                    tx_s = tx_user_by_user[user][sym_win]
                    hard = (llr_s > 0).astype(np.int64)
                    wrong = hard != tx_s
                    wrong_frac = float(wrong.mean())

                    wrong_count_per_re = wrong.sum(axis=(0, 1))
                    total_wrong = int(wrong_count_per_re.sum())
                    n_wrong_res = int((wrong_count_per_re > 0).sum())
                    rel_err_re = chan_rel_err[:, user]
                    if total_wrong > 0:
                        top_re = int(wrong_count_per_re.argmax())
                        top_re_share = float(wrong_count_per_re[top_re] / total_wrong)
                        abs_l = np.abs(llr_s)
                        wrong_l_max = float(abs_l[wrong].max())
                        wrong_l_mean = float(abs_l[wrong].mean())
                        # Does the RE with the most wrong bits also have poor channel
                        # estimation, and across all REs in this row, do wrong-bit counts
                        # track estimation error at all? Pearson r needs nonzero variance on
                        # both sides (skipped - nan - when wrong bits are perfectly uniform
                        # across REs, e.g. exactly 1 wrong bit/RE everywhere).
                        chan_err_at_top_re = float(rel_err_re[top_re])
                        if wrong_count_per_re.std() > 0 and rel_err_re.std() > 0:
                            chan_wrong_corr = float(np.corrcoef(rel_err_re, wrong_count_per_re)[0, 1])
                        else:
                            chan_wrong_corr = float("nan")
                    else:
                        top_re, top_re_share, wrong_l_max, wrong_l_mean = -1, 0.0, 0.0, 0.0
                        chan_err_at_top_re, chan_wrong_corr = 0.0, float("nan")

                    rows.append(dict(
                        cdi=cdi, slot=slot, user=user, crc_fail=bool(fail_per_user[user]),
                        wrong_frac=wrong_frac, total_wrong=total_wrong,
                        n_wrong_res=n_wrong_res, num_res=num_res_here,
                        top_re=top_re, top_re_share=top_re_share,
                        wrong_l_max=wrong_l_max, wrong_l_mean=wrong_l_mean,
                        chan_err_mean=float(rel_err_re.mean()), chan_err_max=float(rel_err_re.max()),
                        chan_err_at_top_re=chan_err_at_top_re, chan_wrong_corr=chan_wrong_corr,
                        noise_var_est=noise_var_est, lmmse_noise_var=lmmse_noise_var,
                    ))
    return rows, meta


def _summarize(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {label}: (none)")
        return
    fields = ("wrong_frac", "total_wrong", "n_wrong_res", "top_re_share", "wrong_l_max", "wrong_l_mean",
              "chan_err_mean", "chan_err_max", "chan_err_at_top_re")
    for field in fields:
        vals = [r[field] for r in rows]
        print(f"  {label:5s} {field:18s} mean={np.mean(vals):8.4f} median={np.median(vals):8.4f} "
              f"min={np.min(vals):8.4f} max={np.max(vals):8.4f}  (n={len(vals)})")
    corrs = [r["chan_wrong_corr"] for r in rows if not np.isnan(r["chan_wrong_corr"])]
    if corrs:
        print(f"  {label:5s} chan_wrong_corr    mean={np.mean(corrs):8.4f} median={np.median(corrs):8.4f} "
              f"min={np.min(corrs):8.4f} max={np.max(corrs):8.4f}  (n={len(corrs)}, "
              f"{len(rows) - len(corrs)} skipped as undefined)")
    # Separate (scientific-notation) block: noise_var_est/lmmse_noise_var are ~1e-3 scale, would
    # print as 0.0000 under the fixed-point format above. This is THE key check for the
    # postEqSINR-blowup mechanism: does the FAIL bucket's noise_var_est distribution sit
    # noticeably below OK's (and below the file-level theoretical noise_var printed in main()),
    # or is the earlier per-row-detail impression (FAIL/worst rows showing low noise_var_est)
    # just selection bias from only ever printing the worst rows?
    for field in ("noise_var_est", "lmmse_noise_var"):
        vals = [r[field] for r in rows]
        print(f"  {label:5s} {field:18s} mean={np.mean(vals):.3e} median={np.median(vals):.3e} "
              f"min={np.min(vals):.3e} max={np.max(vals):.3e}  (n={len(vals)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5_path", help="path to a ..._diag.h5 file written by ekf.py")
    ap.add_argument("--mcs", type=int, default=2, help="conf.mcs the run used (default 2, QPSK)")
    ap.add_argument("--num-res", type=int, default=96, help="conf.num_res the run used (default 96)")
    args = ap.parse_args()

    rows, meta = analyze(args.h5_path, args.mcs, args.num_res)
    fails = [r for r in rows if r["crc_fail"]]
    oks = [r for r in rows if not r["crc_fail"]]

    print(f"{args.h5_path}")
    print(f"file-level theoretical noise_var={meta['noise_var']:.3e} "
          f"override_noise_var={meta['override_noise_var']} "
          f"(False -> LMMSE actually used each group's own noise_var_est, not this constant)")
    print(f"{len(rows)} (group, slot, user) rows total: {len(fails)} FAIL, {len(oks)} OK "
          f"(BLER={len(fails) / max(len(rows), 1):.4%})\n")
    print("Aggregate wrong-bit-concentration stats, FAIL vs OK:")
    _summarize("FAIL", fails)
    _summarize("OK", oks)

    if fails:
        print("\nPer-FAIL-row detail:")
        for r in fails:
            print(f"  cdi={r['cdi']:5d} slot={r['slot']} user={r['user']} crc=FAIL "
                  f"wrong_frac={r['wrong_frac']:.4f} "
                  f"n_wrong_res={r['n_wrong_res']:3d}/{r['num_res']} "
                  f"top_re_share={r['top_re_share']:.3f}(re={r['top_re']}) "
                  f"wrong|L|_mean={r['wrong_l_mean']:.1f} wrong|L|_max={r['wrong_l_max']:.1f} "
                  f"chan_err(mean/max/at_top_re)={r['chan_err_mean']:.3f}/{r['chan_err_max']:.3f}/"
                  f"{r['chan_err_at_top_re']:.3f} chan_wrong_corr={r['chan_wrong_corr']:.3f} "
                  f"noise_var_est={r['noise_var_est']:.3e} lmmse_noise_var={r['lmmse_noise_var']:.3e}")

    # Top-N by wrong_frac regardless of crc_fail - lines up the worst-surviving OK rows
    # against the FAIL rows directly, since bucket-level means/medians can hide a close call
    # (e.g. an OK row with MORE total wrong bits than either FAIL row - see module docstring's
    # "near-miss" discussion).
    top_n = 15
    worst_rows = sorted(rows, key=lambda r: r["wrong_frac"], reverse=True)[:top_n]
    print(f"\nTop {top_n} rows by wrong_frac (either crc outcome):")
    for r in worst_rows:
        print(f"  cdi={r['cdi']:5d} slot={r['slot']} user={r['user']} "
              f"crc={'FAIL' if r['crc_fail'] else 'OK  '} "
              f"wrong_frac={r['wrong_frac']:.4f} total_wrong={r['total_wrong']:4d} "
              f"n_wrong_res={r['n_wrong_res']:3d}/{r['num_res']} "
              f"top_re_share={r['top_re_share']:.3f}(re={r['top_re']}) "
              f"wrong|L|_mean={r['wrong_l_mean']:.1f} wrong|L|_max={r['wrong_l_max']:.1f} "
              f"chan_err(mean/max/at_top_re)={r['chan_err_mean']:.3f}/{r['chan_err_max']:.3f}/"
              f"{r['chan_err_at_top_re']:.3f} chan_wrong_corr={r['chan_wrong_corr']:.3f} "
              f"noise_var_est={r['noise_var_est']:.3e} lmmse_noise_var={r['lmmse_noise_var']:.3e}")


if __name__ == "__main__":
    main()
