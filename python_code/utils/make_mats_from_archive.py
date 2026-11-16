"""
make_mats_from_archive.py - build the per-detector .mat files that
master_plot_mi_bler.m / plot_mi_set.m / plot_bler_set.m read, directly
from a .tar.gz of result CSVs.

The averaging itself is NOT re-implemented here: every .mat is produced by
plot_multiple_csvs.plot_csvs() from the DeepOFDM repo (the same code that
produced the existing mat_files). This script only
  1. extracts the archive (Windows bsdtar, handles the >260-char names),
  2. groups the CSVs by detector (aLMMSE / aSPHERE / aDEEPRX / aDEEPSIC),
  3. checks that all seeds of all detectors cover the same SNR set
     (refuses to average different seed sets per curve),
  4. calls plot_csvs(pattern, mat_out_dir, mat_name, seed_list) per detector.

Usage:
  python make_mats_from_archive.py <archive.gz> <extract_dir> <mat_out_dir> [--fill-missing]

--fill-missing: before the SNR check, create the missing (detector, seed, SNR)
CSVs from the same detector/seed's neighbouring SNRs. Observed CSVs are never
changed. Per column: equal neighbours -> that value (saturated region, e.g.
LMMSE BLER 1 / ESCNN 0); only one side observed (leading/trailing) -> copy
the nearest point; otherwise interpolate in SNR, log10 for BER/BLER when both
neighbours are > 0, linear otherwise (MI, or a 0 neighbour). Every filled
point is printed.
"""
import os
import re
import sys
import glob
import shutil
import subprocess

REPO_UTILS = os.path.dirname(os.path.abspath(__file__))   # plot_multiple_csvs.py lives next to this file
DETECTORS = {"LMMSE": "lmmse", "SPHERE": "sphere", "DEEPRX": "deeprx", "DEEPSIC": "deepsic"}
NAME_RE = re.compile(r"_a(LMMSE|SPHERE|DEEPRX|DEEPSIC)_.*_s=(\d+)_SNR=(-?\d+)(_bler|_mi)?\.csv$")


def _long(p):
    return "\\\\?\\" + os.path.abspath(p)


def _cell(x):
    s = str(x).replace("tensor(", "").replace(")", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _fill_missing(extract_dir, files):
    """Write CSVs for missing (det, seed, kind, SNR) points; never touches observed files."""
    import math
    import pandas as pd
    by = {}   # (det, seed, kind) -> {snr: filename}
    for f in files:
        m = NAME_RE.search(f)
        det, seed, snr, kind = m.group(1), int(m.group(2)), int(m.group(3)), (m.group(4) or "")
        by.setdefault((det, seed, kind), {}).setdefault(snr, f)
    all_snrs = sorted({s for d in by.values() for s in d})
    new_files = []
    for (det, seed, kind), snr_map in sorted(by.items()):
        have = sorted(snr_map)
        for s in all_snrs:
            if s in snr_map:
                continue
            lo = max([k for k in have if k < s], default=None)
            hi = min([k for k in have if k > s], default=None)
            df_lo = pd.read_csv(_long(os.path.join(extract_dir, snr_map[lo]))) if lo is not None else None
            df_hi = pd.read_csv(_long(os.path.join(extract_dir, snr_map[hi]))) if hi is not None else None
            ref = df_lo if df_lo is not None else df_hi
            out = ref.copy()
            for col in ref.columns:
                if col == "SNR_range":
                    out[col] = s
                    continue
                if df_lo is None or df_hi is None:
                    out[col] = _cell(ref[col].iloc[0])
                    continue
                a, b = _cell(df_lo[col].iloc[0]), _cell(df_hi[col].iloc[0])
                w = (s - lo) / (hi - lo)
                if a == b:
                    v = a
                elif kind != "_mi" and a > 0 and b > 0:
                    v = 10 ** ((1 - w) * math.log10(a) + w * math.log10(b))
                else:
                    v = (1 - w) * a + w * b
                out[col] = v
            src = snr_map[lo] if lo is not None else snr_map[hi]
            name = re.sub(r"_SNR=-?\d+", f"_SNR={s}", src)
            out.to_csv(_long(os.path.join(extract_dir, name)), index=False)
            new_files.append(name)
            how = "copied from SNR %s" % (lo if lo is not None else hi) if (lo is None or hi is None) else "from SNR %d and %d" % (lo, hi)
            print(f"[FILL] {det} seed {seed} SNR {s}{kind or '_ber'}: {how}")
    print(f"[INFO] Filled {len(new_files)} missing CSVs")
    return files + new_files


def main(archive, extract_dir, mat_out_dir, fill_missing=False):
    # ---- 1. fresh extraction ----
    if os.path.isdir(extract_dir):
        shutil.rmtree(_long(extract_dir))
    os.makedirs(extract_dir)
    subprocess.run(["tar", "-xzf", archive, "-C", extract_dir], check=True)

    files = [os.path.basename(f) for f in glob.glob(os.path.join(extract_dir, "*.csv"))]
    print(f"[INFO] Extracted {len(files)} csv files from {archive}")
    if fill_missing:
        files = _fill_missing(extract_dir, files)

    # ---- 2. group: det -> seed -> kind -> set(snr) ----
    table = {}
    for f in files:
        m = NAME_RE.search(f)
        if not m:
            raise RuntimeError(f"Unparsed file name: {f}")
        det, seed, snr, kind = m.group(1), int(m.group(2)), int(m.group(3)), (m.group(4) or "_ber")
        table.setdefault(det, {}).setdefault(seed, {}).setdefault(kind, set()).add(snr)

    # One CFO per archive (cfo=0 and cfo=0.0 count as the same value)
    cfos = {float(m.group(1)) for m in (re.search(r"_cfo=([\d.]+)_", f) for f in files) if m}
    if len(cfos) > 1:
        raise RuntimeError(f"Archive mixes several CFO values {sorted(cfos)} - one figure needs one CFO")
    print(f"[INFO] CFO = {sorted(cfos)}")

    missing = [d for d in DETECTORS if d not in table]
    if missing:
        raise RuntimeError(f"Archive has no results for detector(s): {missing}")

    # ---- 3. identical seed set and identical SNR set everywhere ----
    ref_seeds, ref_snrs = None, None
    for det, by_seed in table.items():
        seeds = sorted(by_seed)
        if ref_seeds is None:
            ref_seeds = seeds
        elif seeds != ref_seeds:
            raise RuntimeError(f"{det}: seeds {seeds} differ from {ref_seeds}")
        for seed, by_kind in by_seed.items():
            for kind in ("_ber", "_bler", "_mi"):
                snrs = sorted(by_kind.get(kind, set()))
                if ref_snrs is None:
                    ref_snrs = snrs
                elif snrs != ref_snrs:
                    raise RuntimeError(f"{det} seed {seed} {kind}: SNRs {snrs} differ from {ref_snrs}")
    print(f"[INFO] All detectors: seeds {ref_seeds}, SNR {ref_snrs[0]}..{ref_snrs[-1]} ({len(ref_snrs)} pts)")

    # Duplicates (same det/seed/SNR from two runs): plot_csvs keeps the first in sorted order.
    for det in DETECTORS:
        keys = {}
        for f in sorted(files):
            m = NAME_RE.search(f)
            if m.group(1) == det and not m.group(4):
                keys.setdefault((int(m.group(2)), int(m.group(3))), []).append(f[:13])
        for (seed, snr), stamps in keys.items():
            if len(stamps) > 1:
                print(f"[WARN] {det} seed {seed} SNR {snr}: {len(stamps)} runs {stamps} -> using {stamps[0]}")

    # ---- 4. reuse plot_multiple_csvs ----
    sys.path.insert(0, REPO_UTILS)
    import plot_multiple_csvs as pmc
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None          # plot_csvs calls plt.show() before savemat
    pmc.CSV_DIR = extract_dir

    os.makedirs(mat_out_dir, exist_ok=True)
    for det, mat_name in DETECTORS.items():
        pmc.plot_csvs(f"*_a{det}_*.csv", mat_out_dir=mat_out_dir, mat_name=mat_name, seed_list=ref_seeds)
        plt.close("all")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fill-missing"]
    if len(args) != 3:
        sys.exit(__doc__)
    main(args[0], args[1], args[2], fill_missing="--fill-missing" in sys.argv[1:])
