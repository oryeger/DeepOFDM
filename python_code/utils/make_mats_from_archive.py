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
  python make_mats_from_archive.py <archive.gz> <extract_dir> <mat_out_dir>
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


def main(archive, extract_dir, mat_out_dir):
    # ---- 1. fresh extraction ----
    if os.path.isdir(extract_dir):
        shutil.rmtree(_long(extract_dir))
    os.makedirs(extract_dir)
    subprocess.run(["tar", "-xzf", archive, "-C", extract_dir], check=True)

    files = [os.path.basename(f) for f in glob.glob(os.path.join(extract_dir, "*.csv"))]
    print(f"[INFO] Extracted {len(files)} csv files from {archive}")

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
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
