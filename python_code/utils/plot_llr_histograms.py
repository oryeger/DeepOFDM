"""
Plot LLR histograms from HDF5 files saved by evaluate.py (save_llrs=True).

- Searches a directory for all *_llrs.h5 files whose filename contains _SNR=<snr>_
- Combines LLRs across all matching files (e.g. different seeds) and all blocks
- Plots one histogram per available detector (LMMSE, ESCNN, Sphere), shared x-axis

Usage (CLI):
    python plot_llr_histograms.py --snr 30
    python plot_llr_histograms.py --snr 30 --dir /path/to/Scratchpad --bins 50

Usage (import):
    from python_code.utils.plot_llr_histograms import plot_llr_histograms_from_dir
    fig = plot_llr_histograms_from_dir(directory, snr=30)
"""

import argparse
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path


# Detectors to look for, in display order: (h5_key, display_label, color)
_DETECTORS = [
    ("lmmse",       "LMMSE",  "red"),
    ("escnn_iter0", "ESCNN",  "blue"),
    ("sphere",      "Sphere", "green"),
]

# Default search directory: ../Scratchpad relative to this file
_DEFAULT_DIR = Path(__file__).resolve().parents[3] / "Scratchpad"


def _snr_from_filename(path: str):
    """Extract the SNR value from a filename like ..._SNR=30_llrs.h5."""
    m = re.search(r'_SNR=(-?\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def find_files_for_snr(directory: str, snr: int):
    """Return all *_llrs.h5 files in directory whose filename encodes the given SNR."""
    pattern = os.path.join(directory, "*_llrs.h5")
    all_files = glob.glob(pattern)
    matched = [f for f in all_files if _snr_from_filename(f) == snr]
    return sorted(matched)


def load_and_combine_llrs(files, snr: int):
    """
    Load LLRs for the given SNR from all provided files, concatenating across
    files and blocks. Returns a dict {detector_key: flat float32 array}.
    """
    accum = {key: [] for key, _, _ in _DETECTORS}

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            snr_key = f"snr_{snr}"
            if snr_key not in f:
                continue
            snr_grp = f[snr_key]
            for block_key in snr_grp:
                blk = snr_grp[block_key]
                for det_key, _, _ in _DETECTORS:
                    if det_key in blk:
                        accum[det_key].append(blk[det_key][()].astype(np.float32).flatten())

    return {k: np.concatenate(v) for k, v in accum.items() if v}


def plot_llr_histograms(data, snr: int,
                        bins: int = 30, n_files: int = 1) -> plt.Figure:
    """
    Plot LLR histograms from a pre-loaded data dict.

    Parameters
    ----------
    data : dict
        {detector_key: flat float32 array} as returned by load_and_combine_llrs.
    snr : int
        SNR value shown in the title.
    bins : int
        Number of histogram bins.
    n_files : int
        Number of source files (shown in title).
    """
    available = [(key, label, color)
                 for key, label, color in _DETECTORS
                 if key in data]
    if not available:
        raise ValueError("No recognisable detector data found.")

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3.5 * n), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (key, label, color) in zip(axes, available):
        x_min = float(np.percentile(data[key], 0.5))
        x_max = float(np.percentile(data[key], 99.5))
        ax.hist(data[key], bins=bins, range=(x_min, x_max),
                color=color, edgecolor='black', alpha=0.7)
        ax.set_xlabel('LLR value')
        ax.set_ylabel('#Values')
        ax.set_title(label)
        ax.set_xlim(x_min, x_max)
        ax.grid(True)

    suffix = f" ({n_files} file{'s' if n_files > 1 else ''})" if n_files > 1 else ""
    fig.suptitle(f"LLR Histograms — SNR={snr} dB{suffix}", fontsize=11)
    fig.tight_layout()
    plt.show()
    return fig


def plot_llr_histograms_from_dir(directory: str, snr: int,
                                 bins: int = 30) -> plt.Figure:
    """
    Find all matching HDF5 files in directory for the given SNR, combine their
    LLRs, and plot histograms.
    """
    files = find_files_for_snr(directory, snr)
    if not files:
        raise FileNotFoundError(
            f"No *_llrs.h5 files with SNR={snr} found in '{directory}'.\n"
            f"Files present: {glob.glob(os.path.join(directory, '*_llrs.h5'))}")
    print(f"Found {len(files)} file(s) for SNR={snr}:")
    for f in files:
        print(f"  {os.path.basename(f)}")
    data = load_and_combine_llrs(files, snr)
    return plot_llr_histograms(data, snr=snr, bins=bins, n_files=len(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot combined LLR histograms from HDF5 files.")
    parser.add_argument("--snr", type=int, required=True,
                        help="SNR point to plot (dB)")
    parser.add_argument("--dir", type=str, default=str(_DEFAULT_DIR),
                        help=f"Directory to search for *_llrs.h5 files "
                             f"(default: {_DEFAULT_DIR})")
    parser.add_argument("--bins", type=int, default=30,
                        help="Number of histogram bins (default: 30)")
    args = parser.parse_args()

    plot_llr_histograms_from_dir(args.dir, snr=args.snr, bins=args.bins)
