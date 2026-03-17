"""
Plot LLR histograms from HDF5 files saved by evaluate.py (save_llrs=True).

- Searches a directory for all *_llrs.h5 files whose filename contains _SNR=<snr>_
- Combines LLRs across all matching files (e.g. different seeds) and all blocks
- Plots one histogram per available detector (LMMSE, ESCNN, Sphere), shared x-axis
- Can also include an analytically-generated "ideal" 64-QAM LLR distribution

Usage (CLI):
    python plot_llr_histograms.py --snr 30
    python plot_llr_histograms.py --snr 30 --dir /path/to/Scratchpad --bins 50
    python plot_llr_histograms.py --snr 26 --ideal          # add ideal panel
    python plot_llr_histograms.py --snr 26 --ideal-only     # just the ideal panel

Usage (import):
    from python_code.utils.plot_llr_histograms import plot_llr_histograms_from_dir
    fig = plot_llr_histogra ms_from_dir(directory, snr=30)
"""

import argparse
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from scipy.special import logsumexp


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


# ---------------------------------------------------------------------------
# Ideal 64-QAM LLR generation
# ---------------------------------------------------------------------------

def _build_64qam_constellation():
    """
    Return (levels_norm, bits) for one PAM-8 dimension of Gray-coded 64-QAM.

    levels_norm : (8,)  normalised amplitudes (E[a^2] = 21 before norm, /sqrt(21))
    bits        : (8, 3) bit labels per level, MSB first
    """
    # Unnormalised 8-PAM levels and their standard Gray-code bit assignments
    # Reflected Gray code: index i -> gray(i), amplitude -7+2i
    levels_unnorm = np.array([-7., -5., -3., -1.,  1.,  3.,  5.,  7.])
    # Gray code for indices 0-7
    gray_bits = np.array([
        [0, 0, 0],   # 0 -> 000
        [0, 0, 1],   # 1 -> 001
        [0, 1, 1],   # 2 -> 011
        [0, 1, 0],   # 3 -> 010
        [1, 1, 0],   # 4 -> 110
        [1, 1, 1],   # 5 -> 111
        [1, 0, 1],   # 6 -> 101
        [1, 0, 0],   # 7 -> 100
    ], dtype=np.float32)
    # Normalise so E[a^2] = 1 per dimension (avg power of 8-PAM = 21)
    levels_norm = levels_unnorm / np.sqrt(21.0)
    return levels_norm, gray_bits


def generate_ideal_64qam_llrs(snr_db: float, n_symbols: int = 500_000,
                               rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate LLRs for ideal (genie) 64-QAM detection over AWGN.

    The SNR is Es/N0 (dB) where Es = 1 (normalised 64-QAM average symbol power).
    I and Q are independent Gray-coded 8-PAM dimensions; exact (soft) LLRs are
    computed per bit via the log-sum-exp formula.

    Returns a flat float32 array of all 6*n_symbols LLR values.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    levels, bits = _build_64qam_constellation()   # (8,), (8,3)
    M = len(levels)                                # 8

    # Noise variance per dimension: Es/N0 = snr_lin (total), split I+Q
    # Es=1 normalised, so noise_var_per_dim = 1 / (2 * snr_lin)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / (2.0 * snr_lin)

    # --- Simulate one PAM-8 dimension and compute exact LLRs ---
    def _pam8_llrs(n):
        # Random symbol indices for one dimension
        idx = rng.integers(0, M, size=n)
        tx = levels[idx]                          # (n,)
        rx = tx + rng.normal(0, np.sqrt(noise_var), size=n)  # (n,)

        # For each received sample compute LLR for each of 3 bits
        # LLR_k = log( sum_{a: bit_k=1} exp(-(rx-a)^2 / noise_var) )
        #               - log( sum_{a: bit_k=0} exp(-(rx-a)^2 / noise_var) )
        # shape (n, M): exponent for each (sample, constellation point)
        diff = rx[:, None] - levels[None, :]      # (n, M)
        log_metric = -(diff ** 2) / noise_var     # (n, M)

        llrs = np.empty((n, 3), dtype=np.float32)
        for k in range(3):
            mask1 = bits[:, k] == 1               # (M,) bool
            mask0 = ~mask1
            log_num = logsumexp(log_metric[:, mask1], axis=1)
            log_den = logsumexp(log_metric[:, mask0], axis=1)
            llrs[:, k] = (log_num - log_den).astype(np.float32)
        return llrs.flatten()

    llrs_i = _pam8_llrs(n_symbols)
    llrs_q = _pam8_llrs(n_symbols)
    return np.concatenate([llrs_i, llrs_q])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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


def plot_llr_comparison_with_ideal(data: dict, snr: int,
                                   bins: int = 50,
                                   n_files: int = 1,
                                   n_ideal_symbols: int = 500_000) -> plt.Figure:
    """
    Plot measured detector LLR histograms alongside an ideal 64-QAM reference.

    The ideal panel uses the same number of bins and a shared density-normalised
    y-axis so shapes are directly comparable.

    Parameters
    ----------
    data            : {detector_key: flat float32 array}
    snr             : SNR in dB (used for ideal generation and title)
    bins            : histogram bins
    n_files         : number of source files (title annotation)
    n_ideal_symbols : symbols to simulate for the ideal panel
    """
    available = [(key, label, color)
                 for key, label, color in _DETECTORS
                 if key in data]

    # Build panel list: measured detectors first, then ideal
    panels = available + [("_ideal_", "Ideal 64-QAM\n(AWGN, exact LLR)", "purple")]
    n = len(panels)

    print(f"Generating ideal 64-QAM LLRs at SNR={snr} dB …")
    ideal_llrs = generate_ideal_64qam_llrs(snr_db=snr, n_symbols=n_ideal_symbols)

    fig, axes = plt.subplots(n, 1, figsize=(6, 3.5 * n), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (key, label, color) in zip(axes, panels):
        vals = ideal_llrs if key == "_ideal_" else data[key]
        x_min = float(np.percentile(vals, 0.5))
        x_max = float(np.percentile(vals, 99.5))
        ax.hist(vals, bins=bins, range=(x_min, x_max),
                color=color, edgecolor='black', alpha=0.7, density=False)
        ax.set_xlabel('LLR value')
        ax.set_ylabel('#Values')
        ax.set_title(label)
        ax.set_xlim(x_min, x_max)
        ax.grid(True)

        # Annotate mean absolute LLR as a measure of "confidence"
        mean_abs = float(np.mean(np.abs(vals)))
        ax.axvline(mean_abs,  color='k', linestyle='--', linewidth=1, alpha=0.6)
        ax.axvline(-mean_abs, color='k', linestyle='--', linewidth=1, alpha=0.6,
                   label=f'±E[|LLR|]={mean_abs:.2f}')
        ax.legend(fontsize=8, loc='upper right')

    suffix = f" ({n_files} file{'s' if n_files > 1 else ''})" if n_files > 1 else ""
    fig.suptitle(f"LLR Histograms vs Ideal — SNR={snr} dB{suffix}", fontsize=11)
    fig.tight_layout()
    plt.show()
    return fig


def plot_ideal_only(snr: int, bins: int = 50,
                    n_symbols: int = 500_000) -> plt.Figure:
    """Generate and plot only the ideal 64-QAM LLR histogram."""
    print(f"Generating ideal 64-QAM LLRs at SNR={snr} dB …")
    ideal_llrs = generate_ideal_64qam_llrs(snr_db=snr, n_symbols=n_symbols)

    fig, ax = plt.subplots(figsize=(6, 4))
    x_min = float(np.percentile(ideal_llrs, 0.5))
    x_max = float(np.percentile(ideal_llrs, 99.5))
    ax.hist(ideal_llrs, bins=bins, range=(x_min, x_max),
            color='purple', edgecolor='black', alpha=0.7)
    mean_abs = float(np.mean(np.abs(ideal_llrs)))
    ax.axvline( mean_abs, color='k', linestyle='--', linewidth=1,
                label=f'±E[|LLR|]={mean_abs:.2f}')
    ax.axvline(-mean_abs, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('LLR value')
    ax.set_ylabel('#Values')
    ax.set_title(f'Ideal 64-QAM (AWGN, exact LLR) — SNR={snr} dB')
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    return fig


def plot_llr_histograms_from_dir(directory: str, snr: int,
                                 bins: int = 30,
                                 with_ideal: bool = False) -> plt.Figure:
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
    if with_ideal:
        return plot_llr_comparison_with_ideal(data, snr=snr, bins=bins,
                                              n_files=len(files))
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
    parser.add_argument("--ideal", action="store_true",
                        help="Add an ideal 64-QAM reference panel")
    parser.add_argument("--ideal-only", action="store_true",
                        help="Plot only the ideal 64-QAM reference (no HDF5 needed)")
    args = parser.parse_args()

    if args.ideal_only:
        plot_ideal_only(snr=args.snr, bins=args.bins)
    else:
        plot_llr_histograms_from_dir(args.dir, snr=args.snr, bins=args.bins,
                                     with_ideal=args.ideal)