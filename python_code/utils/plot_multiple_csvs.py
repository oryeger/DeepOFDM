import os
import sys
import platform
import subprocess
import glob
import pandas as pd
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'none'

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import re
from scipy.interpolate import interp1d
import numpy as np
from collections import defaultdict
from scipy.io import savemat

# 🔧 Configuration
CSV_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Scratchpad"))
seeds = [123, 17, 41, 58, 1011, 1809, 3008, 1806, 912, 1505, 1807, 1109]
MIN_SNR = -np.inf
MAX_SNR = np.inf

# ---- Missing / cleanup handling configuration ----
CLEANUP_ENABLED = False              # Master switch: False = plot raw averaged data, no cleanup at all

INTERPOLATE_MISSING_PER_SEED = True  and CLEANUP_ENABLED
MAX_INTERP_GAP = 3                  if CLEANUP_ENABLED else 0   # Fill only interior gaps of up to this many consecutive missing SNRs
PRINT_INTERP_SUMMARY = True  and CLEANUP_ENABLED
USE_FULL_INTEGER_SNR_GRID = False   # False = union of observed SNRs
REMOVE_ISOLATED_NONMONO_POINTS = True  and CLEANUP_ENABLED
MAX_BAD_POINTS_PER_CURVE = 1        if CLEANUP_ENABLED else 0   # Max non-monotonic points to remove per curve
PRINT_NONMONO_SUMMARY = True  and CLEANUP_ENABLED

# ---- Helper: build pretty title from a filename (your exact logic) ----
def build_cleaned_title_from_filename(original_name: str) -> str:
    cleaned_name = re.sub(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_", "", original_name)
    cleaned_name = re.sub(".csv", "", cleaned_name)
    cleaned_name = re.sub("Clip=100%", "", cleaned_name)
    cleaned_name = re.sub(r"_SNR=-?\d+", "", cleaned_name)
    cleaned_name = re.sub("_scs", "scs", cleaned_name)
    cleaned_name = re.sub("cfo_in_Rx", "cfo", cleaned_name)
    cleaned_name = re.sub(r"seed=\d+|_s=\d+", "", cleaned_name)
    cleaned_name = re.sub("_", ", ", cleaned_name)
    cleaned_name = re.sub("twostage, ", "", cleaned_name)
    cleaned_name = re.sub(", , , three, layers=123", ", three layes", cleaned_name)
    cleaned_name = cleaned_name.rstrip(", ")
    cleaned_name = "\n".join([cleaned_name[i:i+80] for i in range(0, len(cleaned_name), 80)])
    return cleaned_name


def _to_float_cell(x):
    """
    Robust conversion for cells like:
      tensor(0.0011), 'tensor(0.0011)', 0, 0.0, '0.0'
    """
    s = str(x)
    s = s.replace("tensor(", "").replace(")", "").strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _get_first_present(df, candidates):
    """Return the first column name that exists in df.columns, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_interp_x_to_y(x, y, x_target):
    """
    Interpolate y(x) safely after sorting x and dropping duplicates.
    Used for finding SNR@targetBER, i.e., y=snr and x=ber.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    x_unique, idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[idx]

    if x_unique.size < 2:
        return np.nan

    f = interp1d(x_unique, y_unique, fill_value="extrapolate", bounds_error=False)
    return float(np.round(f(x_target), 1))


def _build_snr_grid(observed_snrs):
    """
    Build the SNR grid over which curves will be averaged.
    Recommended default: union of observed SNRs.
    """
    observed_snrs = sorted(set(int(s) for s in observed_snrs if MIN_SNR <= s <= MAX_SNR))
    if not observed_snrs:
        return []

    if USE_FULL_INTEGER_SNR_GRID:
        return list(range(min(observed_snrs), max(observed_snrs) + 1))
    return observed_snrs


def _remove_isolated_nonmonotonic_points(snrs, values, max_bad_points=1):
    """
    Remove isolated interior points that violate monotonic non-increasing behavior.
    Such points are replaced by NaN so they can later be interpolated.

    A point y[i] is flagged if:
      - y[i-1], y[i], y[i+1] are finite
      - y[i] > y[i-1]  (increase relative to previous point)
      - and y[i] > y[i+1] (local bump / spike)

    This is conservative and avoids removing a whole sloped region.

    Parameters
    ----------
    snrs : list/array
        Sorted SNRs
    values : list/array
        Curve values with possible NaNs
    max_bad_points : int
        Maximum number of suspicious points to remove

    Returns
    -------
    cleaned_values : np.ndarray
    removed_positions : list[int]
    """
    y = np.asarray(values, dtype=float).copy()
    n = len(y)
    removed_positions = []

    if n < 3:
        return y, removed_positions

    candidates = []
    for i in range(1, n - 1):
        if not (np.isfinite(y[i - 1]) and np.isfinite(y[i]) and np.isfinite(y[i + 1])):
            continue

        # local upward spike for a non-increasing curve
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            severity = y[i] - max(y[i - 1], y[i + 1])
            candidates.append((severity, i))

    # remove only the strongest few
    candidates.sort(reverse=True, key=lambda t: t[0])

    for _, i in candidates[:max_bad_points]:
        y[i] = np.nan
        removed_positions.append(i)

    return y, removed_positions


def _fill_missing_curve(snrs, values, use_log_interp=True, max_gap=3):
    """
    Fill small interior missing gaps in a single seed curve.

    Parameters
    ----------
    snrs : list/array of SNRs (must be sorted)
    values : list/array of y-values, with np.nan for missing points
    use_log_interp : bool
        True for BER/BLER-like positive quantities
        False for MI or linear-scale quantities
    max_gap : int
        Maximum consecutive missing points to fill

    Returns
    -------
    filled_values : np.ndarray
    filled_positions : list[int]
        indices in the array that were interpolated
    """
    x = np.asarray(snrs, dtype=float)
    y = np.asarray(values, dtype=float).copy()

    n = len(y)
    filled_positions = []

    if n == 0:
        return y, filled_positions

    finite = np.isfinite(y)
    if finite.sum() < 2:
        return y, filled_positions

    i = 0
    while i < n:
        if np.isfinite(y[i]):
            i += 1
            continue

        start = i
        while i < n and not np.isfinite(y[i]):
            i += 1
        end = i - 1

        gap_len = end - start + 1
        left = start - 1
        right = end + 1

        # Only fill interior gaps bracketed by finite points
        if left < 0 or right >= n:
            continue
        if not np.isfinite(y[left]) or not np.isfinite(y[right]):
            continue
        if gap_len > max_gap:
            continue

        x_left, x_right = x[left], x[right]
        y_left, y_right = y[left], y[right]

        if use_log_interp:
            if y_left <= 0 or y_right <= 0:
                continue
            ly_left = np.log10(y_left)
            ly_right = np.log10(y_right)

            for k in range(start, end + 1):
                alpha = (x[k] - x_left) / (x_right - x_left)
                y[k] = 10 ** ((1 - alpha) * ly_left + alpha * ly_right)
                filled_positions.append(k)
        else:
            for k in range(start, end + 1):
                alpha = (x[k] - x_left) / (x_right - x_left)
                y[k] = (1 - alpha) * y_left + alpha * y_right
                filled_positions.append(k)

    return y, filled_positions


def plot_csvs(filter_pattern=None, plot_all_iters=False):
    """
    Plot BER/BLER/MI from CSV files in CSV_DIR.

    Args:
        filter_pattern: Optional glob pattern string to select files.
        plot_all_iters: If True, plot all iterations (ESCNN1/2/3 etc.).
    """
    plot_all_escnn = plot_all_iters

    if filter_pattern is not None:
        all_files = sorted(glob.glob(os.path.join(CSV_DIR, filter_pattern)))
    else:
        all_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))

    print(f"[INFO] Found {len(all_files)} files matching pattern.")

    pat_upper = (filter_pattern or "").upper()
    show_sphere = "SPHERE" in pat_upper and "PRIME_1" not in pat_upper

    mi_files_exist = any(f.endswith("_mi.csv") for f in all_files)

    if mi_files_exist:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
        subplot_index = {"BER": 2, "BLER": 0, "MI": 1}
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.5))
        subplot_index = {"BER": 1, "BLER": 0}

    used_seeds_overall = set()
    title_source_file = None

    plot_types = ["BLER", "BER"]
    if mi_files_exist:
        plot_types = ["BLER", "MI", "BER"]

    bler_snrs = None
    bler_avg_curves = None
    ber_snrs = None
    ber_avg_curves = None
    mi_snrs = None
    mi_avg_curves = None

    for plot_type in plot_types:
        ax = axes[subplot_index[plot_type]]

        if plot_type == "BER":
            search_pattern = r"SNR=(-?\d+)"
            target_y = 0.01
            ylabel_cur = "BER"
            use_log_interp_for_missing = True
        elif plot_type == "BLER":
            search_pattern = r"_SNR=(-?\d+)_bler\.csv$"
            target_y = 0.1
            ylabel_cur = "BLER"
            use_log_interp_for_missing = True
        else:  # MI
            search_pattern = r"_SNR=(-?\d+)_mi\.csv$"
            target_y = None
            ylabel_cur = "MI"
            use_log_interp_for_missing = False

        # Store values per seed per SNR per key
        seed_snr_dict = defaultdict(lambda: defaultdict(dict))
        observed_snrs = set()

        for seed in seeds:
            seed_files = sorted(
                f for f in all_files
                if (f"seed={seed}" in os.path.basename(f) or f"_s={seed}_" in os.path.basename(f))
                and "_SNR=" in os.path.basename(f)
            )
            if not seed_files:
                continue

            seen_snr = set()
            unique_files = []

            for file in seed_files:
                match = re.search(search_pattern, file)
                if match:
                    snr = int(match.group(1))
                    if snr not in seen_snr:
                        seen_snr.add(snr)
                        unique_files.append(file)

            if not unique_files:
                continue

            used_seeds_overall.add(seed)
            if title_source_file is None:
                title_source_file = unique_files[0]

            for file in unique_files:
                match = re.search(search_pattern, file)
                if not match:
                    continue

                snr = int(match.group(1))
                if not (MIN_SNR <= snr <= MAX_SNR):
                    continue

                observed_snrs.add(snr)
                safe_file = "\\\\?\\" + os.path.abspath(file) if platform.system() == "Windows" else file
                df = pd.read_csv(safe_file)

                # ---- ESCNN ----
                if "total_ber_1" in df.columns:
                    seed_snr_dict[seed][snr]["ber_1"] = _to_float_cell(df["total_ber_1"].iloc[0])

                if "total_ber_2" in df.columns:
                    seed_snr_dict[seed][snr]["ber_2"] = _to_float_cell(df["total_ber_2"].iloc[0])
                elif "total_ber_1" in df.columns:
                    seed_snr_dict[seed][snr]["ber_2"] = _to_float_cell(df["total_ber_1"].iloc[0])

                if "total_ber_3" in df.columns:
                    seed_snr_dict[seed][snr]["ber_3"] = _to_float_cell(df["total_ber_3"].iloc[0])
                elif "total_ber_1" in df.columns:
                    seed_snr_dict[seed][snr]["ber_3"] = _to_float_cell(df["total_ber_1"].iloc[0])

                # ---- JointLLR ----
                joint_col = _get_first_present(df, ["total_ber_jointllr_1", "total_ber_jointllr"])
                if joint_col is not None:
                    seed_snr_dict[seed][snr]["ber_jointllr_1"] = _to_float_cell(df[joint_col].iloc[0])

                # ---- MHSA ----
                if any(col.startswith("total_ber_mhsa") for col in df.columns):
                    if "total_ber_mhsa" in df.columns and not any(col.startswith("total_ber_mhsa_") for col in df.columns):
                        val = _to_float_cell(df["total_ber_mhsa"].iloc[0])
                        for key in ["ber_mhsa_1", "ber_mhsa_2", "ber_mhsa_3"]:
                            seed_snr_dict[seed][snr][key] = val
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_mhsa_{k}"
                            if colname in df.columns:
                                seed_snr_dict[seed][snr][f"ber_mhsa_{k}"] = _to_float_cell(df[colname].iloc[0])
                            elif "total_ber_mhsa" in df.columns:
                                seed_snr_dict[seed][snr][f"ber_mhsa_{k}"] = _to_float_cell(df["total_ber_mhsa"].iloc[0])

                # ---- TDFDCNN (tfdfcnn) ----
                if any(col.startswith("total_ber_tfdfcnn") for col in df.columns):
                    if "total_ber_tfdfcnn" in df.columns and not any(col.startswith("total_ber_tfdfcnn_") for col in df.columns):
                        val = _to_float_cell(df["total_ber_tfdfcnn"].iloc[0])
                        for key in ["ber_tdfdcnn_1", "ber_tdfdcnn_2", "ber_tdfdcnn_3"]:
                            seed_snr_dict[seed][snr][key] = val
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_tfdfcnn_{k}"
                            if colname in df.columns:
                                seed_snr_dict[seed][snr][f"ber_tdfdcnn_{k}"] = _to_float_cell(df[colname].iloc[0])
                            elif "total_ber_tdfdcnn" in df.columns:
                                seed_snr_dict[seed][snr][f"ber_tdfdcnn_{k}"] = _to_float_cell(df["total_ber_tfdfcnn"].iloc[0])

                # ---- TDFDCNN (tdfdcnn) ----
                if any(col.startswith("total_ber_tdfdcnn") for col in df.columns):
                    if "total_ber_tdfdcnn" in df.columns and not any(col.startswith("total_ber_tdfdcnn_") for col in df.columns):
                        val = _to_float_cell(df["total_ber_tdfdcnn"].iloc[0])
                        for key in ["ber_tdfdcnn_1", "ber_tdfdcnn_2", "ber_tdfdcnn_3"]:
                            seed_snr_dict[seed][snr][key] = val
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_tdfdcnn_{k}"
                            if colname in df.columns:
                                seed_snr_dict[seed][snr][f"ber_tdfdcnn_{k}"] = _to_float_cell(df[colname].iloc[0])
                            elif "total_ber_tdfdcnn" in df.columns:
                                seed_snr_dict[seed][snr][f"ber_tdfdcnn_{k}"] = _to_float_cell(df["total_ber_tdfdcnn"].iloc[0])

                # ---- DeepSIC ----
                if any(col.startswith("total_ber_deepsic") for col in df.columns):
                    if "total_ber_deepsic" in df.columns:
                        for k in [1, 2, 3]:
                            seed_snr_dict[seed][snr][f"ber_deepsic_{k}"] = _to_float_cell(df["total_ber_deepsic"].iloc[0])
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_deepsic_{k}"
                            if colname in df.columns:
                                seed_snr_dict[seed][snr][f"ber_deepsic_{k}"] = _to_float_cell(df[colname].iloc[0])
                            elif "total_ber_deepsic_1" in df.columns:
                                seed_snr_dict[seed][snr][f"ber_deepsic_{k}"] = _to_float_cell(df["total_ber_deepsic_1"].iloc[0])

                # ---- DeepSIC-MB ----
                if any(col.startswith("total_ber_deepsicmb") for col in df.columns):
                    if "total_ber_deepsicmb" in df.columns:
                        for k in [1, 2, 3]:
                            seed_snr_dict[seed][snr][f"ber_deepsicmb_{k}"] = _to_float_cell(df["total_ber_deepsicmb"].iloc[0])
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_deepsicmb_{k}"
                            if colname in df.columns:
                                seed_snr_dict[seed][snr][f"ber_deepsicmb_{k}"] = _to_float_cell(df[colname].iloc[0])
                            elif "total_ber_deepsicmb_1" in df.columns:
                                seed_snr_dict[seed][snr][f"ber_deepsicmb_{k}"] = _to_float_cell(df["total_ber_deepsicmb_1"].iloc[0])

                # ---- DeepSTAG ----
                if any(col.startswith("total_ber_deepstag") for col in df.columns):
                    if "total_ber_deepstag" in df.columns:
                        for k in [1, 2, 3]:
                            seed_snr_dict[seed][snr][f"ber_deepstag_{k}"] = _to_float_cell(df["total_ber_deepstag"].iloc[0])
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_deepstag_{k}"
                            if colname in df.columns:
                                seed_snr_dict[seed][snr][f"ber_deepstag_{k}"] = _to_float_cell(df[colname].iloc[0])
                            elif "total_ber_deepstag_1" in df.columns:
                                seed_snr_dict[seed][snr][f"ber_deepstag_{k}"] = _to_float_cell(df["total_ber_deepstag_1"].iloc[0])

                # ---- DeepRx ----
                if "total_ber_deeprx" in df.columns:
                    seed_snr_dict[seed][snr]["ber_deeprx"] = _to_float_cell(df["total_ber_deeprx"].iloc[0])

                # ---- LMMSE ----
                if "total_ber_lmmse" in df.columns:
                    seed_snr_dict[seed][snr]["ber_lmmse"] = _to_float_cell(df["total_ber_lmmse"].iloc[0])

                # ---- Sphere ----
                if "total_ber_sphere" in df.columns:
                    seed_snr_dict[seed][snr]["ber_sphere"] = _to_float_cell(df["total_ber_sphere"].iloc[0])

                # ---- MI parsing ----
                if plot_type == "MI":
                    if "total_ber_1" in df.columns:
                        seed_snr_dict[seed][snr]["mi_1"] = _to_float_cell(df["total_ber_1"].iloc[0])

                    if "total_ber_2" in df.columns:
                        seed_snr_dict[seed][snr]["mi_2"] = _to_float_cell(df["total_ber_2"].iloc[0])
                    elif "total_ber_1" in df.columns:
                        seed_snr_dict[seed][snr]["mi_2"] = _to_float_cell(df["total_ber_1"].iloc[0])

                    if "total_ber_3" in df.columns:
                        seed_snr_dict[seed][snr]["mi_3"] = _to_float_cell(df["total_ber_3"].iloc[0])
                    elif "total_ber_1" in df.columns:
                        seed_snr_dict[seed][snr]["mi_3"] = _to_float_cell(df["total_ber_1"].iloc[0])

                    if "total_ber_lmmse" in df.columns:
                        seed_snr_dict[seed][snr]["mi_lmmse"] = _to_float_cell(df["total_ber_lmmse"].iloc[0])

                    if "total_ber_sphere" in df.columns:
                        seed_snr_dict[seed][snr]["mi_sphere"] = _to_float_cell(df["total_ber_sphere"].iloc[0])

                    if "total_ber_deeprx" in df.columns:
                        seed_snr_dict[seed][snr]["mi_deeprx"] = _to_float_cell(df["total_ber_deeprx"].iloc[0])

                    for k in [1, 2, 3]:
                        colname = f"total_ber_deepsic_{k}"
                        if colname in df.columns:
                            seed_snr_dict[seed][snr][f"mi_deepsic_{k}"] = _to_float_cell(df[colname].iloc[0])

                    mi_joint_col = _get_first_present(df, ["total_ber_jointllr_1", "total_ber_jointllr"])
                    if mi_joint_col is not None:
                        seed_snr_dict[seed][snr]["mi_jointllr_1"] = _to_float_cell(df[mi_joint_col].iloc[0])

        snrs = _build_snr_grid(observed_snrs)

        if len(snrs) == 0:
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel_cur)
            if plot_type != "MI":
                ax.set_yscale("log")
            ax.grid(True)
            ax.set_title(f"No matching files found for {plot_type}")
            continue

        if plot_type == "BLER":
            bler_snrs = snrs
        elif plot_type == "BER":
            ber_snrs = snrs
        else:  # MI
            mi_snrs = snrs

        def avg(key, use_log_interp):
            curves = []
            interp_report = []
            nonmono_report = []

            for seed in seeds:
                y = []
                for s in snrs:
                    y.append(seed_snr_dict.get(seed, {}).get(s, {}).get(key, np.nan))

                y = np.asarray(y, dtype=float)

                # Step 1: remove isolated non-monotonic points
                if REMOVE_ISOLATED_NONMONO_POINTS and use_log_interp:
                    y_cleaned, removed_idx = _remove_isolated_nonmonotonic_points(
                        snrs, y, max_bad_points=MAX_BAD_POINTS_PER_CURVE
                    )
                    if PRINT_NONMONO_SUMMARY and removed_idx:
                        nonmono_report.append((seed, [snrs[i] for i in removed_idx]))
                    y = y_cleaned

                # Step 2: interpolate the resulting missing points
                if INTERPOLATE_MISSING_PER_SEED:
                    y_filled, filled_idx = _fill_missing_curve(
                        snrs,
                        y,
                        use_log_interp=use_log_interp,
                        max_gap=MAX_INTERP_GAP
                    )
                    if PRINT_INTERP_SUMMARY and filled_idx:
                        interp_report.append((seed, [snrs[i] for i in filled_idx]))
                    y = y_filled

                curves.append(y)

            if PRINT_NONMONO_SUMMARY and nonmono_report:
                pretty = ", ".join([f"seed {seed}: {pts}" for seed, pts in nonmono_report])
                print(f"[INFO] Removed non-monotonic points for key '{key}' in panel '{plot_type}': {pretty}")

            if PRINT_INTERP_SUMMARY and interp_report:
                pretty = ", ".join([f"seed {seed}: {pts}" for seed, pts in interp_report])
                print(f"[INFO] Interpolated missing points for key '{key}' in panel '{plot_type}': {pretty}")

            curves = np.asarray(curves, dtype=float)
            with np.errstate(invalid="ignore"):
                mean_curve = np.nanmean(curves, axis=0)

            # Remove isolated non-monotonic spikes from the averaged curve, then interpolate
            if use_log_interp:
                mean_curve, removed_idx = _remove_isolated_nonmonotonic_points(
                    snrs, mean_curve, max_bad_points=MAX_BAD_POINTS_PER_CURVE
                )
                if PRINT_NONMONO_SUMMARY and removed_idx:
                    print(f"[INFO] Removed non-monotonic points from averaged curve for key '{key}': SNRs={[snrs[i] for i in removed_idx]}")
                mean_curve, _ = _fill_missing_curve(snrs, mean_curve, use_log_interp=True, max_gap=MAX_INTERP_GAP)

            return mean_curve.tolist()

        markers = ["o", "*", "x", "D", "+", "s"]
        dashes = [":", "-.", "--", "-", "-"]

        if plot_type == "MI":
            mi_1 = avg("mi_1", use_log_interp=False)
            mi_2 = avg("mi_2", use_log_interp=False)
            mi_3 = avg("mi_3", use_log_interp=False)
            mi_jointllr_1 = avg("mi_jointllr_1", use_log_interp=False)
            mi_lmmse = avg("mi_lmmse", use_log_interp=False)
            mi_sphere = avg("mi_sphere", use_log_interp=False)
            mi_deeprx = avg("mi_deeprx", use_log_interp=False)
            mi_deepsic_1 = avg("mi_deepsic_1", use_log_interp=False)

            mi_avg_curves = {
                "mi_1":           mi_1,
                "mi_2":           mi_2,
                "mi_3":           mi_3,
                "mi_lmmse":       mi_lmmse,
                "mi_sphere":      mi_sphere,
                "mi_deeprx":      mi_deeprx,
                "mi_deepsic_1":   mi_deepsic_1,
                "mi_jointllr_1":  mi_jointllr_1,
            }

            ax.plot(snrs, mi_1, linestyle=dashes[0], marker=markers[0], color="g", label="ESCNN1")
            if plot_all_escnn:
                ax.plot(snrs, mi_2, linestyle=dashes[1], marker=markers[1], color="g", label="ESCNN2")
                ax.plot(snrs, mi_3, linestyle=dashes[2], marker=markers[2], color="g", label="ESCNN3")

            mi_joint_arr = np.asarray(mi_jointllr_1, dtype=float)
            if np.isfinite(mi_joint_arr).any():
                ax.plot(snrs, mi_jointllr_1, linestyle="-", marker=markers[5], color="b", label="JointLLR1")

            mi_lmmse_arr = np.asarray(mi_lmmse, dtype=float)
            if np.isfinite(mi_lmmse_arr).any():
                ax.plot(snrs, mi_lmmse, linestyle=dashes[4], marker=markers[4], color="r", label="LMMSE")

            if show_sphere:
                mi_sphere_arr = np.asarray(mi_sphere, dtype=float)
                if np.isfinite(mi_sphere_arr).any():
                    ax.plot(snrs, mi_sphere, linestyle=dashes[4], marker=markers[4], color="brown", label="Sphere")

            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel_cur)
            ax.grid(True)
            ax.legend()

        else:
            ber_1 = avg("ber_1", use_log_interp=use_log_interp_for_missing)
            ber_2 = avg("ber_2", use_log_interp=use_log_interp_for_missing)
            ber_3 = avg("ber_3", use_log_interp=use_log_interp_for_missing)
            ber_jointllr_1 = avg("ber_jointllr_1", use_log_interp=use_log_interp_for_missing)
            ber_lmmse = avg("ber_lmmse", use_log_interp=use_log_interp_for_missing)
            ber_sphere = avg("ber_sphere", use_log_interp=use_log_interp_for_missing)

            snr_target_1 = _safe_interp_x_to_y(ber_1, snrs, target_y)
            ax.semilogy(snrs, ber_1, linestyle=dashes[0], marker=markers[0], color="g",
                        label=f"ESCNN1 @ {round(100 * target_y)}% = {snr_target_1}")

            if plot_all_escnn:
                snr_target_2 = _safe_interp_x_to_y(ber_2, snrs, target_y)
                ax.semilogy(snrs, ber_2, linestyle=dashes[1], marker=markers[1], color="g",
                            label=f"ESCNN2 @ {round(100 * target_y)}% = {snr_target_2}")

                snr_target_3 = _safe_interp_x_to_y(ber_3, snrs, target_y)
                ax.semilogy(snrs, ber_3, linestyle=dashes[2], marker=markers[2], color="g",
                            label=f"ESCNN3 @ {round(100 * target_y)}% = {snr_target_3}")

            joint_arr = np.asarray(ber_jointllr_1, dtype=float)
            if np.isfinite(joint_arr).any() and not np.all(joint_arr[np.isfinite(joint_arr)] == 0):
                snr_target_joint = _safe_interp_x_to_y(ber_jointllr_1, snrs, target_y)
                ax.semilogy(snrs, ber_jointllr_1, linestyle="-", marker=markers[5], color="b",
                            label=f"JointLLR1 @ {round(100 * target_y)}% = {snr_target_joint}")

            snr_target_lmmse = _safe_interp_x_to_y(ber_lmmse, snrs, target_y)
            ax.semilogy(snrs, ber_lmmse, linestyle=dashes[4], marker=markers[4], color="r",
                        label=f"LMMSE @ {round(100 * target_y)}% = {snr_target_lmmse}")

            if show_sphere:
                sphere_arr = np.asarray(ber_sphere, dtype=float)
                if np.isfinite(sphere_arr).any() and not np.all(sphere_arr[np.isfinite(sphere_arr)] == 0):
                    snr_target_sphere = _safe_interp_x_to_y(ber_sphere, snrs, target_y)
                    ax.semilogy(snrs, ber_sphere, linestyle=dashes[4], marker=markers[4], color="brown",
                                label=f"Sphere @ {round(100 * target_y)}% = {snr_target_sphere}")

            ber_deepsic_1 = avg("ber_deepsic_1", use_log_interp=use_log_interp_for_missing)
            deepsic_arr = np.asarray(ber_deepsic_1, dtype=float)
            if np.isfinite(deepsic_arr).any() and not np.all(deepsic_arr[np.isfinite(deepsic_arr)] == 0):
                snr_target_deepsic = _safe_interp_x_to_y(ber_deepsic_1, snrs, target_y)
                ax.semilogy(snrs, ber_deepsic_1, linestyle=dashes[0], marker=markers[0], color="purple",
                            label=f"DeepSIC1 @ {round(100 * target_y)}% = {snr_target_deepsic}")

            if plot_all_escnn:
                ber_deepsic_2 = avg("ber_deepsic_2", use_log_interp=use_log_interp_for_missing)
                deepsic2_arr = np.asarray(ber_deepsic_2, dtype=float)
                if np.isfinite(deepsic2_arr).any() and not np.all(deepsic2_arr[np.isfinite(deepsic2_arr)] == 0):
                    snr_target_deepsic2 = _safe_interp_x_to_y(ber_deepsic_2, snrs, target_y)
                    ax.semilogy(snrs, ber_deepsic_2, linestyle=dashes[1], marker=markers[1], color="purple",
                                label=f"DeepSIC2 @ {round(100 * target_y)}% = {snr_target_deepsic2}")

                ber_deepsic_3 = avg("ber_deepsic_3", use_log_interp=use_log_interp_for_missing)
                deepsic3_arr = np.asarray(ber_deepsic_3, dtype=float)
                if np.isfinite(deepsic3_arr).any() and not np.all(deepsic3_arr[np.isfinite(deepsic3_arr)] == 0):
                    snr_target_deepsic3 = _safe_interp_x_to_y(ber_deepsic_3, snrs, target_y)
                    ax.semilogy(snrs, ber_deepsic_3, linestyle=dashes[2], marker=markers[2], color="purple",
                                label=f"DeepSIC3 @ {round(100 * target_y)}% = {snr_target_deepsic3}")

            ber_deeprx = avg("ber_deeprx", use_log_interp=use_log_interp_for_missing)
            deeprx_arr = np.asarray(ber_deeprx, dtype=float)
            if np.isfinite(deeprx_arr).any() and not np.all(deeprx_arr[np.isfinite(deeprx_arr)] == 0):
                snr_target_deeprx = _safe_interp_x_to_y(ber_deeprx, snrs, target_y)
                ax.semilogy(snrs, ber_deeprx, linestyle=dashes[3], marker=markers[3], color="cyan",
                            label=f"DeepRx @ {round(100 * target_y)}% = {snr_target_deeprx}")

            ber_mhsa_1 = avg("ber_mhsa_1", use_log_interp=use_log_interp_for_missing)
            mhsa_arr = np.asarray(ber_mhsa_1, dtype=float)
            if np.isfinite(mhsa_arr).any() and not np.all(mhsa_arr[np.isfinite(mhsa_arr)] == 0):
                snr_target_mhsa = _safe_interp_x_to_y(ber_mhsa_1, snrs, target_y)
                ax.semilogy(snrs, ber_mhsa_1, linestyle=dashes[0], marker=markers[0], color="orange",
                            label=f"MHSA1 @ {round(100 * target_y)}% = {snr_target_mhsa}")

            ber_tdfdcnn_1 = avg("ber_tdfdcnn_1", use_log_interp=use_log_interp_for_missing)
            tdfdcnn_arr = np.asarray(ber_tdfdcnn_1, dtype=float)
            if np.isfinite(tdfdcnn_arr).any() and not np.all(tdfdcnn_arr[np.isfinite(tdfdcnn_arr)] == 0):
                snr_target_tdfdcnn = _safe_interp_x_to_y(ber_tdfdcnn_1, snrs, target_y)
                ax.semilogy(snrs, ber_tdfdcnn_1, linestyle=dashes[1], marker=markers[1], color="olive",
                            label=f"TDFDCNN1 @ {round(100 * target_y)}% = {snr_target_tdfdcnn}")

            ber_deepsicmb_1 = avg("ber_deepsicmb_1", use_log_interp=use_log_interp_for_missing)
            deepsicmb_arr = np.asarray(ber_deepsicmb_1, dtype=float)
            if np.isfinite(deepsicmb_arr).any() and not np.all(deepsicmb_arr[np.isfinite(deepsicmb_arr)] == 0):
                snr_target_deepsicmb = _safe_interp_x_to_y(ber_deepsicmb_1, snrs, target_y)
                ax.semilogy(snrs, ber_deepsicmb_1, linestyle=dashes[2], marker=markers[2], color="magenta",
                            label=f"DeepSIC-MB1 @ {round(100 * target_y)}% = {snr_target_deepsicmb}")

            ber_deepstag_1 = avg("ber_deepstag_1", use_log_interp=use_log_interp_for_missing)
            deepstag_arr = np.asarray(ber_deepstag_1, dtype=float)
            if np.isfinite(deepstag_arr).any() and not np.all(deepstag_arr[np.isfinite(deepstag_arr)] == 0):
                snr_target_deepstag = _safe_interp_x_to_y(ber_deepstag_1, snrs, target_y)
                ax.semilogy(snrs, ber_deepstag_1, linestyle=dashes[3], marker=markers[5], color="teal",
                            label=f"DeepSTAG1 @ {round(100 * target_y)}% = {snr_target_deepstag}")

            if plot_type == "BLER":
                bler_avg_curves = {
                    "ber_1":         ber_1,
                    "ber_2":         ber_2,
                    "ber_3":         ber_3,
                    "ber_lmmse":     ber_lmmse,
                    "ber_sphere":    ber_sphere,
                    "ber_deeprx":    ber_deeprx,
                    "ber_deepsic_1": ber_deepsic_1,
                    "ber_mhsa_1":    ber_mhsa_1,
                    "ber_jointllr_1":ber_jointllr_1,
                }
            elif plot_type == "BER":
                ber_avg_curves = {
                    "ber_1":         ber_1,
                    "ber_2":         ber_2,
                    "ber_3":         ber_3,
                    "ber_lmmse":     ber_lmmse,
                    "ber_sphere":    ber_sphere,
                    "ber_deeprx":    ber_deeprx,
                    "ber_deepsic_1": ber_deepsic_1,
                    "ber_mhsa_1":    ber_mhsa_1,
                    "ber_jointllr_1":ber_jointllr_1,
                }

            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel_cur)
            ax.set_yscale("log")
            ax.grid(True)
            ax.legend()

    used_seeds_sorted = sorted(used_seeds_overall)
    if title_source_file is not None:
        original_name = os.path.basename(title_source_file)
        cleaned_name = build_cleaned_title_from_filename(original_name)
    else:
        cleaned_name = "(No files found)"

    global_title_text = "Averaged across seeds actually used: " + ", ".join(map(str, used_seeds_sorted)) + "\n" + cleaned_name

    fig.suptitle(global_title_text, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    save_path = os.path.join(CSV_DIR, "plot_output.png")
    fig.savefig(save_path, dpi=180)
    print(f"[INFO] Plot saved to: {save_path}")

    try:
        from PIL import Image
        img = Image.open(save_path)
        scale = 0.70
        new_size = (int(img.width * scale), int(img.height * scale))
        img_resized = img.resize(new_size, Image.LANCZOS)
        resized_path = save_path.replace(".png", "_clipboard.png")
        img_resized.save(resized_path)

        if platform.system() == "Windows":
            cmd = (
                'powershell -command "'
                'Add-Type -AssemblyName System.Windows.Forms;'
                f"[System.Windows.Forms.Clipboard]::SetImage("
                f"[System.Drawing.Image]::FromFile('{resized_path}'))"
                '"'
            )
            subprocess.run(cmd, shell=True, check=True)
        else:
            subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-i", resized_path],
                check=True
            )
        os.remove(resized_path)
        print(f"[INFO] Plot copied to clipboard ({int(scale*100)}% size: {new_size[0]}x{new_size[1]}).")
    except Exception as e:
        print(f"[INFO] Could not copy to clipboard: {e}")

    plt.show()

    # ---- Save averaged BLER/BER/MI curves to .mat file ----
    if bler_avg_curves is not None and bler_snrs is not None:
        def _a_bler(key):
            return np.asarray(bler_avg_curves.get(key, [np.nan] * len(bler_snrs)), dtype=float)

        def _a_ber(key):
            if ber_avg_curves is None:
                return np.full(len(bler_snrs), np.nan)
            src_len = len(ber_snrs) if ber_snrs is not None else len(bler_snrs)
            return np.asarray(ber_avg_curves.get(key, [np.nan] * src_len), dtype=float)

        def _a_mi(key):
            if mi_avg_curves is None:
                return np.full(len(bler_snrs), np.nan)
            src_len = len(mi_snrs) if mi_snrs is not None else len(bler_snrs)
            return np.asarray(mi_avg_curves.get(key, [np.nan] * src_len), dtype=float)

        # BLER arrays (pulled from BLER iteration)
        bler_escnn_arr   = _a_bler("ber_1")
        bler_escnn_2_arr = _a_bler("ber_2")
        bler_escnn_3_arr = _a_bler("ber_3")
        bler_lmmse_arr   = _a_bler("ber_lmmse")
        bler_sphere_arr  = _a_bler("ber_sphere")
        bler_deeprx_arr  = _a_bler("ber_deeprx")
        bler_deepsic_arr = _a_bler("ber_deepsic_1")
        bler_mhsa_arr    = _a_bler("ber_mhsa_1")
        bler_jointllr_arr= _a_bler("ber_jointllr_1")

        # Real BER arrays (pulled from BER iteration)
        ber_escnn_arr    = _a_ber("ber_1")
        ber_escnn_2_arr  = _a_ber("ber_2")
        ber_escnn_3_arr  = _a_ber("ber_3")
        ber_lmmse_arr    = _a_ber("ber_lmmse")
        ber_sphere_arr   = _a_ber("ber_sphere")
        ber_deeprx_arr   = _a_ber("ber_deeprx")
        ber_deepsic_arr  = _a_ber("ber_deepsic_1")
        ber_mhsa_arr     = _a_ber("ber_mhsa_1")
        ber_jointllr_arr = _a_ber("ber_jointllr_1")

        # MI arrays (pulled from MI iteration)
        mi_escnn_arr     = _a_mi("mi_1")
        mi_escnn_2_arr   = _a_mi("mi_2")
        mi_escnn_3_arr   = _a_mi("mi_3")
        mi_lmmse_arr     = _a_mi("mi_lmmse")
        mi_sphere_arr    = _a_mi("mi_sphere")
        mi_deeprx_arr    = _a_mi("mi_deeprx")
        mi_deepsic_arr   = _a_mi("mi_deepsic_1")
        mi_jointllr_arr  = _a_mi("mi_jointllr_1")

        snrs_arr = np.array(bler_snrs)
        ber_snrs_arr = np.array(ber_snrs) if ber_snrs is not None else snrs_arr
        mi_snrs_arr  = np.array(mi_snrs)  if mi_snrs  is not None else snrs_arr
        bler_target_mat = 0.1

        aug_match = re.search(r"AUGMENT_(LMMSE|SPHERE|DEEPSIC|DEEPRX)", title_source_file or "")
        aug_type = aug_match.group(1) if aug_match else "LMMSE"

        bler_no_aug_map = {
            "LMMSE":   bler_lmmse_arr,
            "SPHERE":  bler_sphere_arr,
            "DEEPRX":  bler_deeprx_arr,
            "DEEPSIC": bler_deepsic_arr,
        }
        if aug_type not in bler_no_aug_map:
            raise ValueError(
                f"plot_multiple_csvs: unknown aug_type {aug_type!r} for BLER no-aug; "
                f"supported: {sorted(bler_no_aug_map)}. Refusing to silently fall back."
            )
        bler_no_aug_arr = bler_no_aug_map[aug_type]

        mi_no_aug_map = {
            "LMMSE":   mi_lmmse_arr,
            "SPHERE":  mi_sphere_arr,
            "DEEPRX":  mi_deeprx_arr,
            "DEEPSIC": mi_deepsic_arr,
        }
        if aug_type not in mi_no_aug_map:
            raise ValueError(
                f"plot_multiple_csvs: unknown aug_type {aug_type!r} for MI no-aug; "
                f"supported: {sorted(mi_no_aug_map)}. Refusing to silently fall back."
            )
        mi_no_aug_arr = mi_no_aug_map[aug_type]

        def _snr_target(arr):
            return _safe_interp_x_to_y(arr.tolist(), snrs_arr.tolist(), bler_target_mat)

        mat_data = {
            "snrs":              snrs_arr,
            "ber_snrs":          ber_snrs_arr,
            "mi_snrs":           mi_snrs_arr,

            # Real BER curves
            "ber_escnn":         ber_escnn_arr,
            "ber_escnn_2":       ber_escnn_2_arr,
            "ber_escnn_3":       ber_escnn_3_arr,
            "ber_lmmse":         ber_lmmse_arr,
            "ber_sphere":        ber_sphere_arr,
            "ber_deeprx":        ber_deeprx_arr,
            "ber_deepsic":       ber_deepsic_arr,
            "ber_mhsa":          ber_mhsa_arr,
            "ber_jointllr":      ber_jointllr_arr,

            # BLER curves
            "bler_escnn":        bler_escnn_arr,
            "bler_escnn_2":      bler_escnn_2_arr,
            "bler_escnn_3":      bler_escnn_3_arr,
            "bler_lmmse":        bler_lmmse_arr,
            "bler_sphere":       bler_sphere_arr,
            "bler_deeprx":       bler_deeprx_arr,
            "bler_deepsic":      bler_deepsic_arr,
            "bler_mhsa":         bler_mhsa_arr,
            "bler_jointllr":     bler_jointllr_arr,
            "bler_aug":          bler_escnn_arr,
            "bler_aug_1":        bler_escnn_arr,
            "bler_aug_2":        bler_escnn_2_arr,
            "bler_aug_3":        bler_escnn_3_arr,
            "bler_no_aug":       bler_no_aug_arr,

            # MI curves
            "mi_escnn":          mi_escnn_arr,
            "mi_escnn_2":        mi_escnn_2_arr,
            "mi_escnn_3":        mi_escnn_3_arr,
            "mi_lmmse":          mi_lmmse_arr,
            "mi_sphere":         mi_sphere_arr,
            "mi_deeprx":         mi_deeprx_arr,
            "mi_deepsic":        mi_deepsic_arr,
            "mi_jointllr":       mi_jointllr_arr,
            "mi_aug":            mi_escnn_arr,
            "mi_aug_1":          mi_escnn_arr,
            "mi_aug_2":          mi_escnn_2_arr,
            "mi_aug_3":          mi_escnn_3_arr,
            "mi_no_aug":         mi_no_aug_arr,

            # SNR targets (BLER @ 10%)
            "snr_target_aug":    _snr_target(bler_escnn_arr),
            "snr_target_aug_1":  _snr_target(bler_escnn_arr),
            "snr_target_aug_2":  _snr_target(bler_escnn_2_arr),
            "snr_target_aug_3":  _snr_target(bler_escnn_3_arr),
            "snr_target_no_aug": _snr_target(bler_no_aug_arr),
            "titletext":         build_cleaned_title_from_filename(os.path.basename(title_source_file)) if title_source_file else "",
        }

        mat_output_dir = os.path.join(CSV_DIR, "mat_files")
        os.makedirs(mat_output_dir, exist_ok=True)
        mat_name = aug_type.lower() if aug_match else "results"
        mat_path = os.path.join(mat_output_dir, mat_name + ".mat")
        savemat(mat_path, mat_data)
        print(f"[INFO] Mat file saved to: {mat_path}")


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else None
    plot_all_iters = "plot_all_iters" in sys.argv[2:] if len(sys.argv) > 2 else False
    plot_csvs(pattern, plot_all_iters=plot_all_iters)