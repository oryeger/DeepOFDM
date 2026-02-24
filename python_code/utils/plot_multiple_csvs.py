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

# 🔧 Configuration
CSV_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Scratchpad"))
seeds = [123, 17, 41, 58]
MIN_SNR = -np.inf  # -np.inf Set to a number (e.g., 20) to limit max SNR, or np.inf to plot all
MAX_SNR = np.inf  # np.inf Set to a number (e.g., 20) to limit max SNR, or np.inf to plot all

# ---- Helper: build pretty title from a filename (your exact logic) ----
def build_cleaned_title_from_filename(original_name: str) -> str:
    cleaned_name = re.sub(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_", "", original_name)
    cleaned_name = re.sub(".csv", "", cleaned_name)
    cleaned_name = re.sub("Clip=100%", "", cleaned_name)
    # Remove any occurrence of _SNR= followed by any number (including negative)
    cleaned_name = re.sub(r"_SNR=-?\d+", "", cleaned_name)
    cleaned_name = re.sub("_scs", "scs", cleaned_name)
    cleaned_name = re.sub("cfo_in_Rx", "cfo", cleaned_name)
    cleaned_name = re.sub(r"seed=\d+", "", cleaned_name)
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
    # handle "0" / "0.0" / empty
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
    Your code does interp1d(ber, snr). That requires x to be monotonic-ish.
    This helper makes it safer by sorting by x and dropping duplicates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan

    # Sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Drop duplicate x (keep first)
    x_unique, idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[idx]

    if x_unique.size < 2:
        return np.nan

    f = interp1d(x_unique, y_unique, fill_value="extrapolate", bounds_error=False)
    return float(np.round(f(x_target), 1))


def plot_csvs(filter_pattern=None, plot_all_iters=False):
    """
    Plot BER/BLER/MI from CSV files in CSV_DIR.

    Args:
        filter_pattern: Optional glob pattern string to select files.
                        E.g. "*TRN=4032*C=No*AUGMENT_LMMSE*0.46*ncPrime_0*s64*.csv"
                        If None, all files in CSV_DIR are used.
        plot_all_iters: If True, plot all iterations (ESCNN1/2/3, DeepSIC1/2/3, etc.).
                        If False (default), plot only the last iteration.
    """
    plot_all_escnn = plot_all_iters

    if filter_pattern is not None:
        all_files = sorted(glob.glob(os.path.join(CSV_DIR, filter_pattern)))
    else:
        all_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))

    print(f"[INFO] Found {len(all_files)} files matching pattern.")

    # ---- Decide whether to show Sphere curves ----
    pat_upper = (filter_pattern or "").upper()
    show_sphere = "SPHERE" in pat_upper and "PRIME_1" not in pat_upper

    # ---- Check if MI files exist ----
    mi_files_exist = any(f.endswith("_mi.csv") for f in all_files)

    # ---- Prepare figure with 2 or 3 subplots depending on MI files ----
    if mi_files_exist:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
        # BLER=left(0), MI=middle(1), BER=right(2)
        subplot_index = {"BER": 2, "BLER": 0, "MI": 1}
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.5))
        # BLER=left(0), BER=right(1)
        subplot_index = {"BER": 1, "BLER": 0}

    # ---- MAIN LOOP: plot BLER, MI (if exists), BER ----
    used_seeds_overall = set()
    title_source_file = None  # keep one real file used (for the cleaned title)

    # Define plot types: BLER, MI (optional), BER
    plot_types = ["BLER", "BER"]
    if mi_files_exist:
        plot_types = ["BLER", "MI", "BER"]

    for plot_type in plot_types:
        ax = axes[subplot_index[plot_type]]

        if plot_type == "BER":
            search_pattern = r"SNR=(-?\d+)"
            ber_target = 0.01
            ylabel_cur = "BER"
        elif plot_type == "BLER":
            search_pattern = r"_SNR=(-?\d+)_bler\.csv$"
            ber_target = 0.1
            ylabel_cur = "BLER"
        else:  # MI
            search_pattern = r"_SNR=(-?\d+)_mi\.csv$"
            ber_target = None  # No target threshold for MI
            ylabel_cur = "MI"

        snr_ber_dict = defaultdict(lambda: {
            "ber_1": [], "ber_2": [], "ber_3": [], "ber_deeprx": [],
            "ber_deepsic_1": [], "ber_deepsic_2": [], "ber_deepsic_3": [],
            "ber_e2e_1": [], "ber_e2e_2": [], "ber_e2e_3": [],
            "ber_deepsicmb_1": [], "ber_deepsicmb_2": [], "ber_deepsicmb_3": [],
            "ber_deepstag_1": [], "ber_deepstag_2": [], "ber_deepstag_3": [],
            "ber_lmmse": [], "ber_sphere": [],
            "ber_mhsa_1": [], "ber_mhsa_2": [], "ber_mhsa_3": [],
            "ber_tdfdcnn_1": [], "ber_tdfdcnn_2": [], "ber_tdfdcnn_3": [],
            # ✅ NEW: JointLLR (support both total_ber_jointllr and total_ber_jointllr_1)
            "ber_jointllr_1": [],
            # MI columns
            "mi_1": [], "mi_2": [], "mi_3": [],
            "mi_lmmse": [], "mi_sphere": [],
            "mi_jointllr_1": [],
        })

        used_seeds_this_panel = set()

        # ---- Load CSVs for each seed ----
        for seed in seeds:
            seed_files = sorted(f for f in all_files
                                if f"seed={seed}" in os.path.basename(f)
                                and "_SNR=" in os.path.basename(f))
            if not seed_files:
                continue

            seen_snr = set()
            unique_files = []

            for file in seed_files:
                match = re.search(search_pattern, file)
                if match:
                    snr = match.group(1)
                    if snr not in seen_snr:
                        seen_snr.add(snr)
                        unique_files.append(file)

            if not unique_files:
                continue

            # If we got here, this seed is actually used (has at least one matching file)
            used_seeds_this_panel.add(seed)
            used_seeds_overall.add(seed)
            if title_source_file is None:
                title_source_file = unique_files[0]

            for file in unique_files:
                match = re.search(search_pattern, file)
                if not match:
                    continue
                snr = int(match.group(1))

                df = pd.read_csv(file)

                # ------------ BER parsing (same as your script) ------------
                snr_ber_dict[snr]["ber_1"].append(_to_float_cell(df["total_ber_1"].iloc[0]))

                if "total_ber_2" in df.columns:
                    snr_ber_dict[snr]["ber_2"].append(_to_float_cell(df["total_ber_2"].iloc[0]))
                else:
                    snr_ber_dict[snr]["ber_2"].append(_to_float_cell(df["total_ber_1"].iloc[0]))

                if "total_ber_3" in df.columns:
                    snr_ber_dict[snr]["ber_3"].append(_to_float_cell(df["total_ber_3"].iloc[0]))
                else:
                    snr_ber_dict[snr]["ber_3"].append(_to_float_cell(df["total_ber_1"].iloc[0]))

                # ✅ NEW: JointLLR parsing
                joint_col = _get_first_present(df, ["total_ber_jointllr_1", "total_ber_jointllr"])
                if joint_col is not None:
                    snr_ber_dict[snr]["ber_jointllr_1"].append(_to_float_cell(df[joint_col].iloc[0]))

                # MHSA parsing
                if any(col.startswith("total_ber_mhsa") for col in df.columns):
                    if "total_ber_mhsa" in df.columns and not any(col.startswith("total_ber_mhsa_") for col in df.columns):
                        val = _to_float_cell(df["total_ber_mhsa"].iloc[0])
                        for key in ["ber_mhsa_1", "ber_mhsa_2", "ber_mhsa_3"]:
                            snr_ber_dict[snr][key].append(val)
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_mhsa_{k}"
                            if colname in df.columns:
                                snr_ber_dict[snr][f"ber_mhsa_{k}"].append(_to_float_cell(df[colname].iloc[0]))
                            elif "total_ber_mhsa" in df.columns:
                                snr_ber_dict[snr][f"ber_mhsa_{k}"].append(_to_float_cell(df["total_ber_mhsa"].iloc[0]))

                # TDFDCNN parsing (tfdfcnn)
                if any(col.startswith("total_ber_tfdfcnn") for col in df.columns):
                    if "total_ber_tfdfcnn" in df.columns and not any(col.startswith("total_ber_tfdfcnn_") for col in df.columns):
                        val = _to_float_cell(df["total_ber_tfdfcnn"].iloc[0])
                        for key in ["ber_tdfdcnn_1", "ber_tdfdcnn_2", "ber_tdfdcnn_3"]:
                            snr_ber_dict[snr][key].append(val)
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_tfdfcnn_{k}"
                            if colname in df.columns:
                                snr_ber_dict[snr][f"ber_tdfdcnn_{k}"].append(_to_float_cell(df[colname].iloc[0]))
                            elif "total_ber_tdfdcnn" in df.columns:
                                snr_ber_dict[snr][f"ber_tdfdcnn_{k}"].append(_to_float_cell(df["total_ber_tfdfcnn"].iloc[0]))

                # TDFDCNN parsing (tdfdcnn)
                if any(col.startswith("total_ber_tdfdcnn") for col in df.columns):
                    if "total_ber_tdfdcnn" in df.columns and not any(col.startswith("total_ber_tdfdcnn_") for col in df.columns):
                        val = _to_float_cell(df["total_ber_tdfdcnn"].iloc[0])
                        for key in ["ber_tdfdcnn_1", "ber_tdfdcnn_2", "ber_tdfdcnn_3"]:
                            snr_ber_dict[snr][key].append(val)
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_tdfdcnn_{k}"
                            if colname in df.columns:
                                snr_ber_dict[snr][f"ber_tdfdcnn_{k}"].append(_to_float_cell(df[colname].iloc[0]))
                            elif "total_ber_tdfdcnn" in df.columns:
                                snr_ber_dict[snr][f"ber_tdfdcnn_{k}"].append(_to_float_cell(df["total_ber_tdfdcnn"].iloc[0]))

                # DeepSIC
                if any(col.startswith("total_ber_deepsic") for col in df.columns):
                    if "total_ber_deepsic" in df.columns:
                        for k in [1, 2, 3]:
                            snr_ber_dict[snr][f"ber_deepsic_{k}"].append(_to_float_cell(df["total_ber_deepsic"].iloc[0]))
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_deepsic_{k}"
                            if colname in df.columns:
                                snr_ber_dict[snr][f"ber_deepsic_{k}"].append(_to_float_cell(df[colname].iloc[0]))
                            else:
                                snr_ber_dict[snr][f"ber_deepsic_{k}"].append(_to_float_cell(df["total_ber_deepsic_1"].iloc[0]))

                # DeepSIC-MB
                if any(col.startswith("total_ber_deepsicmb") for col in df.columns):
                    if "total_ber_deepsicmb" in df.columns:
                        for k in [1, 2, 3]:
                            snr_ber_dict[snr][f"ber_deepsicmb_{k}"].append(_to_float_cell(df["total_ber_deepsicmb"].iloc[0]))
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_deepsicmb_{k}"
                            if colname in df.columns:
                                snr_ber_dict[snr][f"ber_deepsicmb_{k}"].append(_to_float_cell(df[colname].iloc[0]))
                            else:
                                snr_ber_dict[snr][f"ber_deepsicmb_{k}"].append(_to_float_cell(df["total_ber_deepsicmb_1"].iloc[0]))

                # DeepSTAG
                if any(col.startswith("total_ber_deepstag") for col in df.columns):
                    if "total_ber_deepstag" in df.columns:
                        for k in [1, 2, 3]:
                            snr_ber_dict[snr][f"ber_deepstag_{k}"].append(_to_float_cell(df["total_ber_deepstag"].iloc[0]))
                    else:
                        for k in [1, 2, 3]:
                            colname = f"total_ber_deepstag_{k}"
                            if colname in df.columns:
                                snr_ber_dict[snr][f"ber_deepstag_{k}"].append(_to_float_cell(df[colname].iloc[0]))
                            else:
                                snr_ber_dict[snr][f"ber_deepstag_{k}"].append(_to_float_cell(df["total_ber_deepstag_1"].iloc[0]))

                # DeepRx
                if "total_ber_deeprx" in df.columns:
                    snr_ber_dict[snr]["ber_deeprx"].append(_to_float_cell(df["total_ber_deeprx"].iloc[0]))

                # LMMSE
                if "total_ber_lmmse" in df.columns:
                    snr_ber_dict[snr]["ber_lmmse"].append(_to_float_cell(df["total_ber_lmmse"].iloc[0]))

                # Sphere
                if "total_ber_sphere" in df.columns:
                    snr_ber_dict[snr]["ber_sphere"].append(_to_float_cell(df["total_ber_sphere"].iloc[0]))

                # ---- MI parsing (for MI files) ----
                if plot_type == "MI":
                    if "total_ber_1" in df.columns:
                        snr_ber_dict[snr]["mi_1"].append(_to_float_cell(df["total_ber_1"].iloc[0]))
                    if "total_ber_2" in df.columns:
                        snr_ber_dict[snr]["mi_2"].append(_to_float_cell(df["total_ber_2"].iloc[0]))
                    elif "total_ber_1" in df.columns:
                        snr_ber_dict[snr]["mi_2"].append(_to_float_cell(df["total_ber_1"].iloc[0]))
                    if "total_ber_3" in df.columns:
                        snr_ber_dict[snr]["mi_3"].append(_to_float_cell(df["total_ber_3"].iloc[0]))
                    elif "total_ber_1" in df.columns:
                        snr_ber_dict[snr]["mi_3"].append(_to_float_cell(df["total_ber_1"].iloc[0]))

                    # MI LMMSE
                    if "total_ber_lmmse" in df.columns:
                        snr_ber_dict[snr]["mi_lmmse"].append(_to_float_cell(df["total_ber_lmmse"].iloc[0]))

                    # MI Sphere
                    if "total_ber_sphere" in df.columns:
                        snr_ber_dict[snr]["mi_sphere"].append(_to_float_cell(df["total_ber_sphere"].iloc[0]))

                    # MI JointLLR
                    mi_joint_col = _get_first_present(df, ["total_ber_jointllr_1", "total_ber_jointllr"])
                    if mi_joint_col is not None:
                        snr_ber_dict[snr]["mi_jointllr_1"].append(_to_float_cell(df[mi_joint_col].iloc[0]))

        # If nothing was loaded for this panel, skip plotting
        if len(snr_ber_dict) == 0:
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel_cur)
            if plot_type != "MI":
                ax.set_yscale("log")
            ax.grid(True)
            ax.set_title(f"No matching files found for {plot_type}")
            continue

        # ---- Averages ----
        snrs = sorted([s for s in snr_ber_dict.keys() if MIN_SNR <= s <= MAX_SNR])

        def avg(key):
            vals = []
            for s in snrs:
                arr = np.asarray(snr_ber_dict[s][key], dtype=float)
                arr = arr[np.isfinite(arr)]
                vals.append(np.mean(arr) if arr.size else np.nan)
            return vals

        markers = ["o", "*", "x", "D", "+", "s"]
        dashes  = [":", "-.", "--", "-", "-"]

        if plot_type == "MI":
            # ---- MI plotting (linear scale) ----
            mi_1 = avg("mi_1")
            mi_2 = avg("mi_2")
            mi_3 = avg("mi_3")
            mi_jointllr_1 = avg("mi_jointllr_1")
            mi_lmmse = avg("mi_lmmse")
            mi_sphere = avg("mi_sphere")

            # Plot ESCNN1/2/3 MI
            ax.plot(snrs, mi_1, linestyle=dashes[0], marker=markers[0], color="g", label="ESCNN1")
            if plot_all_escnn:
                ax.plot(snrs, mi_2, linestyle=dashes[1], marker=markers[1], color="g", label="ESCNN2")
                ax.plot(snrs, mi_3, linestyle=dashes[2], marker=markers[2], color="g", label="ESCNN3")

            # JointLLR MI
            mi_joint_arr = np.asarray(mi_jointllr_1, dtype=float)
            if np.isfinite(mi_joint_arr).any():
                ax.plot(snrs, mi_jointllr_1, linestyle="-", marker=markers[5], color="b", label="JointLLR1")

            # LMMSE MI
            mi_lmmse_arr = np.asarray(mi_lmmse, dtype=float)
            if np.isfinite(mi_lmmse_arr).any():
                ax.plot(snrs, mi_lmmse, linestyle=dashes[4], marker=markers[4], color="r", label="LMMSE")

            # Sphere MI
            if show_sphere:
                mi_sphere_arr = np.asarray(mi_sphere, dtype=float)
                if np.isfinite(mi_sphere_arr).any():
                    ax.plot(snrs, mi_sphere, linestyle=dashes[4], marker=markers[4], color="brown", label="Sphere")

            # Formatting for MI (linear scale)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel_cur)
            ax.grid(True)
            ax.legend()

        else:
            # ---- BER/BLER plotting (log scale) ----
            ber_1 = avg("ber_1")
            ber_2 = avg("ber_2")
            ber_3 = avg("ber_3")
            ber_jointllr_1 = avg("ber_jointllr_1")
            ber_lmmse = avg("ber_lmmse")
            ber_sphere = avg("ber_sphere")

            # ---- Plot ESCNN1/2/3 ----
            snr_target_1 = _safe_interp_x_to_y(ber_1, snrs, ber_target)
            ax.semilogy(snrs, ber_1, linestyle=dashes[0], marker=markers[0], color="g",
                        label=f"ESCNN1 @ {round(100*ber_target)}% = {snr_target_1}")

            if plot_all_escnn:
                snr_target_2 = _safe_interp_x_to_y(ber_2, snrs, ber_target)
                ax.semilogy(snrs, ber_2, linestyle=dashes[1], marker=markers[1], color="g",
                            label=f"ESCNN2 @ {round(100*ber_target)}% = {snr_target_2}")

                snr_target_3 = _safe_interp_x_to_y(ber_3, snrs, ber_target)
                ax.semilogy(snrs, ber_3, linestyle=dashes[2], marker=markers[2], color="g",
                            label=f"ESCNN3 @ {round(100*ber_target)}% = {snr_target_3}")

            # JointLLR curve (only if exists / not all-NaN / not flat single value)
            joint_arr = np.asarray(ber_jointllr_1, dtype=float)
            if np.isfinite(joint_arr).any() and not np.all(joint_arr[np.isfinite(joint_arr)] == 0):
                snr_target_joint = _safe_interp_x_to_y(ber_jointllr_1, snrs, ber_target)
                ax.semilogy(snrs, ber_jointllr_1, linestyle="-", marker=markers[5], color="b",
                            label=f"JointLLR1 @ {round(100*ber_target)}% = {snr_target_joint}")

            # ---- LMMSE ----
            snr_target_lmmse = _safe_interp_x_to_y(ber_lmmse, snrs, ber_target)
            ax.semilogy(snrs, ber_lmmse, linestyle=dashes[4], marker=markers[4], color="r",
                        label=f"LMMSE @ {round(100*ber_target)}% = {snr_target_lmmse}")

            # ---- Sphere ----
            if show_sphere:
                sphere_arr = np.asarray(ber_sphere, dtype=float)
                if np.isfinite(sphere_arr).any() and not np.all(sphere_arr[np.isfinite(sphere_arr)] == 0):
                    snr_target_sphere = _safe_interp_x_to_y(ber_sphere, snrs, ber_target)
                    ax.semilogy(snrs, ber_sphere, linestyle=dashes[4], marker=markers[4], color="brown",
                                label=f"Sphere @ {round(100*ber_target)}% = {snr_target_sphere}")

            # ---- DeepSIC ----
            ber_deepsic_1 = avg("ber_deepsic_1")
            deepsic_arr = np.asarray(ber_deepsic_1, dtype=float)
            if np.isfinite(deepsic_arr).any() and not np.all(deepsic_arr[np.isfinite(deepsic_arr)] == 0):
                snr_target_deepsic = _safe_interp_x_to_y(ber_deepsic_1, snrs, ber_target)
                ax.semilogy(snrs, ber_deepsic_1, linestyle=dashes[0], marker=markers[0], color="purple",
                            label=f"DeepSIC1 @ {round(100*ber_target)}% = {snr_target_deepsic}")

            if plot_all_escnn:
                ber_deepsic_2 = avg("ber_deepsic_2")
                deepsic2_arr = np.asarray(ber_deepsic_2, dtype=float)
                if np.isfinite(deepsic2_arr).any() and not np.all(deepsic2_arr[np.isfinite(deepsic2_arr)] == 0):
                    snr_target_deepsic2 = _safe_interp_x_to_y(ber_deepsic_2, snrs, ber_target)
                    ax.semilogy(snrs, ber_deepsic_2, linestyle=dashes[1], marker=markers[1], color="purple",
                                label=f"DeepSIC2 @ {round(100*ber_target)}% = {snr_target_deepsic2}")

                ber_deepsic_3 = avg("ber_deepsic_3")
                deepsic3_arr = np.asarray(ber_deepsic_3, dtype=float)
                if np.isfinite(deepsic3_arr).any() and not np.all(deepsic3_arr[np.isfinite(deepsic3_arr)] == 0):
                    snr_target_deepsic3 = _safe_interp_x_to_y(ber_deepsic_3, snrs, ber_target)
                    ax.semilogy(snrs, ber_deepsic_3, linestyle=dashes[2], marker=markers[2], color="purple",
                                label=f"DeepSIC3 @ {round(100*ber_target)}% = {snr_target_deepsic3}")

            # ---- DeepRx ----
            ber_deeprx = avg("ber_deeprx")
            deeprx_arr = np.asarray(ber_deeprx, dtype=float)
            if np.isfinite(deeprx_arr).any() and not np.all(deeprx_arr[np.isfinite(deeprx_arr)] == 0):
                snr_target_deeprx = _safe_interp_x_to_y(ber_deeprx, snrs, ber_target)
                ax.semilogy(snrs, ber_deeprx, linestyle=dashes[3], marker=markers[3], color="cyan",
                            label=f"DeepRx @ {round(100*ber_target)}% = {snr_target_deeprx}")

            # ---- MHSA ----
            ber_mhsa_1 = avg("ber_mhsa_1")
            mhsa_arr = np.asarray(ber_mhsa_1, dtype=float)
            if np.isfinite(mhsa_arr).any() and not np.all(mhsa_arr[np.isfinite(mhsa_arr)] == 0):
                snr_target_mhsa = _safe_interp_x_to_y(ber_mhsa_1, snrs, ber_target)
                ax.semilogy(snrs, ber_mhsa_1, linestyle=dashes[0], marker=markers[0], color="orange",
                            label=f"MHSA1 @ {round(100*ber_target)}% = {snr_target_mhsa}")

            # ---- TDFDCNN ----
            ber_tdfdcnn_1 = avg("ber_tdfdcnn_1")
            tdfdcnn_arr = np.asarray(ber_tdfdcnn_1, dtype=float)
            if np.isfinite(tdfdcnn_arr).any() and not np.all(tdfdcnn_arr[np.isfinite(tdfdcnn_arr)] == 0):
                snr_target_tdfdcnn = _safe_interp_x_to_y(ber_tdfdcnn_1, snrs, ber_target)
                ax.semilogy(snrs, ber_tdfdcnn_1, linestyle=dashes[1], marker=markers[1], color="olive",
                            label=f"TDFDCNN1 @ {round(100*ber_target)}% = {snr_target_tdfdcnn}")

            # ---- DeepSIC-MB ----
            ber_deepsicmb_1 = avg("ber_deepsicmb_1")
            deepsicmb_arr = np.asarray(ber_deepsicmb_1, dtype=float)
            if np.isfinite(deepsicmb_arr).any() and not np.all(deepsicmb_arr[np.isfinite(deepsicmb_arr)] == 0):
                snr_target_deepsicmb = _safe_interp_x_to_y(ber_deepsicmb_1, snrs, ber_target)
                ax.semilogy(snrs, ber_deepsicmb_1, linestyle=dashes[2], marker=markers[2], color="magenta",
                            label=f"DeepSIC-MB1 @ {round(100*ber_target)}% = {snr_target_deepsicmb}")

            # ---- DeepSTAG ----
            ber_deepstag_1 = avg("ber_deepstag_1")
            deepstag_arr = np.asarray(ber_deepstag_1, dtype=float)
            if np.isfinite(deepstag_arr).any() and not np.all(deepstag_arr[np.isfinite(deepstag_arr)] == 0):
                snr_target_deepstag = _safe_interp_x_to_y(ber_deepstag_1, snrs, ber_target)
                ax.semilogy(snrs, ber_deepstag_1, linestyle=dashes[3], marker=markers[5], color="teal",
                            label=f"DeepSTAG1 @ {round(100*ber_target)}% = {snr_target_deepstag}")

            # ---- Formatting ----
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel_cur)
            ax.set_yscale("log")
            ax.grid(True)
            ax.legend()

    # ---- Build global title using ONLY seeds that were actually used ----
    used_seeds_sorted = sorted(used_seeds_overall)
    if title_source_file is not None:
        original_name = os.path.basename(title_source_file)
        cleaned_name = build_cleaned_title_from_filename(original_name)
    else:
        cleaned_name = "(No files found)"

    global_title_text = "Averaged across seeds actually used: " + ", ".join(map(str, used_seeds_sorted)) + "\n" + cleaned_name

    # ---- Add global title centered above all plots ----
    fig.suptitle(global_title_text, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    # Save to file (works on headless servers), then try to show
    save_path = os.path.join(CSV_DIR, "plot_output.png")
    fig.savefig(save_path, dpi=180)
    print(f"[INFO] Plot saved to: {save_path}")

    # Try to copy resized image to clipboard (75% to keep it manageable)
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


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else None
    plot_all_iters = "plot_all_iters" in sys.argv[2:] if len(sys.argv) > 2 else False
    plot_csvs(pattern, plot_all_iters=plot_all_iters)
