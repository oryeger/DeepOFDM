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
MIN_SNR = -np.inf  # -np.inf Set to a number (e.g., 0) to limit min SNR, or -np.inf to plot all
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


def plot_csvs(filter_pattern=None):
    """
    Plot BER/BLER/MI from CSV files in CSV_DIR.

    Args:
        filter_pattern: Optional glob pattern string to select files.
                        E.g. "*TRN=4032*C=No*AUGMENT_LMMSE*0.46*ncPrime_0*s64*.csv"
                        If None, all files in CSV_DIR are used.
    """
    print(f"[DEBUG] CSV_DIR = {CSV_DIR}")
    print(f"[DEBUG] CSV_DIR exists = {os.path.isdir(CSV_DIR)}")
    print(f"[DEBUG] filter_pattern = {filter_pattern}")

    if filter_pattern is not None:
        glob_expr = os.path.join(CSV_DIR, filter_pattern)
        all_files = sorted(glob.glob(glob_expr))
    else:
        glob_expr = os.path.join(CSV_DIR, "*.csv")
        all_files = sorted(glob.glob(glob_expr))

    print(f"[DEBUG] glob expression = {glob_expr}")
    print(f"[DEBUG] matched {len(all_files)} files")
    if all_files:
        print(f"[DEBUG] first file: {all_files[0]}")
        print(f"[DEBUG] last file:  {all_files[-1]}")
    else:
        print(f"[DEBUG] No files found! Listing CSV_DIR contents:")
        if os.path.isdir(CSV_DIR):
            contents = os.listdir(CSV_DIR)
            print(f"[DEBUG]   {len(contents)} items in directory")
            for f in contents[:10]:
                print(f"[DEBUG]   - {f}")
            if len(contents) > 10:
                print(f"[DEBUG]   ... and {len(contents) - 10} more")
        else:
            print(f"[DEBUG]   Directory does not exist!")

    # ---- Check if MI files exist ----
    mi_files_exist = any(f.endswith("_mi.csv") for f in all_files)

    # ---- Prepare figure with 2 or 3 subplots depending on MI files ----
    if mi_files_exist:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        # BLER=left(0), MI=middle(1), BER=right(2)
        subplot_index = {"BER": 2, "BLER": 0, "MI": 1}
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
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
            print(f"[DEBUG] {plot_type}: seed={seed} -> {len(seed_files)} files")
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

            snr_target_2 = _safe_interp_x_to_y(ber_2, snrs, ber_target)
            ax.semilogy(snrs, ber_2, linestyle=dashes[1], marker=markers[1], color="g",
                        label=f"ESCNN2 @ {round(100*ber_target)}% = {snr_target_2}")

            snr_target_3 = _safe_interp_x_to_y(ber_3, snrs, ber_target)
            ax.semilogy(snrs, ber_3, linestyle=dashes[2], marker=markers[2], color="g",
                        label=f"ESCNN3 @ {round(100*ber_target)}% = {snr_target_3}")

            # JointLLR curve (only if exists / not all-NaN / not flat single value)
            joint_arr = np.asarray(ber_jointllr_1, dtype=float)
            if np.isfinite(joint_arr).any() and (np.unique(joint_arr[np.isfinite(joint_arr)]).shape[0] > 1):
                snr_target_joint = _safe_interp_x_to_y(ber_jointllr_1, snrs, ber_target)
                ax.semilogy(snrs, ber_jointllr_1, linestyle="-", marker=markers[5], color="b",
                            label=f"JointLLR1 @ {round(100*ber_target)}% = {snr_target_joint}")
            elif np.isfinite(joint_arr).any():
                # still plot it (flat) but without target label confusion
                ax.semilogy(snrs, ber_jointllr_1, linestyle="-", marker=markers[5], color="b",
                            label=f"JointLLR1")

            # ---- LMMSE ----
            snr_target_lmmse = _safe_interp_x_to_y(ber_lmmse, snrs, ber_target)
            ax.semilogy(snrs, ber_lmmse, linestyle=dashes[4], marker=markers[4], color="r",
                        label=f"LMMSE @ {round(100*ber_target)}% = {snr_target_lmmse}")

            # ---- Sphere ----
            sphere_arr = np.asarray(ber_sphere, dtype=float)
            if np.isfinite(sphere_arr).any() and not (np.unique(sphere_arr[np.isfinite(sphere_arr)]).shape[0] == 1):
                snr_target_sphere = _safe_interp_x_to_y(ber_sphere, snrs, ber_target)
                ax.semilogy(snrs, ber_sphere, linestyle=dashes[4], marker=markers[4], color="brown",
                            label=f"Sphere @ {round(100*ber_target)}% = {snr_target_sphere}")
            elif np.isfinite(sphere_arr).any():
                ax.semilogy(snrs, ber_sphere, linestyle=dashes[4], marker=markers[4], color="brown",
                            label=f"Sphere")

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
    fig.savefig(save_path, dpi=150)
    print(f"[INFO] Plot saved to: {save_path}")

    # Try to copy resized image to clipboard (75% to keep it manageable)
    try:
        from PIL import Image
        img = Image.open(save_path)
        scale = 0.80
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
    print(f"[DEBUG] sys.argv = {sys.argv}")
    pattern = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"[DEBUG] pattern = {pattern}")
    plot_csvs(pattern)
