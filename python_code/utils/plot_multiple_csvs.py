import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.interpolate import interp1d
import numpy as np
from collections import defaultdict

# 🔧 Adjust path
CSV_DIR = r"C:\Projects\Scratchpad"
seeds = [123, 17, 41, 58]

# ---- Prepare a single combined figure with two subplots ----
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
subplot_index = {1: 1, 0: 0}  # BER=1 → right, BER=0 → left

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

# ---- MAIN LOOP: plot BER=1 (right) and BER=0 (left) ----
used_seeds_overall = set()
title_source_file = None  # keep one real file used (for the cleaned title)

for BER in [1, 0]:
    ax = axes[subplot_index[BER]]

    if BER:
        search_pattern = r"SNR=(-?\d+)"
        ber_target = 0.01
        ylabel_cur = "BER"
    else:
        search_pattern = r"_SNR=(-?\d+)_bler\.csv$"
        ber_target = 0.1
        ylabel_cur = "BLER"

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
    })

    used_seeds_this_panel = set()

    # ---- Load CSVs for each seed ----
    for seed in seeds:
        seed_files = sorted(glob.glob(os.path.join(CSV_DIR, f"*seed={seed}*_SNR=*")))
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
            # Examples you gave:
            #   total_ber_jointllr_1
            # Sometimes it might be total_ber_jointllr (no suffix)
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

    # If nothing was loaded for this panel, skip plotting
    if len(snr_ber_dict) == 0:
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel_cur)
        ax.set_yscale("log")
        ax.grid(True)
        ax.set_title(f"No matching files found for {'BER' if BER else 'BLER'}")
        continue

    # ---- Averages ----
    snrs = sorted(snr_ber_dict.keys())

    def avg(key):
        vals = []
        for s in snrs:
            arr = np.asarray(snr_ber_dict[s][key], dtype=float)
            arr = arr[np.isfinite(arr)]
            vals.append(np.mean(arr) if arr.size else np.nan)
        return vals

    ber_1 = avg("ber_1")
    ber_2 = avg("ber_2")
    ber_3 = avg("ber_3")
    ber_jointllr_1 = avg("ber_jointllr_1")  # ✅ NEW
    ber_lmmse = avg("ber_lmmse")
    ber_sphere = avg("ber_sphere")

    markers = ["o", "*", "x", "D", "+", "s"]
    dashes  = [":", "-.", "--", "-", "-"]

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

    # ✅ NEW: JointLLR curve (only if exists / not all-NaN / not flat single value)
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

# ---- Add global title centered above BOTH plots ----
fig.suptitle(global_title_text, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
