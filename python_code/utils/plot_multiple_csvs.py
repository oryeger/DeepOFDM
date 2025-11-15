import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.interpolate import interp1d
import numpy as np
from collections import defaultdict
from scipy.io import savemat

# 🔧 Adjust path
CSV_DIR = r"C:\Projects\Scratchpad"
seeds = [123, 17, 41, 58]

# ---- Prepare a single combined figure with two subplots ----
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
subplot_index = {1: 1, 0: 0}  # BER=1 → right, BER=0 → left

# ---- We will compute the meaningful title once (after collecting a file) ----
# Get an example filename to reconstruct the title
example_file = sorted(glob.glob(os.path.join(CSV_DIR, "*seed="+str(seeds[0])+"*_SNR=*")))[0]
original_name = os.path.basename(example_file)

# Clean formatting exactly like your original code
cleaned_name = re.sub(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_", "", original_name)
cleaned_name = re.sub(".csv", "", cleaned_name)
cleaned_name = re.sub("Clip=100%", "", cleaned_name)
cleaned_name = re.sub(r"_SNR=\d+$", "", cleaned_name)
cleaned_name = re.sub("_scs", "scs", cleaned_name)
cleaned_name = re.sub("cfo_in_Rx", "cfo", cleaned_name)
cleaned_name = re.sub(r"seed=\d+", "", cleaned_name)
cleaned_name = re.sub("_", ", ", cleaned_name)
cleaned_name = re.sub("twostage, ","", cleaned_name)
cleaned_name = re.sub(", , , three, layers=123",", three layes", cleaned_name)
cleaned_name = cleaned_name.rstrip(", ")

# Wrap long title text into multiple lines every 80 characters
cleaned_name = "\n".join([cleaned_name[i:i+80] for i in range(0, len(cleaned_name), 80)])

global_title_text = "Averaged across seeds: " + ", ".join(map(str, seeds)) + "\n" + cleaned_name


# ---- MAIN LOOP: plot BER=1 (right) and BER=0 (left) ----
for BER in [1, 0]:

    ax = axes[subplot_index[BER]]

    if BER:
        search_pattern = r"SNR=(-?\d+)"
        ber_target = 0.01
        ylabel_cur = 'BER'
    else:
        search_pattern = r'_SNR=(-?\d+)_bler\.csv$'
        ber_target = 0.1
        ylabel_cur = 'BLER'

    snr_ber_dict = defaultdict(lambda: {
        'ber_1': [], 'ber_2': [], 'ber_3': [], 'ber_deeprx': [],
        'ber_deepsic_1': [], 'ber_deepsic_2': [], 'ber_deepsic_3': [],
        'ber_e2e_1': [], 'ber_e2e_2': [], 'ber_e2e_3': [],
        'ber_deepsicmb_1': [], 'ber_deepsicmb_2': [], 'ber_deepsicmb_3': [],
        'ber_deepstag_1': [], 'ber_deepstag_2': [], 'ber_deepstag_3': [],
        'ber_lmmse': [], 'ber_sphere': [],
        'ber_mhsa_1': [], 'ber_mhsa_2': [], 'ber_mhsa_3': []
    })

    plot_sphere = False

    # ---- Load CSVs for each seed ----
    for seed in seeds:
        seed_files = sorted(glob.glob(os.path.join(CSV_DIR, f"*seed={seed}*_SNR=*")))
        seen_snr = set()
        unique_files = []

        for file in seed_files:
            match = re.search(search_pattern, file)
            if match:
                snr = match.group(1)
                if snr not in seen_snr:
                    seen_snr.add(snr)
                    unique_files.append(file)

        for file in unique_files:
            match = re.search(search_pattern, file)
            if not match:
                continue
            snr = int(match.group(1))

            df = pd.read_csv(file)

            # ------------ BER parsing (same as your script) ------------
            snr_ber_dict[snr]['ber_1'].append(float(df["total_ber_1"]))
            snr_ber_dict[snr]['ber_2'].append(float(df["total_ber_2"]) if "total_ber_2" in df.columns else float(df["total_ber_1"]))
            snr_ber_dict[snr]['ber_3'].append(float(df["total_ber_3"]) if "total_ber_3" in df.columns else float(df["total_ber_1"]))

            # MHSA parsing
            if any(col.startswith("total_ber_mhsa") for col in df.columns):
                if "total_ber_mhsa" in df.columns and not any(col.startswith("total_ber_mhsa_") for col in df.columns):
                    val = float(df["total_ber_mhsa"])
                    for key in ["ber_mhsa_1","ber_mhsa_2","ber_mhsa_3"]:
                        snr_ber_dict[snr][key].append(val)
                else:
                    for k in [1,2,3]:
                        colname = f"total_ber_mhsa_{k}"
                        if colname in df.columns:
                            snr_ber_dict[snr][f"ber_mhsa_{k}"].append(float(df[colname]))
                        elif "total_ber_mhsa" in df.columns:
                            snr_ber_dict[snr][f"ber_mhsa_{k}"].append(float(df["total_ber_mhsa"]))

            # DeepSIC
            if any(col.startswith("total_ber_deepsic") for col in df.columns):
                if "total_ber_deepsic" in df.columns:
                    for k in [1, 2, 3]:
                        snr_ber_dict[snr][f"ber_deepsic_{k}"].append(float(df["total_ber_deepsic"]))
                else:
                    for k in [1,2,3]:
                        colname = f"total_ber_deepsic_{k}"
                        if colname in df.columns:
                            snr_ber_dict[snr][f"ber_deepsic_{k}"].append(float(df[colname]))
                        else:
                            snr_ber_dict[snr][f"ber_deepsic_{k}"].append(float(df["total_ber_deepsic_1"]))

            # DeepSIC-MB
            if any(col.startswith("total_ber_deepsicmb") for col in df.columns):
                if "total_ber_deepsicmb" in df.columns:
                    for k in [1,2,3]:
                        snr_ber_dict[snr][f"ber_deepsicmb_{k}"].append(float(df["total_ber_deepsicmb"]))
                else:
                    for k in [1,2,3]:
                        colname = f"total_ber_deepsicmb_{k}"
                        if colname in df.columns:
                            snr_ber_dict[snr][f"ber_deepsicmb_{k}"].append(float(df[colname]))
                        else:
                            snr_ber_dict[snr][f"ber_deepsicmb_{k}"].append(float(df["total_ber_deepsicmb_1"]))

            # DeepSTAG
            if any(col.startswith("total_ber_deepstag") for col in df.columns):
                if "total_ber_deepstag" in df.columns:
                    for k in [1,2,3]:
                        snr_ber_dict[snr][f"ber_deepstag_{k}"].append(float(df["total_ber_deepstag"]))
                else:
                    for k in [1,2,3]:
                        colname = f"total_ber_deepstag_{k}"
                        if colname in df.columns:
                            snr_ber_dict[snr][f"ber_deepstag_{k}"].append(float(df[colname]))
                        else:
                            snr_ber_dict[snr][f"ber_deepstag_{k}"].append(float(df["total_ber_deepstag_1"]))

            # DeepRx
            if "total_ber_deeprx" in df.columns:
                val = str(df["total_ber_deeprx"].iloc[0]).replace("tensor(","").replace(")","")
                snr_ber_dict[snr]['ber_deeprx'].append(float(val))

            # LMMSE
            val = str(df["total_ber_lmmse"].iloc[0]).replace("tensor(","").replace(")","")
            snr_ber_dict[snr]['ber_lmmse'].append(float(val))

            # Sphere
            if "total_ber_sphere" in df.columns:
                val = str(df["total_ber_sphere"].iloc[0]).replace("tensor(","").replace(")","")
                snr_ber_dict[snr]['ber_sphere'].append(float(val))

    # ---- Averages ----
    snrs = sorted(snr_ber_dict.keys())
    def avg(key): return [np.mean(snr_ber_dict[s][key]) for s in snrs]

    ber_1 = avg('ber_1')
    ber_2 = avg('ber_2')
    ber_3 = avg('ber_3')
    ber_mhsa_1 = avg('ber_mhsa_1')
    ber_mhsa_2 = avg('ber_mhsa_2')
    ber_mhsa_3 = avg('ber_mhsa_3')
    ber_lmmse = avg('ber_lmmse')
    ber_sphere = avg('ber_sphere')
    ber_deeprx = avg('ber_deeprx')

    markers = ['o', '*', 'x', 'D', '+']
    dashes  = [':', '-.', '--', '-', '-']

    # ---- Plot ESCNN1/2/3 ----
    interp_func = interp1d(ber_1, snrs, fill_value="extrapolate")
    snr_target_1 = np.round(interp_func(ber_target), 1)
    ax.semilogy(snrs, ber_1, linestyle=dashes[0], marker=markers[0], color='g',
                label=f'ESCNN1 @ {round(100*ber_target)}% = {snr_target_1}')

    interp_func = interp1d(ber_2, snrs, fill_value="extrapolate")
    snr_target_2 = np.round(interp_func(ber_target), 1)
    ax.semilogy(snrs, ber_2, linestyle=dashes[1], marker=markers[1], color='g',
                label=f'ESCNN2 @ {round(100*ber_target)}% = {snr_target_2}')

    interp_func = interp1d(ber_3, snrs, fill_value="extrapolate")
    snr_target_3 = np.round(interp_func(ber_target), 1)
    ax.semilogy(snrs, ber_3, linestyle=dashes[2], marker=markers[2], color='g',
                label=f'ESCNN3 @ {round(100*ber_target)}% = {snr_target_3}')

    # ---- LMMSE ----
    interp_func = interp1d(ber_lmmse, snrs, fill_value="extrapolate")
    snr_target_lmmse = np.round(interp_func(ber_target), 1)
    ax.semilogy(snrs, ber_lmmse, linestyle=dashes[4], marker=markers[4], color='r',
                label=f'LMMSE @ {round(100*ber_target)}% = {snr_target_lmmse}')

    # ---- Sphere only for BER (right panel) ----
    if BER == 1 and not (np.unique(ber_sphere).shape[0] == 1):
        interp_func = interp1d(ber_sphere, snrs, fill_value="extrapolate")
        snr_target_sphere = np.round(interp_func(ber_target), 1)
        ax.semilogy(snrs, ber_sphere, linestyle=dashes[4], marker=markers[4], color='brown',
                    label=f'Sphere @ {round(100*ber_target)}% = {snr_target_sphere}')

    # ---- Formatting ----
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel_cur)
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

# ---- Add global title centered above BOTH plots ----
fig.suptitle(global_title_text, fontsize=12, y=1.03)

plt.tight_layout()
plt.show()
