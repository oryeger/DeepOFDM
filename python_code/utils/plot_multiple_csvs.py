import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.interpolate import interp1d
import numpy as np
from collections import defaultdict

# 🔧 Adjust path as needed
CSV_DIR = r"C:\Projects\Scratchpad"
seeds = [123, 17, 41, 58]
# seeds = [58]
ber_target = 0.01

# Step 1: Collect and aggregate data
snr_ber_dict = defaultdict(lambda: {
    'ber_1': [], 'ber_2': [], 'ber_3': [], 'ber_deeprx': [], 'ber_deepsicsb_1': [], 'ber_deepsicsb_2': [], 'ber_deepsicsb_3': [],
    'ber_deepsicmb_1': [], 'ber_deepsicmb_2': [], 'ber_deepsicmb_3': [], 'ber_deepstag_1': [], 'ber_deepstag_2': [], 'ber_deepstag_3': [],
    'ber_legacy': [], 'ber_sphere': []
})

for seed in seeds:
    seed_files = sorted(glob.glob(os.path.join(CSV_DIR, f"*seed={seed}*_SNR=*")))

    seen_snr = set()
    unique_files = []
    for file in seed_files:
        match = re.search(r'_SNR=(\d+)\.csv$', file)
        if match:
            snr = match.group(1)
            if snr not in seen_snr:
                seen_snr.add(snr)
                unique_files.append(file)

    for file in unique_files:

        match = re.search(r"SNR=(\d+)", file)
        if not match:
            continue
        snr = int(match.group(1))

        try:
            df = pd.read_csv(file)
            snr_ber_dict[snr]['ber_1'].append(float(df["total_ber_1"]))
            if "total_ber_2" in df.columns:
                snr_ber_dict[snr]['ber_2'].append(float(df["total_ber_2"]))
            else:
                snr_ber_dict[snr]['ber_2'].append(float(df["total_ber_1"]))
            if "total_ber_3" in df.columns:
                snr_ber_dict[snr]['ber_3'].append(float(df["total_ber_3"]))
            else:
                snr_ber_dict[snr]['ber_3'].append(float(df["total_ber_1"]))

            if "total_ber_deepsicsb" in df.columns:
                snr_ber_dict[snr]['ber_deepsicsb_1'].append(float(df["total_ber_deepsicsb"]))
                snr_ber_dict[snr]['ber_deepsicsb_2'].append(float(df["total_ber_deepsicsb"]))
                snr_ber_dict[snr]['ber_deepsicsb_3'].append(float(df["total_ber_deepsicsb"]))
            else:
                snr_ber_dict[snr]['ber_deepsicsb_1'].append(float(df["total_ber_deepsicsb_1"]))
                if "total_ber_deepsicsb_2" in df.columns:
                    snr_ber_dict[snr]['ber_deepsicsb_2'].append(float(df["total_ber_deepsicsb_2"]))
                else:
                    snr_ber_dict[snr]['ber_deepsicsb_2'].append(float(df["total_ber_deepsicsb_1"]))
                if "total_ber_deepsicsb_3" in df.columns:
                    snr_ber_dict[snr]['ber_deepsicsb_3'].append(float(df["total_ber_deepsicsb_3"]))
                else:
                    snr_ber_dict[snr]['ber_deepsicsb_3'].append(float(df["total_ber_deepsicsb_1"]))

            if "total_ber_deepsicmb" in df.columns:
                snr_ber_dict[snr]['ber_deepsicmb_1'].append(float(df["total_ber_deepsicmb"]))
                snr_ber_dict[snr]['ber_deepsicmb_2'].append(float(df["total_ber_deepsicmb"]))
                snr_ber_dict[snr]['ber_deepsicmb_3'].append(float(df["total_ber_deepsicmb"]))
            else:
                snr_ber_dict[snr]['ber_deepsicmb_1'].append(float(df["total_ber_deepsicmb_1"]))
                if "total_ber_deepsicmb_2" in df.columns:
                    snr_ber_dict[snr]['ber_deepsicmb_2'].append(float(df["total_ber_deepsicmb_2"]))
                else:
                    snr_ber_dict[snr]['ber_deepsicmb_2'].append(float(df["total_ber_deepsicmb_1"]))
                if "total_ber_deepsicmb_3" in df.columns:
                    snr_ber_dict[snr]['ber_deepsicmb_3'].append(float(df["total_ber_deepsicmb_3"]))
                else:
                    snr_ber_dict[snr]['ber_deepsicmb_3'].append(float(df["total_ber_deepsicmb_1"]))

            if "total_ber_deepstag" in df.columns:
                snr_ber_dict[snr]['ber_deepstag_1'].append(float(df["total_ber_deepstag"]))
                snr_ber_dict[snr]['ber_deepstag_2'].append(float(df["total_ber_deepstag"]))
                snr_ber_dict[snr]['ber_deepstag_3'].append(float(df["total_ber_deepstag"]))
            else:
                snr_ber_dict[snr]['ber_deepstag_1'].append(float(df["total_ber_deepstag_1"]))
                if "total_ber_deepstag_2" in df.columns:
                    snr_ber_dict[snr]['ber_deepstag_2'].append(float(df["total_ber_deepstag_2"]))
                else:
                    snr_ber_dict[snr]['ber_deepstag_2'].append(float(df["total_ber_deepstag_1"]))
                if "total_ber_deepstag_3" in df.columns:
                    snr_ber_dict[snr]['ber_deepstag_3'].append(float(df["total_ber_deepstag_3"]))
                else:
                    snr_ber_dict[snr]['ber_deepstag_3'].append(float(df["total_ber_deepstag_1"]))





            ber_deeprx_val = str(df["total_ber_deeprx"].iloc[0]).replace("tensor(", "").replace(")", "")
            snr_ber_dict[snr]['ber_deeprx'].append(float(ber_deeprx_val))
            ber_legacy_val = str(df["total_ber_legacy"].iloc[0]).replace("tensor(", "").replace(")", "")
            snr_ber_dict[snr]['ber_legacy'].append(float(ber_legacy_val))
            ber_sphere_val = str(df["total_ber_sphere"].iloc[0]).replace("tensor(", "").replace(")", "")
            snr_ber_dict[snr]['ber_sphere'].append(float(ber_sphere_val))
        except Exception as e:
            print(f"Error processing file {file}: {e}")

# Step 2: Sort SNRs and compute averages
snrs = sorted(snr_ber_dict.keys())
ber_1 = [np.mean(snr_ber_dict[snr]['ber_1']) for snr in snrs]
ber_2 = [np.mean(snr_ber_dict[snr]['ber_2']) for snr in snrs]
ber_3 = [np.mean(snr_ber_dict[snr]['ber_3']) for snr in snrs]
ber_deepsicsb_1 = [np.mean(snr_ber_dict[snr]['ber_deepsicsb_1']) for snr in snrs]
ber_deepsicsb_2 = [np.mean(snr_ber_dict[snr]['ber_deepsicsb_2']) for snr in snrs]
ber_deepsicsb_3 = [np.mean(snr_ber_dict[snr]['ber_deepsicsb_3']) for snr in snrs]
ber_deepsicmb_1 = [np.mean(snr_ber_dict[snr]['ber_deepsicmb_1']) for snr in snrs]
ber_deepsicmb_2 = [np.mean(snr_ber_dict[snr]['ber_deepsicmb_2']) for snr in snrs]
ber_deepsicmb_3 = [np.mean(snr_ber_dict[snr]['ber_deepsicmb_3']) for snr in snrs]
ber_deepstag_1 = [np.mean(snr_ber_dict[snr]['ber_deepstag_1']) for snr in snrs]
ber_deepstag_2 = [np.mean(snr_ber_dict[snr]['ber_deepstag_2']) for snr in snrs]
ber_deepstag_3 = [np.mean(snr_ber_dict[snr]['ber_deepstag_3']) for snr in snrs]
ber_deeprx = [np.mean(snr_ber_dict[snr]['ber_deeprx']) for snr in snrs]

ber_legacy = [np.mean(snr_ber_dict[snr]['ber_legacy']) for snr in snrs]
ber_sphere = [np.mean(snr_ber_dict[snr]['ber_sphere']) for snr in snrs]


# Step 3: Plotting
plt.figure(figsize=(10, 6))
markers = ['o', '*', 'x', 'D', '+']
dashes = [':', '-.', '--', '-', '-']

interp_func = interp1d(ber_1, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_1, linestyle=dashes[0], marker=markers[0], color='g',
             label='DeepSIC1, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_2, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_2, linestyle=dashes[1], marker=markers[1], color='g',
             label='DeepSIC2, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_3, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_3, linestyle=dashes[2], marker=markers[2], color='g',
             label='DeepSIC3, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

# Uncomment below to plot DeepRx
if np.unique(ber_deeprx).shape[0] != 1:
    interp_func = interp1d(ber_deeprx, snrs, kind='linear', fill_value="extrapolate")
    plt.semilogy(snrs, ber_deeprx, linestyle=dashes[3], marker=markers[3], color='c',
                 label='DeepRx, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_deepsicsb_1, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_deepsicsb_1, linestyle=dashes[0], marker=markers[0], color='orange',
             label='DeepSICSB1, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_deepsicsb_2, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_deepsicsb_2, linestyle=dashes[1], marker=markers[1], color='orange',
             label='DeepSICSB2, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_deepsicsb_3, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_deepsicsb_3, linestyle=dashes[2], marker=markers[2], color='orange',
             label='DeepSICSB3, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_deepsicmb_1, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_deepsicmb_1, linestyle=dashes[0], marker=markers[0], color='black',
             label='DeepSICMB1, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_deepsicmb_2, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_deepsicmb_2, linestyle=dashes[1], marker=markers[1], color='black',
             label='DeepSICMB2, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_deepsicmb_3, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_deepsicmb_3, linestyle=dashes[2], marker=markers[2], color='black',
             label='DeepSICMB3, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_deepstag_1, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_deepstag_1, linestyle=dashes[0], marker=markers[0], color='pink',
             label='DeepSTAG1, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_deepstag_2, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_deepstag_2, linestyle=dashes[1], marker=markers[1], color='pink',
             label='DeepSTAG2, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_deepstag_3, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_deepstag_3, linestyle=dashes[2], marker=markers[2], color='pink',
             label='DeepSTAG3, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))



interp_func = interp1d(ber_legacy, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_legacy, linestyle=dashes[4], marker=markers[4], color='r',
             label='Legacy, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

interp_func = interp1d(ber_sphere, snrs, kind='linear', fill_value="extrapolate")
plt.semilogy(snrs, ber_sphere, linestyle=dashes[4], marker=markers[4], color='brown',
             label='Sphere, SNR @1%=' + str(np.round(interp_func(ber_target), 1)))

plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.yscale("log")
plt.grid(True)
plt.legend()

# Optional: Create a title from one of the filenames
example_file = sorted(glob.glob(os.path.join(CSV_DIR, "*seed="+str(seeds[0])+"*_SNR=*")))[0]
original_name = os.path.basename(example_file)
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
cleaned_name = "\n".join([cleaned_name[i:i+80] for i in range(0, len(cleaned_name), 80)])
cleaned_name = cleaned_name.rstrip(", ")
plt.title("Averaged across seeds: " + ", ".join(map(str, seeds)) + "\n" + cleaned_name)
plt.tight_layout()
plt.show()
