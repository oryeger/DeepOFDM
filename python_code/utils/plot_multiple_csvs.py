import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.interpolate import interp1d
import numpy as np

# 🔧 Adjust path if needed
CSV_DIR = r"C:\Projects\Scratchpad\16QAM_0"

# Step 1: Collect matching files
files = sorted(glob.glob(os.path.join(CSV_DIR, "*_SNR=*")))

snrs = []
ber_1 = []
ber_2 = []
ber_3 = []
ber_deeprx = []
ber_legacy = []

# Step 2: Parse each CSV and extract values
for file in files:
    match = re.search(r"SNR=(\d+)", file)
    if not match:
        continue
    snr = int(match.group(1))

    df = pd.read_csv(file)
    snrs.append(snr)
    ber_1.append(float(df["total_ber_1"]))
    ber_2.append(float(df["total_ber_2"]))
    ber_3.append(float(df["total_ber_3"]))
    ber_deeprx.append(float(df["total_ber_deeprx"]))
    ber_legacy.append(float(str(df["total_ber_legacy"].iloc[0]).replace("tensor(", "").replace(")", "")))

# Step 3: Sort by SNR
sorted_idx = sorted(range(len(snrs)), key=lambda i: snrs[i])
snrs = [snrs[i] for i in sorted_idx]
ber_1 = [ber_1[i] for i in sorted_idx]
ber_2 = [ber_2[i] for i in sorted_idx]
ber_3 = [ber_3[i] for i in sorted_idx]
ber_deeprx = [ber_deeprx[i] for i in sorted_idx]
ber_legacy = [ber_legacy[i] for i in sorted_idx]

markers = ['o', '*', 'x', 'D', '+', 'o']
dashes = [':', '-.', '--', '-', '-', '-']


ber_target = 0.01

# Step 4: Plot
plt.figure(figsize=(10, 6))
interp_func = interp1d(ber_1, snrs, kind='linear', fill_value="extrapolate")
snr_at_target = np.round(interp_func(ber_target), 1)
plt.semilogy(snrs, ber_1, linestyle=dashes[0], marker=markers[0],color='g', label='DeepSIC1' + ', SNR @1%=' + str(snr_at_target))
interp_func = interp1d(ber_2, snrs, kind='linear', fill_value="extrapolate")
snr_at_target = np.round(interp_func(ber_target), 1)
plt.semilogy(snrs, ber_2, linestyle=dashes[1], marker=markers[1],color='g', label='DeepSIC2' + ', SNR @1%=' + str(snr_at_target))
interp_func = interp1d(ber_3, snrs, kind='linear', fill_value="extrapolate")
snr_at_target = np.round(interp_func(ber_target), 1)
plt.semilogy(snrs, ber_3, linestyle=dashes[2], marker=markers[2],color='g', label='DeepSIC3' + ', SNR @1%=' + str(snr_at_target))
# interp_func = interp1d(ber_deeprx, snrs, kind='linear', fill_value="extrapolate")
# snr_at_target = np.round(interp_func(ber_target), 1)
# plt.semilogy(snrs, ber_deeprx, '-o', color='c', label='DeepRx,   SNR @1%=' + str(snr_at_target))
interp_func = interp1d(ber_legacy, snrs, kind='linear', fill_value="extrapolate")
snr_at_target = np.round(interp_func(ber_target), 1)
plt.semilogy(snrs, ber_legacy, '-o', color='r', label='Legacy,    SNR @1%=' + str(snr_at_target))

plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.yscale("log")
plt.grid(True)
plt.legend()

# Step 5: Generate cleaned title
original_name = os.path.basename(files[0])
# Remove timestamp prefix (e.g., 2025_06_05_23_26_)
cleaned_name = re.sub(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_", "", original_name)
# Remove _SNR=... at the end
cleaned_name = re.sub(".csv", "", cleaned_name)
cleaned_name = re.sub(r"_SNR=\d+$", "", cleaned_name)

cleaned_name = re.sub("_scs", "scs", cleaned_name)
cleaned_name = re.sub("cfo_in_Rx", "cfo", cleaned_name)
cleaned_name = re.sub("_", ", ", cleaned_name)
# Add line breaks every 80 characters
cleaned_name = "\n".join([cleaned_name[i:i+80] for i in range(0, len(cleaned_name), 80)])

plt.title(cleaned_name)
plt.tight_layout()
plt.show()
