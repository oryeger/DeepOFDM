import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re

# 🔧 Adjust path if needed
CSV_DIR = "C:\Projects\Scratchpad"

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

# Step 4: Plot
plt.figure(figsize=(10, 6))
plt.semilogy(snrs, ber_1, marker='o', label="total_ber_1")
plt.semilogy(snrs, ber_2, marker='s', label="total_ber_2")
plt.semilogy(snrs, ber_3, marker='^', label="total_ber_3")
plt.semilogy(snrs, ber_deeprx, marker='D', label="total_ber_deeprx")
plt.semilogy(snrs, ber_legacy, marker='x', label="total_ber_legacy")

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
