import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.interpolate import interp1d
import numpy as np
from collections import defaultdict
from scipy.io import savemat

# 🔧 Adjust path as needed
CSV_DIR = r"C:\Projects\Scratchpad"
seeds = [123, 17, 41, 58]
# seeds = [41, 58]

# BER = 0 # Set to False if you want to plot SNR instead of BER
plot_sphere = False

for BER in [1, 0]:
    if BER:
        search_pattern = r"SNR=(-?\d+)"
        ber_target = 0.01
        ylabel_cur = 'BER'
    else:
        search_pattern = r'_SNR=(-?\d+)_bler\.csv$'
        ber_target = 0.1
        ylabel_cur = 'BLER'

    snr_ber_dict = defaultdict(lambda: {
        'ber_1': [], 'ber_2': [], 'ber_3': [], 'ber_deeprx': [], 'ber_deepsic_1': [], 'ber_deepsic_2': [], 'ber_deepsic_3': [],
        'ber_e2e_1': [], 'ber_e2e_2': [], 'ber_e2e_3': [], 'ber_deepsicmb_1': [], 'ber_deepsicmb_2': [], 'ber_deepsicmb_3': [],
        'ber_deepstag_1': [], 'ber_deepstag_2': [], 'ber_deepstag_3': [], 'ber_lmmse': [], 'ber_sphere': []
    })


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
            snr_ber_dict[snr]['ber_1'].append(float(df["total_ber_1"]))
            if "total_ber_2" in df.columns:
                snr_ber_dict[snr]['ber_2'].append(float(df["total_ber_2"]))
            else:
                snr_ber_dict[snr]['ber_2'].append(float(df["total_ber_1"]))
            if "total_ber_3" in df.columns:
                snr_ber_dict[snr]['ber_3'].append(float(df["total_ber_3"]))
            else:
                snr_ber_dict[snr]['ber_3'].append(float(df["total_ber_1"]))

            if any(col.startswith("total_ber_deepsic") for col in df.columns):
                if "total_ber_deepsic" in df.columns:
                    snr_ber_dict[snr]['ber_deepsic_1'].append(float(df["total_ber_deepsic"]))
                    snr_ber_dict[snr]['ber_deepsic_2'].append(float(df["total_ber_deepsic"]))
                    snr_ber_dict[snr]['ber_deepsic_3'].append(float(df["total_ber_deepsic"]))
                else:
                    snr_ber_dict[snr]['ber_deepsic_1'].append(float(df["total_ber_deepsic_1"]))
                    if "total_ber_deepsic_2" in df.columns:
                        snr_ber_dict[snr]['ber_deepsic_2'].append(float(df["total_ber_deepsic_2"]))
                    else:
                        snr_ber_dict[snr]['ber_deepsic_2'].append(float(df["total_ber_deepsic_1"]))
                    if "total_ber_deepsic_3" in df.columns:
                        snr_ber_dict[snr]['ber_deepsic_3'].append(float(df["total_ber_deepsic_3"]))
                    else:
                        snr_ber_dict[snr]['ber_deepsic_3'].append(float(df["total_ber_deepsic_1"]))

            if any(col.startswith("total_ber_deepsicmb") for col in df.columns):
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

            if any(col.startswith("total_ber_deepstag") for col in df.columns):
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

            if any(col.startswith("total_ber_e2e") for col in df.columns):
                snr_ber_dict[snr]['ber_e2e_1'].append(float(df["total_ber_e2e_1"]))
                if "total_ber_e2e_2" in df.columns:
                    snr_ber_dict[snr]['ber_e2e_2'].append(float(df["total_ber_e2e_2"]))
                else:
                    snr_ber_dict[snr]['ber_e2e_2'].append(float(df["total_ber_e2e_1"]))
                if "total_ber_e2e_3" in df.columns:
                    snr_ber_dict[snr]['ber_e2e_3'].append(float(df["total_ber_e2e_3"]))
                else:
                    snr_ber_dict[snr]['ber_e2e_3'].append(float(df["total_ber_e2e_1"]))

            if any(col.startswith("total_ber_deeprx") for col in df.columns):
                ber_deeprx_val = str(df["total_ber_deeprx"].iloc[0]).replace("tensor(", "").replace(")", "")
                snr_ber_dict[snr]['ber_deeprx'].append(float(ber_deeprx_val))
            ber_lmmse_val = str(df["total_ber_lmmse"].iloc[0]).replace("tensor(", "").replace(")", "")
            snr_ber_dict[snr]['ber_lmmse'].append(float(ber_lmmse_val))
            if any(col.startswith("total_ber_sphere") for col in df.columns):
                ber_sphere_val = str(df["total_ber_sphere"].iloc[0]).replace("tensor(", "").replace(")", "")
                snr_ber_dict[snr]['ber_sphere'].append(float(ber_sphere_val))

    # Step 2: Sort SNRs and compute averages
    snrs = sorted(snr_ber_dict.keys())
    ber_1 = [np.mean(snr_ber_dict[snr]['ber_1']) for snr in snrs]
    ber_2 = [np.mean(snr_ber_dict[snr]['ber_2']) for snr in snrs]
    ber_3 = [np.mean(snr_ber_dict[snr]['ber_3']) for snr in snrs]
    ber_deepsic_1 = [np.mean(snr_ber_dict[snr]['ber_deepsic_1']) for snr in snrs]
    ber_deepsic_2 = [np.mean(snr_ber_dict[snr]['ber_deepsic_2']) for snr in snrs]
    ber_deepsic_3 = [np.mean(snr_ber_dict[snr]['ber_deepsic_3']) for snr in snrs]
    ber_deepsicmb_1 = [np.mean(snr_ber_dict[snr]['ber_deepsicmb_1']) for snr in snrs]
    ber_deepsicmb_2 = [np.mean(snr_ber_dict[snr]['ber_deepsicmb_2']) for snr in snrs]
    ber_deepsicmb_3 = [np.mean(snr_ber_dict[snr]['ber_deepsicmb_3']) for snr in snrs]
    ber_deepstag_1 = [np.mean(snr_ber_dict[snr]['ber_deepstag_1']) for snr in snrs]
    ber_deepstag_2 = [np.mean(snr_ber_dict[snr]['ber_deepstag_2']) for snr in snrs]
    ber_deepstag_3 = [np.mean(snr_ber_dict[snr]['ber_deepstag_3']) for snr in snrs]
    ber_deeprx = [np.mean(snr_ber_dict[snr]['ber_deeprx']) for snr in snrs]
    ber_e2e_1 = [np.mean(snr_ber_dict[snr]['ber_e2e_1']) for snr in snrs]
    ber_e2e_2 = [np.mean(snr_ber_dict[snr]['ber_e2e_2']) for snr in snrs]
    ber_e2e_3 = [np.mean(snr_ber_dict[snr]['ber_e2e_3']) for snr in snrs]


    ber_lmmse = [np.mean(snr_ber_dict[snr]['ber_lmmse']) for snr in snrs]
    ber_sphere = [np.mean(snr_ber_dict[snr]['ber_sphere']) for snr in snrs]


    # Step 3: Plotting
    plt.figure(figsize=(10, 6))
    markers = ['o', '*', 'x', 'D', '+']
    dashes = [':', '-.', '--', '-', '-']

    interp_func = interp1d(ber_1, snrs, kind='linear', fill_value="extrapolate")
    snr_target_1 = np.round(interp_func(ber_target), 1)
    plt.semilogy(snrs, ber_1, linestyle=dashes[0], marker=markers[0], color='g',
                 label='ESCNN1, SNR @'+str(round(100*ber_target))+'%=' + str(snr_target_1))

    interp_func = interp1d(ber_2, snrs, kind='linear', fill_value="extrapolate")
    snr_target_2 = np.round(interp_func(ber_target), 1)
    plt.semilogy(snrs, ber_2, linestyle=dashes[1], marker=markers[1], color='g',
                 label='ESCNN2, SNR @'+str(round(100*ber_target))+'%=' + str(snr_target_2))

    interp_func = interp1d(ber_3, snrs, kind='linear', fill_value="extrapolate")
    snr_target_3 = np.round(interp_func(ber_target), 1)
    plt.semilogy(snrs, ber_3, linestyle=dashes[2], marker=markers[2], color='g',
                 label='ESCNN3, SNR @'+str(round(100*ber_target))+'%=' + str(snr_target_3))

    if np.unique(ber_deeprx).shape[0] != 1:
        interp_func = interp1d(ber_deeprx, snrs, kind='linear', fill_value="extrapolate")
        snr_target_deeprx = np.round(interp_func(ber_target), 1)
        plt.semilogy(snrs, ber_deeprx, linestyle=dashes[3], marker=markers[3], color='c',
                     label='DeepRx, SNR @'+str(round(100*ber_target))+'%=' + str(snr_target_deeprx))

    if any(col.startswith("total_ber_deepsic") for col in df.columns):
        interp_func = interp1d(ber_deepsic_1, snrs, kind='linear', fill_value="extrapolate")
        snr_target_deepsic = np.round(interp_func(ber_target), 1)
        plt.semilogy(snrs, ber_deepsic_1, linestyle=dashes[0], marker=markers[0], color='orange',
                     label='DeepSIC1, SNR @'+str(round(100*ber_target))+'%=' + str(snr_target_deepsic))
        ber_deepsic = ber_deepsic_1

        interp_func = interp1d(ber_deepsic_2, snrs, kind='linear', fill_value="extrapolate")
        plt.semilogy(snrs, ber_deepsic_2, linestyle=dashes[1], marker=markers[1], color='orange',
                     label='DeepSIC2, SNR @'+str(round(100*ber_target))+'%=' + str(np.round(interp_func(ber_target), 1)))

        interp_func = interp1d(ber_deepsic_3, snrs, kind='linear', fill_value="extrapolate")
        plt.semilogy(snrs, ber_deepsic_3, linestyle=dashes[2], marker=markers[2], color='orange',
                     label='DeepSIC3, SNR @'+str(round(100*ber_target))+'%=' + str(np.round(interp_func(ber_target), 1)))

    if any(col.startswith("total_ber_deepsicmb") for col in df.columns):
        interp_func = interp1d(ber_deepsicmb_1, snrs, kind='linear', fill_value="extrapolate")
        plt.semilogy(snrs, ber_deepsicmb_1, linestyle=dashes[0], marker=markers[0], color='black',
                     label='DeepSICMB1, SNR @'+str(round(100*ber_target))+'%=' + str(np.round(interp_func(ber_target), 1)))

        interp_func = interp1d(ber_deepsicmb_2, snrs, kind='linear', fill_value="extrapolate")
        plt.semilogy(snrs, ber_deepsicmb_2, linestyle=dashes[1], marker=markers[1], color='black',
                     label='DeepSICMB2, SNR @'+str(round(100*ber_target))+'%=' + str(np.round(interp_func(ber_target), 1)))

        interp_func = interp1d(ber_deepsicmb_3, snrs, kind='linear', fill_value="extrapolate")
        plt.semilogy(snrs, ber_deepsicmb_3, linestyle=dashes[2], marker=markers[2], color='black',
                     label='DeepSICMB3, SNR @'+str(round(100*ber_target))+'%=' + str(np.round(interp_func(ber_target), 1)))

    if any(col.startswith("total_ber_deepstag") for col in df.columns):
        interp_func = interp1d(ber_deepstag_1, snrs, kind='linear', fill_value="extrapolate")
        plt.semilogy(snrs, ber_deepstag_1, linestyle=dashes[0], marker=markers[0], color='pink',
                     label='DeepSTAG1, SNR @'+str(round(100*ber_target))+'%=' + str(np.round(interp_func(ber_target), 1)))

        interp_func = interp1d(ber_deepstag_2, snrs, kind='linear', fill_value="extrapolate")
        plt.semilogy(snrs, ber_deepstag_2, linestyle=dashes[1], marker=markers[1], color='pink',
                     label='DeepSTAG2, SNR @'+str(round(100*ber_target))+'%=' + str(np.round(interp_func(ber_target), 1)))

        interp_func = interp1d(ber_deepstag_3, snrs, kind='linear', fill_value="extrapolate")
        plt.semilogy(snrs, ber_deepstag_3, linestyle=dashes[2], marker=markers[2], color='pink',
                     label='DeepSTAG3, SNR @'+str(round(100*ber_target))+'%=' + str(np.round(interp_func(ber_target), 1)))


    if any(col.startswith("total_ber_e2e") for col in df.columns) and np.unique(ber_deeprx).shape[0] != 1:
        interp_func = interp1d(ber_e2e_1, snrs, kind='linear', fill_value="extrapolate")
        plt.semilogy(snrs, ber_e2e_1, linestyle=dashes[0], marker=markers[0], color='magenta',
                     label='DeepSICe2e1, SNR @'+str(round(100*ber_target))+'%=' + str(np.round(interp_func(ber_target), 1)))

        interp_func = interp1d(ber_e2e_2, snrs, kind='linear', fill_value="extrapolate")
        plt.semilogy(snrs, ber_e2e_2, linestyle=dashes[1], marker=markers[1], color='magenta',
                     label='DeepSICe2e2, SNR @'+str(round(100*ber_target))+'%=' + str(np.round(interp_func(ber_target), 1)))

        interp_func = interp1d(ber_e2e_3, snrs, kind='linear', fill_value="extrapolate")
        plt.semilogy(snrs, ber_e2e_3, linestyle=dashes[2], marker=markers[2], color='magenta',
                     label='DeepSICe2e3, SNR @'+str(round(100*ber_target))+'%=' + str(np.round(interp_func(ber_target), 1)))


    interp_func = interp1d(ber_lmmse, snrs, kind='linear', fill_value="extrapolate")
    snr_target_lmmse = np.round(interp_func(ber_target), 1)
    plt.semilogy(snrs, ber_lmmse, linestyle=dashes[4], marker=markers[4], color='r',
                 label='lmmse, SNR @'+str(round(100*ber_target))+'%=' + str(snr_target_lmmse))

    if not (np.unique(ber_sphere).shape[0] == 1) and (BER == 1):
        plot_sphere = True

    if plot_sphere:
        interp_func = interp1d(ber_sphere, snrs, kind='linear', fill_value="extrapolate")
        snr_target_sphere = np.round(interp_func(ber_target), 1)
        plt.semilogy(snrs, ber_sphere, linestyle=dashes[4], marker=markers[4], color='brown',
                     label='Sphere, SNR @'+str(round(100*ber_target))+'%=' + str(snr_target_sphere))

    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel_cur)
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


relevant = 'lmmse'
snr_target_no_aug = globals()['snr_target_' + relevant]
ber_no_aug =  globals()['ber_' + relevant]

data = {
    'snrs': snrs,
    'snr_target_no_aug': snr_target_no_aug,
    'snr_target_aug_1': snr_target_1,
    'snr_target_aug_2': snr_target_2,
    'snr_target_aug_3': snr_target_3,
    'bler_no_aug': ber_no_aug,
    'bler_aug_1': ber_1,
    'bler_aug_2': ber_2,
    'bler_aug_3': ber_3
}
project_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))  # Goes from utils -> python_code -> DeepOFDM
output_dir = os.path.join(project_dir, 'Scratchpad', 'mat_files')
file_path = os.path.abspath(os.path.join(output_dir, relevant) + ".mat")
savemat(file_path,data)