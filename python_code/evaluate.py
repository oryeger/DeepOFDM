import os
from pathlib import Path
import time
from python_code.detectors.escnn.escnn_trainer import ESCNNTrainer
from python_code.detectors.deepsice2e.deepsice2e_trainer import DeepSICe2eTrainer
from python_code.detectors.deeprx.deeprx_trainer import DeepRxTrainer
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.detectors.deepsicmb.deepsicmb_trainer import DeepSICMBTrainer
from python_code.detectors.deepstag.deepstag_trainer import DeepSTAGTrainer
from python_code.detectors.mhsa.mhsa_trainer import MHSATrainer
from python_code.detectors.tdcnn.tdcnn_trainer import TDCNNTrainer
from python_code.detectors.tdfdcnn.tdfdcnn_trainer import TDFDCNNTrainer
from python_code import DEVICE


from typing import List

import torch
from python_code import conf
from python_code.utils.metrics import calculate_ber
import matplotlib.pyplot as plt
from python_code.utils.probs_utils import relevant_indices
from python_code.utils.probs_utils import skip_indices

from python_code.utils.constants import (TRAIN_PERCENTAGE, GENIE_CFO,
                                         FFT_size, FIRST_CP, CP, NUM_SYMB_PER_SLOT, NUM_SAMPLES_PER_SLOT, PLOT_MI)

import pandas as pd

from python_code.channel.channel_dataset import ChannelModelDataset
from scipy.stats import entropy
from scipy.interpolate import interp1d

from python_code.detectors.sphere.sphere_decoder import SphereDecoder
from python_code.detectors.lmmse.lmmse_equalizer import LmmseEqualize,LmmseDemod

from datetime import datetime
from scipy.io import savemat

import argparse

from python_code.coding.mcs_table import get_mcs

from python_code.coding.ldpc_wrapper import LDPC5GCodec
from python_code.coding.crc_wrapper import CRC5GCodec

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

path = Path(__file__).resolve()
projects_root = path.parents[2]
base_dir = projects_root / "Scratchpad"
base_dir.mkdir(parents=True, exist_ok=True)
pass


import numpy as np


def entropy_with_bin_width(data, bin_width):
    """Estimate entropy using histogram binning with a specified bin width."""
    min_x, max_x = np.min(data), np.max(data)
    bins = np.arange(min_x, max_x + bin_width, bin_width)  # Define bin edges
    hist, _ = np.histogram(data, bins=bins, density=True)  # Compute histogram
    hist = hist[hist > 0]  # Remove zero probabilities
    return entropy(hist, base=2)  # Compute entropy


def calc_mi(tx_data: np.ndarray, llrs_mat: np.ndarray, num_bits_data: int, n_users: int, num_res: int) -> np.ndarray:
    llr_1 = llrs_mat[:, :, :, :].squeeze(-1)
    llr_2 = llr_1.reshape(int(tx_data.shape[0] / num_bits_data), n_users, num_bits_data, num_res)
    llr_3 = llr_2.swapaxes(1, 2)
    llr_4 = llr_3.reshape(tx_data.shape[0], n_users, num_res)
    llr_for_mi = llr_4.flatten()
    tx_data_for_mi = tx_data.flatten()

    if (llr_for_mi.shape[0] > 50000) & (tx_data_for_mi.shape[0] > 50000):
        tx_data_for_mi = tx_data_for_mi[:50000]
        llr_for_mi = llr_for_mi[:50000]

    # H_y calculation
    H_y = entropy_with_bin_width(llr_for_mi.numpy(), 0.1)
    # H_y_x calculation
    # x=0
    zero_indexes = np.where(tx_data_for_mi == 0)[0]
    H_y_x_0 = entropy_with_bin_width(llr_for_mi[zero_indexes].numpy(), 0.1)
    # x=1
    one_indexes = np.where(tx_data_for_mi == 1)[0]
    H_y_x_1 = entropy_with_bin_width(llr_for_mi[one_indexes].numpy(), 0.1)

    H_y_x = 0.5 * H_y_x_0 + 0.5 * H_y_x_1

    mi = H_y - H_y_x
    mi = np.maximum(mi, 0)
    return mi


def get_next_divisible(num, divisor):
    return (num + divisor - 1) // divisor * divisor


def plot_loss_and_LLRs(train_loss_vect, val_loss_vect, llrs_mat, snr_cur, detector, kernel_size, train_samples,
                       val_samples, mod_text, cfo_str, ber, ber_lmmse, iteration):
    num_res = conf.num_res
    p_len = conf.epochs * (iteration + 1)
    if detector == 'ESCNN' or detector == 'MHSA' or detector == 'FILM' or detector == 'TDFDCNN':
        iters_txt = ', #iterations=' + str(conf.iterations)
    elif detector == 'DeepSICe2e':
        iters_txt = ', #iters_e2e=' + str(conf.iters_e2e)
    else:
        iters_txt = ''

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 4.8))
    epochs_vect = list(range(1, len(train_loss_vect) + 1))
    axes[0].plot(epochs_vect[:p_len], train_loss_vect[:p_len], linestyle='-', color='b',
                 label='Training Loss')
    axes[0].plot(epochs_vect[:p_len], val_loss_vect[:p_len], linestyle='-', color='r',
                 label='Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    title_string = (detector + ', ' + mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(
        val_samples) + ', SNR=' + str(
        snr_cur) + ", #REs=" + str(num_res) + ', #UEs=' + str(
        conf.n_users) + '\n ' +
                    cfo_str + ', Epochs=' + str(conf.epochs) + iters_txt + ', CNN kernel size=' + str(kernel_size))

    axes[0].set_title(title_string, fontsize=8)
    axes[0].legend()
    axes[0].grid()

    axes[1].hist(llrs_mat.cpu().flatten(), bins=30, color='blue', edgecolor='black', alpha=0.7)
    if (detector == 'ESCNN') or (detector == 'DeepSICe2e') or (detector == 'MHSA') or (detector == 'TDFDCNN'):
        axes[1].set_xlabel('LLRs iteration ' + str(iteration + 1))
    else:
        axes[1].set_xlabel('LLRs')
    axes[1].set_ylabel('#Values')
    axes[1].grid()
    text = 'BER ' + detector + ':' + str(f"{ber:.4f}") + '\
             BER lmmse:' + str(f"{ber_lmmse:.4f}")
    fig.text(0.15, 0.02, text, ha="left", va="center", fontsize=8)
    plt.tight_layout()
    plt.show()
    return fig


def run_evaluate(escnn_trainer, deepsice2e_trainer, deeprx_trainer, deepsic_trainer, deepsicmb_trainer,
                 deepstag_trainer, mhsa_trainer, tdcnn_trainer, tdfdcnn_trainer) -> List[float]:
    """
    The online evaluation run. Main function for running the experiments of sequential transmission of pilots and
    data blocks for the paper.
    :return: list of ber per timestep
    """

    fig_nums = plt.get_fignums()  # Get a list of all figure numbers
    for fig_num in fig_nums[:-1]:  # Exclude the last figure
        plt.close(fig_num)
    plt.close('all')

    num_res = conf.num_res


    if conf.mcs > -1:
        qm, code_rate = get_mcs(conf.mcs)
        num_bits_data = int(qm)
        mod_data = int(2 ** qm)
        ldpc_n = int(conf.num_res * NUM_SYMB_PER_SLOT * qm)
        ldpc_k = int(ldpc_n * code_rate)
    else:
        mod_data = conf.mod_data
        num_bits_data = int(np.log2(mod_data))
        ldpc_n = 0
        ldpc_k = 0

    n_users = conf.n_users
    n_ants = conf.n_ants
    iterations = conf.iterations
    iters_e2e = conf.iters_e2e
    epochs = conf.epochs
    half_kernel = int(np.ceil(conf.kernel_size / 2))

    if mod_data == 2:
        mod_text = 'BPSK'
    elif mod_data == 4:
        mod_text = 'QPSK'
    else:
        mod_text = [str(mod_data) + 'QAM']
        mod_text = mod_text[0]

    if conf.TDL_model[0] == 'N':
        chan_text = 'Flat'
    else:
        chan_text = 'TDL-' + conf.TDL_model + '-' + str(int(round(float(conf.delay_spread) * 1e9)))

    now = datetime.now()
    formatted_date = now.strftime("%Y_%m_%d_%H_%M_")

    if conf.full_e2e:
        iters_e2e_disp = 1
    else:
        iters_e2e_disp = iters_e2e

    total_ber_list = [[] for _ in range(iterations)]
    total_bler_list = [[] for _ in range(iterations)]
    total_ber_e2e_list = [[] for _ in range(iters_e2e_disp)]
    total_ber_deepsic_list = [[] for _ in range(iterations)]
    total_bler_deepsic_list = [[] for _ in range(iterations)]
    total_ber_tdfdcnn_list = [[] for _ in range(iterations)]
    total_bler_tdfdcnn_list = [[] for _ in range(iterations)]
    total_ber_deepsicmb_list = [[] for _ in range(iterations)]
    total_ber_deepstag_list = [[] for _ in range(iterations * 2)]
    total_ber_mhsa_list = [[] for _ in range(iterations)]
    total_bler_mhsa_list = [[] for _ in range(iterations)]
    total_ber_deeprx = []
    total_ber_lmmse = []
    total_bler_lmmse = []
    total_ber_sphere = []
    total_bler_sphere = []
    total_bler_deeprx = []


    SNR_range = list(range(conf.snr, conf.snr + conf.num_snrs, conf.snr_step))
    total_mi_list = [[] for _ in range(iterations)]
    total_mi_e2e_list = [[] for _ in range(iters_e2e_disp)]
    total_mi_deeprx = []
    if mod_data == 4:
        total_mi_lmmse = []
    Final_SNR = conf.snr + conf.num_snrs - 1

    if mod_data == 2:
        constellation_factor = 1
    elif mod_data == 4:
        constellation_factor = 2
    elif mod_data == 16:
        constellation_factor = 10
    elif mod_data == 64:
        constellation_factor = 42
    elif mod_data == 256:
        constellation_factor = 170

    if conf.run_sphere or conf.which_augment == 'AUGMENT_SPHERE':
        run_sphere = True
    else:
        run_sphere = False

    if conf.run_deepsic or conf.which_augment == 'AUGMENT_DEEPSIC':
        run_deepsic = True
    else:
        run_deepsic = False

    if conf.run_mhsa or conf.which_augment == 'AUGMENT_MHSA':
        run_mhsa = True
    else:
        run_mhsa = False

    if conf.run_deeprx or conf.which_augment == 'AUGMENT_DEEPRX':
        run_deeprx = True
    else:
        run_deeprx = False

    for snr_cur in SNR_range:
        ber_sum = np.zeros(iterations)
        ber_per_re = np.zeros((iterations, conf.num_res))
        ber_per_re_deepsic = np.zeros((iterations, conf.num_res))
        ber_per_re_deepsicmb = np.zeros((iterations, conf.num_res))
        ber_per_re_deepstag = np.zeros((iterations * 2, conf.num_res))
        ber_per_re_mhsa = np.zeros((iterations, conf.num_res))
        ber_sum_e2e = np.zeros(iters_e2e_disp)
        ber_sum_tdfdcnn = np.zeros(iterations)
        ber_sum_deeprx = 0
        ber_sum_lmmse = 0
        ber_per_re_lmmse = np.zeros(conf.num_res)
        ber_sum_sphere = 0
        ber_sum_deepsic = np.zeros(iterations)
        ber_sum_deepsicmb = np.zeros(iterations)
        ber_sum_deepstag = np.zeros(iterations * 2)
        ber_sum_mhsa = np.zeros(iterations)

        if conf.mod_pilot> 0:
            num_bits_pilot = int(np.log2(conf.mod_pilot))
        else:
            num_bits_pilot = num_bits_data



        pilot_data_ratio = num_bits_pilot/num_bits_data

        escnn_trainer._initialize_detector(num_bits_pilot, n_users, n_ants)  # For reseting the weights
        if conf.run_tdfdcnn:
            tdfdcnn_trainer._initialize_detector(num_bits_pilot, n_users, n_ants)  # For reseting the weights
        deepsice2e_trainer._initialize_detector(num_bits_pilot, n_users, n_ants)
        deeprx_trainer._initialize_detector(num_bits_pilot, n_users, n_ants)
        if run_deepsic:
            deepsic_trainer._initialize_detector(num_bits_pilot, n_users, n_ants)

        if conf.run_deepsicmb:
            deepsicmb_trainer._initialize_detector(num_bits_pilot, n_users, n_ants)

        if conf.run_deepstag:
            deepstag_trainer._initialize_detector(num_bits_pilot, n_users, n_ants)

        if run_mhsa:
            mhsa_trainer._initialize_detector(num_bits_pilot, n_users, n_ants)

        if conf.run_tdcnn:
            tdcnn_trainer._initialize_detector(num_bits_pilot, n_users, n_ants)  # For reseting the weights

        if conf.run_tdfdcnn:
            tdfdcnn_trainer._initialize_detector(num_bits_pilot, n_users, n_ants)  # For reseting the weights

        pilot_size = get_next_divisible(conf.pilot_size, num_bits_pilot * NUM_SYMB_PER_SLOT * 3) # The 3 is for the possible 64QAM->16QAM->QPSK
        pilot_chunk = int(pilot_size / num_bits_pilot)

        data_size = get_next_divisible(conf.pilot_size*(conf.block_length_factor-1), num_bits_data * NUM_SYMB_PER_SLOT * 3)


        noise_var = 10 ** (-0.1 * snr_cur) * constellation_factor

        channel_dataset = ChannelModelDataset(block_length=conf.block_length,
                                              pilot_length=pilot_size,
                                              data_length=data_size,
                                              blocks_num=conf.blocks_num,
                                              num_res=conf.num_res,
                                              clip_percentage_in_tx=conf.clip_percentage_in_tx,
                                              cfo_and_iqmm_in_rx=conf.cfo_and_iqmm_in_rx,
                                              kernel_size=conf.kernel_size,
                                              n_users=n_users)

        transmitted_words, received_words, received_words_ce, hs, s_orig_words, received_words_clean = channel_dataset.__getitem__(
            noise_var_list=[noise_var], num_bits_pilot=num_bits_pilot, num_bits_data=num_bits_data, n_users
            =n_users, mod_data=mod_data, ldpc_k=ldpc_k, ldpc_n=ldpc_n)

        train_samples = int(pilot_size * TRAIN_PERCENTAGE / 100)
        val_samples = pilot_size - train_samples

        # detect sequentially
        for block_ind in range(conf.blocks_num):
            print('*' * 20)
            # get current word and channel
            tx, h, rx, rx_ce, s_orig, rx_clean = transmitted_words[block_ind], hs[block_ind], received_words[block_ind], \
                received_words_ce[block_ind], s_orig_words[block_ind], received_words_clean[block_ind]

            NUM_SLOTS = int(s_orig.shape[0] / NUM_SYMB_PER_SLOT)
            if conf.cfo != 0:
                # Compensate for CFO linear phase
                pointer = 0
                n = np.arange(int(NUM_SLOTS * NUM_SAMPLES_PER_SLOT))
                if GENIE_CFO:
                    cfo_est = conf.cfo
                else:
                    cfo_est_vect = np.zeros(n_users)
                    for user in range(n_users):
                        grad_sum = 0
                        for re in range(conf.num_res):
                            s_orig_pilot = s_orig[:pilot_chunk, user, re]
                            rx_pilot_ce_cur = rx_ce[user, :pilot_chunk, :, re]
                            cur_ce = 1 / s_orig_pilot.shape[0] * (s_orig_pilot[:, None].conj() / (
                                    torch.abs(s_orig_pilot[:, None]) ** 2) * rx_pilot_ce_cur)
                            cur_ce = cur_ce.cpu().numpy()
                            grad_sum = grad_sum + np.sum(np.sum(cur_ce[:-1, :] * np.conj(cur_ce[1:, :]), axis=0))
                        cfo_est_vect[user] = -np.angle(grad_sum) * FFT_size / (2 * np.pi * (FFT_size + CP))
                    cfo_est = np.mean(cfo_est_vect)

                cfo_phase = -2 * np.pi * cfo_est * n / FFT_size  # CFO phase shift
                cfo_comp_vect = np.array([])
                cp_length = FIRST_CP
                for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                    pointer += (cp_length + int(FFT_size / 2))
                    cfo_comp_vect = np.concatenate(
                        (cfo_comp_vect, np.array([np.exp(1j * cfo_phase[pointer])])))
                    pointer += int(FFT_size / 2)
                    cp_length = CP
                cfo_comp_vect = np.tile(cfo_comp_vect, NUM_SLOTS)

                for i in range(s_orig.shape[0]):
                    rx[i, :, :] = rx[i, :, :] * cfo_comp_vect[i]
                    rx_ce[:, i, :, :] = rx_ce[:, i, :, :] * cfo_comp_vect[i]

            if conf.plot_channel:
                symbols_to_plot = [0, 4, 8, 12]
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 4.8))
                rx_np = rx.detach().cpu().numpy()
                axes[0].plot(20 * np.log10(np.abs(rx_np[symbols_to_plot[0], 0, :])), '-o', color='g',
                             label='Symbol ' + str(symbols_to_plot[0]))
                axes[0].plot(20 * np.log10(np.abs(rx_np[symbols_to_plot[1], 0, :])), '-o', color='r',
                             label='Symbol ' + str(symbols_to_plot[1]))
                axes[0].plot(20 * np.log10(np.abs(rx_np[symbols_to_plot[2], 0, :])), '-o', color='k',
                             label='Symbol ' + str(symbols_to_plot[2]))
                axes[0].plot(20 * np.log10(np.abs(rx_np[symbols_to_plot[3], 0, :])), '-o', color='b',
                             label='Symbol ' + str(symbols_to_plot[3]))
                axes[0].set_xlabel('Subcarrier')
                axes[0].set_ylabel('Amp (dB)')
                axes[0].set_title('TDL-' + conf.TDL_model + ', delay spread=' + str(
                    int(round(float(conf.delay_spread) * 1e9))) + ' nsec', fontsize=10)
                axes[0].legend()
                axes[0].grid()

                axes[1].plot(20 * np.unwrap(np.angle(rx_np[symbols_to_plot[0], 0, :])), '-', color='g',
                             label='Symbol ' + str(symbols_to_plot[0]))
                axes[1].plot(20 * np.unwrap(np.angle(rx_np[symbols_to_plot[1], 0, :])), '-', color='r',
                             label='Symbol ' + str(symbols_to_plot[1]))
                axes[1].plot(20 * np.unwrap(np.angle(rx_np[symbols_to_plot[2], 0, :])), '-', color='k',
                             label='Symbol ' + str(symbols_to_plot[2]))
                axes[1].plot(20 * np.unwrap(np.angle(rx_np[symbols_to_plot[3], 0, :])), '-', color='b',
                             label='Symbol ' + str(symbols_to_plot[3]))
                axes[1].set_xlabel('Subcarrier')
                axes[1].set_ylabel('Phase (Rad)')
                axes[1].set_title('TDL-' + conf.TDL_model + ', delay spread=' + str(
                    int(round(float(conf.delay_spread) * 1e9))) + ' nsec', fontsize=10)
                axes[1].legend()
                axes[1].grid()
                fig.tight_layout()
                plt.show()

                # === Real part ===
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 4.8))
                axes[0].plot(np.real(rx_np[symbols_to_plot[0], 0, :]), '-o', color='g',
                             label=f'Symbol {symbols_to_plot[0]}')
                axes[0].plot(np.real(rx_np[symbols_to_plot[1], 0, :]), '-o', color='r',
                             label=f'Symbol {symbols_to_plot[1]}')
                axes[0].plot(np.real(rx_np[symbols_to_plot[2], 0, :]), '-o', color='k',
                             label=f'Symbol {symbols_to_plot[2]}')
                axes[0].plot(np.real(rx_np[symbols_to_plot[3], 0, :]), '-o', color='b',
                             label=f'Symbol {symbols_to_plot[3]}')
                axes[0].set_xlabel('Subcarrier')
                axes[0].set_ylabel('Real Part')
                axes[0].set_title(
                    f'TDL-{conf.TDL_model}, delay spread={int(round(float(conf.delay_spread) * 1e9))} nsec',
                    fontsize=10)
                axes[0].legend()
                axes[0].grid()

                # === Imaginary part ===
                axes[1].plot(np.imag(rx_np[symbols_to_plot[0], 0, :]), '-o', color='g',
                             label=f'Symbol {symbols_to_plot[0]}')
                axes[1].plot(np.imag(rx_np[symbols_to_plot[1], 0, :]), '-o', color='r',
                             label=f'Symbol {symbols_to_plot[1]}')
                axes[1].plot(np.imag(rx_np[symbols_to_plot[2], 0, :]), '-o', color='k',
                             label=f'Symbol {symbols_to_plot[2]}')
                axes[1].plot(np.imag(rx_np[symbols_to_plot[3], 0, :]), '-o', color='b',
                             label=f'Symbol {symbols_to_plot[3]}')
                axes[1].set_xlabel('Subcarrier')
                axes[1].set_ylabel('Imaginary Part')
                axes[1].set_title(
                    f'TDL-{conf.TDL_model}, delay spread={int(round(float(conf.delay_spread) * 1e9))} nsec',
                    fontsize=10)
                axes[1].legend()
                axes[1].grid()

                fig.tight_layout()
                plt.show()

            # Time domain processing:
            if conf.run_tdcnn:
                s_t_matrix = torch.zeros((NUM_SLOTS, 2 * conf.n_ants, FFT_size, NUM_SYMB_PER_SLOT), dtype=torch.float32)
                s_t_matrix_clean = torch.zeros_like(s_t_matrix)
                for ant in range(conf.n_ants):
                    for slot_num in range(NUM_SLOTS):
                        for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                            cur_index = slot_num * NUM_SYMB_PER_SLOT + ofdm_symbol
                            s_t = torch.fft.ifft(rx[cur_index, ant, :], n=FFT_size)
                            s_t_matrix[slot_num, 2*ant, :, ofdm_symbol] = torch.real(s_t)
                            s_t_matrix[slot_num, 2*ant+1, :, ofdm_symbol] = torch.imag(s_t)
                            s_t_clean = torch.fft.ifft(rx_clean[cur_index, ant, :], n=FFT_size)
                            s_t_matrix_clean[slot_num, 2*ant, :, ofdm_symbol] = torch.real(s_t_clean)
                            s_t_matrix_clean[slot_num, 2*ant+1, :, ofdm_symbol] = torch.imag(s_t_clean)

                if tdcnn_trainer.is_online_training:
                    s_t_matrix = s_t_matrix.to(DEVICE)
                    s_t_matrix_clean = s_t_matrix_clean.to(DEVICE)
                    train_loss_vect_tdcnn, val_loss_vect_tdcnn = tdcnn_trainer._online_training(
                        s_t_matrix_clean[:NUM_SLOTS//5], s_t_matrix[:NUM_SLOTS//5], num_bits_data, n_users, iterations, epochs, False, torch.empty(0))
                    with torch.no_grad():
                        s_t_matrix, _ = tdcnn_trainer._forward(s_t_matrix, num_bits_data,
                                                             n_users,
                                                             iterations,
                                                             torch.empty(0))

                for ant in range(conf.n_ants):
                    index = 0
                    for slot_num in range(NUM_SLOTS):
                        for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                            rx_t = s_t_matrix[slot_num, 2*ant, :, ofdm_symbol] + 1j * s_t_matrix[slot_num, 2*ant+1, :, ofdm_symbol]
                            rx_f = torch.fft.fft(rx_t, n=FFT_size)
                            rx[index, ant, :] = rx_f[:num_res]
                            index += 1


            # Interleave real and imaginary parts of Rx into a real tensor
            real_part = rx.real
            imag_part = rx.imag
            rx_real = torch.empty((rx.shape[0], rx.shape[1] * 2, rx.shape[2]), dtype=torch.float32)
            rx_real[:, 0::2, :] = real_part  # Real parts in even rows
            rx_real[:, 1::2, :] = imag_part  # Imaginary parts in odd rows

            # split words into data and pilot part
            tx_pilot, tx_data = tx[:pilot_size], tx[pilot_size:]
            rx_pilot, rx_data = rx_real[:pilot_chunk], rx_real[pilot_chunk:]

            rx_c = rx.cpu()  # rx_c is the rx in CPU
            llrs_mat_lmmse_for_aug = np.zeros((rx_c.shape[0], num_bits_pilot * n_users, num_res, 1))
            llrs_mat_sphere_for_aug = np.zeros((rx_c.shape[0], num_bits_pilot * n_users, num_res, 1))
            detected_word_lmmse_for_aug = np.zeros((int(rx_c.shape[0] * num_bits_pilot), n_users, num_res))
            detected_word_sphere_for_aug = np.zeros((int(rx_c.shape[0] * num_bits_pilot), n_users, num_res))
            time_ce = 0
            time_lmmse = 0
            time_sphere = 0
            # for re in range(conf.num_res):
            for re in range(conf.num_res):
                H = torch.zeros((conf.n_ants, conf.n_users), dtype=rx_ce.dtype, device=rx_ce.device)

                # Regular CE
                if conf.pilot_channel_seed > 0:
                    # Calling twice to perform separate channel estimation for the pilot and the data part - works only for LMMSE, not for sphere
                    LmmseDemod(rx_ce[:, :pilot_chunk, :, :], rx_c[:pilot_chunk, :, :], s_orig[:pilot_chunk, :, :],
                               noise_var, pilot_chunk, re, num_bits_pilot, llrs_mat_lmmse_for_aug[:pilot_chunk, :, :, :],
                               detected_word_lmmse_for_aug[:pilot_size, :, :], H)
                    LmmseDemod(rx_ce[:, pilot_chunk:, :, :], rx_c[pilot_chunk:, :, :], s_orig[pilot_chunk:, :, :],
                               noise_var, pilot_chunk, re, num_bits_data, llrs_mat_lmmse_for_aug[pilot_chunk:, :, :, :],
                               detected_word_lmmse_for_aug[pilot_size:, :, :], H)
                else:
                    equalized, postEqSINR = LmmseEqualize(rx_ce, rx_c, s_orig,
                               noise_var, pilot_chunk, re, H)
                    if not(conf.increase_prime_modulation):
                        LmmseDemod(equalized[:pilot_chunk], postEqSINR, num_bits_pilot, re, llrs_mat_lmmse_for_aug[:pilot_chunk, :, :, :],
                                   detected_word_lmmse_for_aug[:pilot_size, :, :], 1)
                        LmmseDemod(equalized[pilot_chunk:], postEqSINR, num_bits_data, re, llrs_mat_lmmse_for_aug[pilot_chunk:, :, :, :],
                                   detected_word_lmmse_for_aug[pilot_size:, :, :], pilot_data_ratio)
                    else:
                        # increase_prime_modulation mode: QPSK→16QAM or 16QAM→64QAM
                        if num_bits_pilot == 2:
                            # QPSK→16QAM: 2 bits → 4 bits
                            # Bits 0,2 (signs) from QPSK; bits 1,3 (magnitudes) unknown
                            detected_word_lmmse_for_aug[1::2,:,re] = 0.5
                            llrs_mat_lmmse_for_aug[:, 1::2, re:re+1, :] = 0
                            LmmseDemod(equalized, postEqSINR, 2, re, llrs_mat_lmmse_for_aug[:, 0::2, :, :],
                                       detected_word_lmmse_for_aug[0::2,:,:], 1)
                        elif num_bits_pilot == 4:
                            # 16QAM→64QAM: 4 bits → 6 bits
                            # 64QAM bits: [sign_I, half_I, pos_I, sign_Q, half_Q, pos_Q]
                            # From 16QAM: [sign_I, mag_I, sign_Q, mag_Q]
                            # Mapping: 64b0=16b0, 64b1=0, 64b2=-16b1, 64b3=16b2, 64b4=0, 64b5=-16b3

                            # First, get 16QAM LLRs into temporary storage
                            llrs_16qam = np.zeros((llrs_mat_lmmse_for_aug.shape[0], 4 * n_users, 1, 1))
                            det_16qam = np.zeros((detected_word_lmmse_for_aug.shape[0] // 6 * 4, n_users, 1))
                            LmmseDemod(equalized, postEqSINR, 4, re, llrs_16qam, det_16qam[:,:,0], 1)

                            # Map to 64QAM (6 bits per user)
                            for user in range(n_users):
                                # 16QAM bit indices for this user
                                i16_b0 = user * 4 + 0  # sign_I
                                i16_b1 = user * 4 + 1  # mag_I
                                i16_b2 = user * 4 + 2  # sign_Q
                                i16_b3 = user * 4 + 3  # mag_Q

                                # 64QAM bit indices for this user
                                i64_b0 = user * 6 + 0  # sign_I
                                i64_b1 = user * 6 + 1  # half_I (unknown)
                                i64_b2 = user * 6 + 2  # pos_I (inverted mag_I)
                                i64_b3 = user * 6 + 3  # sign_Q
                                i64_b4 = user * 6 + 4  # half_Q (unknown)
                                i64_b5 = user * 6 + 5  # pos_Q (inverted mag_Q)

                                # Copy signs directly
                                llrs_mat_lmmse_for_aug[:, i64_b0, re:re+1, :] = llrs_16qam[:, i16_b0, :, :]
                                llrs_mat_lmmse_for_aug[:, i64_b3, re:re+1, :] = llrs_16qam[:, i16_b2, :, :]

                                # Unknown bits (half selection) - set to 0
                                llrs_mat_lmmse_for_aug[:, i64_b1, re:re+1, :] = 0
                                llrs_mat_lmmse_for_aug[:, i64_b4, re:re+1, :] = 0

                                # Inverted magnitude bits
                                llrs_mat_lmmse_for_aug[:, i64_b2, re:re+1, :] = -llrs_16qam[:, i16_b1, :, :]
                                llrs_mat_lmmse_for_aug[:, i64_b5, re:re+1, :] = -llrs_16qam[:, i16_b3, :, :]

                            # Set detected words for unknown bits to 0.5
                            detected_word_lmmse_for_aug[1::3,:,re] = 0.5  # half bits (positions 1,4 in each 6-bit group)



                if run_sphere:
                    H = H.cpu().numpy()

                    if not (conf.increase_prime_modulation):
                        llr_out, detected_word_sphere_for_aug[:, :, re] = SphereDecoder(H, rx_c[:, :, re].numpy(),
                                                                                        noise_var, conf.sphere_radius)
                    else:
                        # increase_prime_modulation mode: QPSK→16QAM or 16QAM→64QAM
                        if num_bits_pilot == 2:
                            # QPSK→16QAM: 2 bits → 4 bits
                            detected_word_sphere_for_aug[1::2,:,re] = 0.5
                            llr_out_red, detected_word_sphere_for_aug[0::2, :, re] = SphereDecoder(H, rx_c[:, :, re].numpy(),
                                                                                            noise_var, conf.sphere_radius)
                            llr_out = np.zeros((int(llr_out_red.shape[0]*2), llr_out_red.shape[1]))
                            llr_out[0::2,:] = llr_out_red
                        elif num_bits_pilot == 4:
                            # 16QAM→64QAM: 4 bits → 6 bits
                            llr_out_16qam, _ = SphereDecoder(H, rx_c[:, :, re].numpy(), noise_var, conf.sphere_radius)

                            # Expand from 4 bits to 6 bits per symbol
                            num_symbols = llr_out_16qam.shape[0] // 4
                            llr_out = np.zeros((num_symbols * 6, n_users))

                            for user in range(n_users):
                                for sym in range(num_symbols):
                                    # 16QAM indices
                                    i16_base = sym * 4
                                    # 64QAM indices
                                    i64_base = sym * 6

                                    # Signs - direct copy
                                    llr_out[i64_base + 0, user] = llr_out_16qam[i16_base + 0, user]  # sign_I
                                    llr_out[i64_base + 3, user] = llr_out_16qam[i16_base + 2, user]  # sign_Q

                                    # Half bits - no info
                                    llr_out[i64_base + 1, user] = 0  # half_I
                                    llr_out[i64_base + 4, user] = 0  # half_Q

                                    # Position bits - inverted magnitude
                                    llr_out[i64_base + 2, user] = -llr_out_16qam[i16_base + 1, user]  # pos_I
                                    llr_out[i64_base + 5, user] = -llr_out_16qam[i16_base + 3, user]  # pos_Q

                            detected_word_sphere_for_aug[1::3,:,re] = 0.5  # half bits unknown

                else:
                    llr_out = np.zeros((rx_c.shape[0] * num_bits_pilot, n_users))
                    detected_word_sphere_for_aug[:, :, re] = np.zeros((rx_c.shape[0] * num_bits_pilot, n_users))

                for user in range(n_users):
                    llrs_mat_sphere_for_aug[:, (user * num_bits_pilot):((user + 1) * num_bits_pilot), re, :] = -llr_out[:,
                                                                                                    user].reshape(
                        int(llr_out[:, user].shape[0] / num_bits_pilot), num_bits_pilot, 1)

            # Time measurements - currently not used
            # time_ce = time_ce / (conf.num_res-1) * conf.num_res
            # time_lmmse = time_lmmse / (conf.num_res-1) * conf.num_res
            # time_sphere = time_sphere / (conf.num_res-1) * conf.num_res

            llrs_mat_lmmse = llrs_mat_lmmse_for_aug[pilot_chunk:, :, :, :]
            llrs_mat_sphere = llrs_mat_sphere_for_aug[pilot_chunk:, :, :, :]
            if conf.which_augment == 'AUGMENT_SPHERE':
                probs_for_aug = torch.sigmoid(torch.tensor(llrs_mat_sphere_for_aug, dtype=torch.float32))
            elif conf.which_augment == 'AUGMENT_LMMSE':
                llrs_mat_lmmse_for_aug_c = llrs_mat_lmmse_for_aug.copy()
                if pilot_data_ratio == 1.5:
                    llrs_mat_lmmse_for_aug_c[pilot_chunk:, 2::3, :, :] *= -1

                probs_for_aug = torch.sigmoid(torch.tensor(llrs_mat_lmmse_for_aug_c, dtype=torch.float32))
            else:
                probs_for_aug = torch.tensor([], dtype=torch.float32)


            if run_deepsic:
                if deepsic_trainer.is_online_training:
                    train_loss_vect_deepsic, val_loss_vect_deepsic = deepsic_trainer._online_training(
                        tx_pilot, rx_pilot, num_bits_pilot, n_users, iterations, epochs, False, torch.empty(0))
                    detected_word_deepsic_list, llrs_mat_deepsic_list = deepsic_trainer._forward(rx_data, num_bits_pilot,
                                                                                                 n_users,
                                                                                                 iterations,
                                                                                                 torch.empty(0))
                    if conf.which_augment == 'AUGMENT_DEEPSIC':
                        _, llrs_mat_deepsic_pilot_list = deepsic_trainer._forward(rx_pilot, num_bits_pilot, n_users,
                                                                                  iterations, torch.empty(0))
                        probs_for_aug = torch.cat(
                            (torch.sigmoid(llrs_mat_deepsic_pilot_list[0]), torch.sigmoid(llrs_mat_deepsic_list[0])),
                            dim=0).cpu()
                        probs_for_aug = probs_for_aug.unsqueeze(-1)

            if run_deeprx:
                if deeprx_trainer.is_online_training:
                    train_loss_vect_deeprx, val_loss_vect_deeprx = deeprx_trainer._online_training(tx_pilot, rx_pilot,
                                                                                                   num_bits_pilot, n_users,
                                                                                                   iterations, epochs,
                                                                                                   False,
                                                                                                   torch.empty(0))
                    detected_word_deeprx, llrs_mat_deeprx = deeprx_trainer._forward(rx_data, num_bits_pilot, n_users,
                                                                                    iterations, torch.empty(0))
                    if conf.which_augment == 'AUGMENT_DEEPRX':
                        _, llrs_mat_deeprx_pilot = deeprx_trainer._forward(rx_pilot, num_bits_pilot, n_users, iterations,
                                                                           torch.empty(0))
                        probs_for_aug = torch.cat(
                            (torch.sigmoid(llrs_mat_deeprx_pilot), torch.sigmoid(llrs_mat_deeprx)), dim=0).cpu()

            if conf.override_augment_with_lmmse:
                probs_for_aug_lmmse = torch.sigmoid(torch.tensor(llrs_mat_lmmse_for_aug, dtype=torch.float32))

            # online training main function
            if escnn_trainer.is_online_training:

                if conf.make_64QAM_16QAM_percentage == 50 and num_bits_pilot == 6:  # 64QAM special case
                    # Divide pilot_chunk into three equal parts
                    third_size = pilot_chunk // 3

                    # First third: skip_indices with ratio 3
                    indexes_first = skip_indices(int(num_bits_pilot*conf.n_users), 3)
                    probs_for_aug[:third_size, indexes_first, :, :] = 0.5

                    # Second third: skip_indices with ratio 1.5
                    indexes_second = skip_indices(int(num_bits_pilot*conf.n_users), 1.5)
                    probs_for_aug[third_size:2*third_size, indexes_second, :, :] = 0.5

                    # Third part: unchanged (full 64QAM), no modification needed

                elif not(conf.make_64QAM_16QAM_percentage
                ) and conf.make_64QAM_16QAM_percentage>0:
                    # Original logic for other percentages
                    indexes = skip_indices(int(num_bits_pilot*conf.n_users), pilot_data_ratio)
                    probs_for_aug[:int(pilot_chunk*conf.make_64QAM_16QAM_percentage/100), indexes, :, :] = 0.5

                train_loss_vect, val_loss_vect = escnn_trainer._online_training(tx_pilot, rx_pilot, num_bits_pilot,
                                                                                n_users, iterations, epochs, False,
                                                                                probs_for_aug[:pilot_chunk], stage="base" )

                if conf.use_film:
                    train_loss_vect_film, val_loss_vect_film = escnn_trainer._online_training(tx_pilot, rx_pilot, num_bits_pilot,
                                                                                    n_users, iterations, epochs, False,
                                                                                    probs_for_aug[:pilot_chunk], stage="film")

                if conf.override_augment_with_lmmse:
                    probs_for_aug.copy_(probs_for_aug_lmmse)

                # OryEger
                # torch.save(rx_data, "rx_data.pt")
                # rx_data = torch.load("rx_data.pt", map_location="cpu")
                # state = torch.load("model_weights.pt", map_location="cpu")
                # escnn_trainer.detector[0][0].load_state_dict(state)
                # escnn_trainer.detector[0][0].eval()
                # for i in range(1,rx_data.shape[2]):
                #     rx_data[0, :, i] = rx_data[0, :, 0]

                detected_word_list, llrs_mat_list = escnn_trainer._forward(rx_data, num_bits_pilot, n_users, iterations,
                                                                           probs_for_aug[pilot_chunk:])



            if conf.run_tdfdcnn:
                if tdfdcnn_trainer.is_online_training:

                    train_loss_vect_tdfdcnn, val_loss_vect_tdfdcnn = tdfdcnn_trainer._online_training(tx_pilot, rx_pilot, num_bits_pilot,
                                                                                    n_users, iterations, epochs, False,
                                                                                    probs_for_aug[:pilot_chunk])

                    detected_word_tdfdcnn_list, llrs_mat_tdfdcnn_list = tdfdcnn_trainer._forward(rx_data, num_bits_pilot, n_users, iterations,
                                                                           probs_for_aug[pilot_chunk:])


            if conf.run_e2e:
                if deepsice2e_trainer.is_online_training:
                    train_loss_vect_e2e, val_loss_vect_e2e = deepsice2e_trainer._online_training(tx_pilot, rx_pilot,
                                                                                                 num_bits_pilot, n_users,
                                                                                                 iters_e2e, epochs,
                                                                                                 False, torch.empty(0))
                    detected_word_e2e_list, llrs_mat_e2e_list = deepsice2e_trainer._forward(rx_data, num_bits_pilot, n_users,
                                                                                            iters_e2e, torch.empty(0))

            if conf.run_deepsicmb:
                if deepsicmb_trainer.is_online_training:
                    train_loss_vect_deepsicmb, val_loss_vect_deepsicmb = deepsicmb_trainer._online_training(
                        tx_pilot, rx_pilot, num_bits_pilot, n_users, iterations, epochs, False, torch.empty(0))
                    detected_word_deepsicmb_list, llrs_mat_deepsicmb_list = deepsicmb_trainer._forward(rx_data,
                                                                                                       num_bits_pilot,
                                                                                                       n_users,
                                                                                                       iterations,
                                                                                                       torch.empty(0))
            if conf.run_deepstag:
                if deepstag_trainer.is_online_training:
                    train_loss_vect_deepstag, val_loss_vect_deepstag = deepstag_trainer._online_training(
                        tx_pilot, rx_pilot, num_bits_pilot, n_users, iterations, epochs, False, torch.empty(0))
                    detected_word_deepstag_list, llrs_mat_deepstag_list = deepstag_trainer._forward(rx_data, num_bits_pilot,
                                                                                                    n_users,
                                                                                                    iterations,
                                                                                                    torch.empty(0))
            if run_mhsa:
                if mhsa_trainer.is_online_training:
                    train_loss_vect_mhsa, val_loss_vect_mhsa = mhsa_trainer._online_training(
                        tx_pilot, rx_pilot, num_bits_pilot, n_users, iterations, epochs, False, torch.empty(0))
                    detected_word_mhsa_list, llrs_mat_mhsa_list = mhsa_trainer._forward(rx_data, num_bits_pilot, n_users,
                                                                                        iterations, torch.empty(0))

            # CE Based
            rx_data_c = rx[pilot_chunk:].cpu()

            for re in range(conf.num_res):
                # Regular CE
                if pilot_data_ratio <= 1:
                    detected_word_lmmse = detected_word_lmmse_for_aug[pilot_size:, :, re]
                    detected_word_sphere = detected_word_sphere_for_aug[pilot_size:, :, re]
                else:
                    detected_word_lmmse = detected_word_lmmse_for_aug[pilot_size+relevant_indices(detected_word_lmmse_for_aug.shape[0]-pilot_size,pilot_data_ratio), :, re]
                    detected_word_sphere = detected_word_sphere_for_aug[pilot_size+relevant_indices(detected_word_sphere_for_aug.shape[0]-pilot_size,pilot_data_ratio), :, re]

                # Sphere:
                if conf.sphere_radius == 'inf':
                    radius = np.inf
                    modulator_text = 'MAP'
                else:
                    radius = float(conf.sphere_radius)
                    modulator_text = 'Sphere, Radius=' + str(conf.sphere_radius)

                # calculate accuracy
                target = tx_data[:, :rx.shape[1], re]
                for iteration in range(iterations):
                    if pilot_data_ratio <= 1:
                        detected_word_cur_re = detected_word_list[iteration][:, :, re, :]
                    else:
                        detected_word_cur_re = detected_word_list[iteration][:, relevant_indices(detected_word_list[iteration].shape[1],pilot_data_ratio), re, :]
                    detected_word_cur_re = detected_word_cur_re.squeeze(-1)
                    detected_word_cur_re = detected_word_cur_re.reshape(int(tx_data.shape[0] / num_bits_data), n_users,
                                                                        num_bits_data).swapaxes(1, 2).reshape(
                        tx_data.shape[0], n_users)

                    if pilot_data_ratio == 1.5:
                        detected_word_cur_re[1::2, :] = 1 - detected_word_cur_re[1::2, :]

                    if conf.ber_on_one_user >= 0:
                        ber = calculate_ber(detected_word_cur_re[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                            target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits_data)
                    else:
                        ber = calculate_ber(detected_word_cur_re.cpu(), target.cpu(), num_bits_data)

                    ber_sum[iteration] += ber
                    ber_per_re[iteration, re] = ber

                if conf.run_e2e:
                    for iteration in range(iters_e2e_disp):
                        detected_word_cur_re_e2e = detected_word_e2e_list[iteration][:, :, re]
                        detected_word_cur_re_e2e = detected_word_cur_re_e2e.squeeze(-1)
                        detected_word_cur_re_e2e = detected_word_cur_re_e2e.reshape(int(tx_data.shape[0] / num_bits_data),
                                                                                    n_users,
                                                                                    num_bits_data).swapaxes(1, 2).reshape(
                            tx_data.shape[0],
                            n_users)
                        if conf.ber_on_one_user >= 0:
                            ber_e2e = calculate_ber(
                                detected_word_cur_re_e2e[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits_data)
                        else:
                            ber_e2e = calculate_ber(detected_word_cur_re_e2e.cpu(), target.cpu(), num_bits_data)
                        ber_sum_e2e[iteration] += ber_e2e

                if conf.run_tdfdcnn:
                    for iteration in range(iterations):
                        detected_word_cur_re_tdfdcnn = detected_word_tdfdcnn_list[iteration][:, :, re]
                        detected_word_cur_re_tdfdcnn = detected_word_cur_re_tdfdcnn.squeeze(-1)
                        detected_word_cur_re_tdfdcnn = detected_word_cur_re_tdfdcnn.reshape(
                            int(tx_data.shape[0] / num_bits_data),
                            n_users,
                            num_bits_data).swapaxes(1, 2).reshape(
                            tx_data.shape[0],
                            n_users)
                        if conf.ber_on_one_user >= 0:
                            ber_tdfdcnn = calculate_ber(
                                detected_word_cur_re_tdfdcnn[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits_data)
                        else:
                            ber_tdfdcnn = calculate_ber(detected_word_cur_re_tdfdcnn.cpu(), target.cpu(), num_bits_data)
                        ber_sum_tdfdcnn[iteration] += ber_tdfdcnn


                if run_deeprx:
                    detected_word_cur_re_deeprx = detected_word_deeprx[:, :, re, :]
                    detected_word_cur_re_deeprx = detected_word_cur_re_deeprx.squeeze(-1)
                    detected_word_cur_re_deeprx = detected_word_cur_re_deeprx.reshape(int(tx_data.shape[0] / num_bits_data),
                                                                                      n_users, num_bits_data).swapaxes(1,
                                                                                                                  2).reshape(
                        tx_data.shape[0], n_users)
                    if conf.ber_on_one_user >= 0:
                        ber_deeprx = calculate_ber(
                            detected_word_cur_re_deeprx[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                            target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits_data)
                    else:
                        ber_deeprx = calculate_ber(detected_word_cur_re_deeprx.cpu(), target.cpu(), num_bits_data)

                if run_deepsic:
                    for iteration in range(iterations):
                        detected_word_cur_re_deepsic = detected_word_deepsic_list[iteration][:, :, re]
                        detected_word_cur_re_deepsic = detected_word_cur_re_deepsic.squeeze(-1)
                        detected_word_cur_re_deepsic = detected_word_cur_re_deepsic.reshape(
                            int(tx_data.shape[0] / num_bits_data),
                            n_users,
                            num_bits_data).swapaxes(1, 2).reshape(
                            tx_data.shape[0],
                            n_users)
                        if conf.ber_on_one_user >= 0:
                            ber_deepsic = calculate_ber(
                                detected_word_cur_re_deepsic[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits_data)
                        else:
                            ber_deepsic = calculate_ber(detected_word_cur_re_deepsic.cpu(), target.cpu(), num_bits_data)
                        ber_sum_deepsic[iteration] += ber_deepsic
                        ber_per_re_deepsic[iteration, re] = ber_deepsic

                if conf.run_deepsicmb:
                    for iteration in range(iterations):
                        detected_word_cur_re_deepsicmb = detected_word_deepsicmb_list[iteration][:, :, re]
                        detected_word_cur_re_deepsicmb = detected_word_cur_re_deepsicmb.squeeze(-1)
                        detected_word_cur_re_deepsicmb = detected_word_cur_re_deepsicmb.reshape(
                            int(tx_data.shape[0] / num_bits_data),
                            n_users, num_bits_data).swapaxes(1, 2).reshape(
                            tx_data.shape[0], n_users)
                        if conf.ber_on_one_user >= 0:
                            ber_deepsicmb = calculate_ber(
                                detected_word_cur_re_deepsicmb[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits_data)
                        else:
                            ber_deepsicmb = calculate_ber(detected_word_cur_re_deepsicmb.cpu(), target.cpu(), num_bits_data)
                        ber_sum_deepsicmb[iteration] += ber_deepsicmb
                        ber_per_re_deepsicmb[iteration, re] = ber_deepsicmb

                if conf.run_deepstag:
                    for iteration in range(iterations * 2):
                        detected_word_cur_re_deepstag = detected_word_deepstag_list[iteration][:, :, re]
                        detected_word_cur_re_deepstag = detected_word_cur_re_deepstag.squeeze(-1)
                        detected_word_cur_re_deepstag = detected_word_cur_re_deepstag.reshape(
                            int(tx_data.shape[0] / num_bits_data),
                            n_users, num_bits_data).swapaxes(1, 2).reshape(
                            tx_data.shape[0], n_users)
                        if conf.ber_on_one_user >= 0:
                            ber_deepstag = calculate_ber(
                                detected_word_cur_re_deepstag[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits_data)
                        else:
                            ber_deepstag = calculate_ber(detected_word_cur_re_deepstag.cpu(), target.cpu(), num_bits_data)
                        ber_sum_deepstag[iteration] += ber_deepstag
                        ber_per_re_deepstag[iteration, re] = ber_deepstag

                if run_mhsa:
                    for iteration in range(iterations):
                        detected_word_cur_re_mhsa = detected_word_mhsa_list[iteration][:, :, re]
                        detected_word_cur_re_mhsa = detected_word_cur_re_mhsa.squeeze(-1)
                        detected_word_cur_re_mhsa = detected_word_cur_re_mhsa.reshape(int(tx_data.shape[0] / num_bits_data),
                                                                                      n_users, num_bits_data).swapaxes(1,
                                                                                                                  2).reshape(
                            tx_data.shape[0], n_users)
                        if conf.ber_on_one_user >= 0:
                            ber_mhsa = calculate_ber(
                                detected_word_cur_re_mhsa[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits_data)
                        else:
                            ber_mhsa = calculate_ber(detected_word_cur_re_mhsa.cpu(), target.cpu(), num_bits_data)
                        ber_sum_mhsa[iteration] += ber_mhsa
                        ber_per_re_mhsa[iteration, re] = ber_mhsa

                if conf.ber_on_one_user >= 0:
                    ber_lmmse = calculate_ber(
                        torch.from_numpy(detected_word_lmmse[:, conf.ber_on_one_user]).unsqueeze(-1),
                        target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits_data)
                    if run_sphere:
                        ber_sphere = calculate_ber(
                            torch.from_numpy(detected_word_sphere[:, conf.ber_on_one_user]).unsqueeze(-1),
                            target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits_data)
                else:
                    ber_lmmse = calculate_ber(torch.from_numpy(detected_word_lmmse), target.cpu(), num_bits_data)
                    if run_sphere:
                        ber_sphere = calculate_ber(torch.from_numpy(detected_word_sphere), target.cpu(), num_bits_data)

                ber_per_re_lmmse[re] = ber_lmmse

                if run_deeprx:
                    ber_sum_deeprx += ber_deeprx
                ber_sum_lmmse += ber_lmmse
                if run_sphere:
                    ber_sum_sphere += ber_sphere

            # LDPC decoding
            bler_list = [None] * iterations
            bler_deepsic_list = [None] * iterations
            bler_tdfdcnn_list = [None] * iterations
            bler_mhsa_list = [None] * iterations
            if conf.mcs > -1:
                # llrs_mat_lmmse = llrs_mat_lmmse * 0.0092
                if ldpc_k > 3824:
                    crc_length = 24
                else:
                    crc_length = 16
                codec = LDPC5GCodec(k=(ldpc_k + crc_length), n=ldpc_n)
                crc = CRC5GCodec(crc_length)
                for iteration in range(iterations):
                    llr_all_res = np.zeros((n_users, int(tx_data.shape[0] * conf.num_res)))
                    llr_all_res_tdfdcnn = np.zeros((n_users, int(tx_data.shape[0] * conf.num_res)))
                    llr_all_res_lmmse = np.zeros((n_users, int(tx_data.shape[0] * conf.num_res)))
                    llr_all_res_sphere = np.zeros((n_users, int(tx_data.shape[0] * conf.num_res)))
                    llr_all_res_deeprx = np.zeros((n_users, int(tx_data.shape[0] * conf.num_res)))
                    llr_all_res_deepsic = np.zeros((n_users, int(tx_data.shape[0] * conf.num_res)))
                    llr_all_res_mhsa = np.zeros((n_users, int(tx_data.shape[0] * conf.num_res)))
                    for re in range(conf.num_res):
                        # ESCNN
                        if pilot_data_ratio<=1:
                            llr_cur_re = llrs_mat_list[iteration][:, :, re, :]
                        else:
                            llr_cur_re = llrs_mat_list[iteration][:, relevant_indices(llrs_mat_list[iteration].shape[1],pilot_data_ratio), re, :]

                        if pilot_data_ratio == 1.5:
                            llr_cur_re[:, 1::2, :] *= -1

                        llr_cur_re = llr_cur_re.squeeze(-1)
                        llr_cur_re = llr_cur_re.reshape(int(tx_data.shape[0] / num_bits_data), n_users,
                                                        num_bits_data).swapaxes(1, 2).reshape(
                            tx_data.shape[0], n_users)
                        llr_all_res[:, re::conf.num_res] = llr_cur_re.swapaxes(0, 1).cpu()

                        # TDFDCNN
                        if conf.run_tdfdcnn:
                            llr_cur_re_tdfdcnn = llrs_mat_tdfdcnn_list[iteration][:, :, re, :]
                            llr_cur_re_tdfdcnn = llr_cur_re_tdfdcnn.squeeze(-1)
                            llr_cur_re_tdfdcnn = llr_cur_re_tdfdcnn.reshape(int(tx_data.shape[0] / num_bits_data), n_users,
                                                            num_bits_data).swapaxes(1, 2).reshape(
                                tx_data.shape[0], n_users)
                            llr_all_res_tdfdcnn[:, re::conf.num_res] = llr_cur_re_tdfdcnn.swapaxes(0, 1).cpu()

                        # lmmse
                        if pilot_data_ratio<=1:
                            llr_cur_re_lmmse = llrs_mat_lmmse[:, :, re, :]
                        else:
                            llr_cur_re_lmmse = llrs_mat_lmmse[:, relevant_indices(llrs_mat_lmmse.shape[1],pilot_data_ratio), re, :]
                        llr_cur_re_lmmse = llr_cur_re_lmmse.squeeze(-1)
                        llr_cur_re_lmmse = llr_cur_re_lmmse.reshape(int(tx_data.shape[0] / num_bits_data), n_users,
                                                                    num_bits_data).swapaxes(1, 2).reshape(
                            tx_data.shape[0], n_users)
                        llr_all_res_lmmse[:, re::conf.num_res] = llr_cur_re_lmmse.swapaxes(0, 1)

                        # Sphere
                        if pilot_data_ratio<=1:
                            llr_cur_re_sphere = llrs_mat_sphere[:, :, re, :]
                        else:
                            llr_cur_re_sphere = llrs_mat_sphere[:, relevant_indices(llrs_mat_sphere.shape[1],pilot_data_ratio), re, :]
                        llr_cur_re_sphere = llr_cur_re_sphere.squeeze(-1)
                        llr_cur_re_sphere = llr_cur_re_sphere.reshape(int(tx_data.shape[0] / num_bits_data), n_users,
                                                                      num_bits_data).swapaxes(1, 2).reshape(
                            tx_data.shape[0], n_users)
                        llr_all_res_sphere[:, re::conf.num_res] = llr_cur_re_sphere.swapaxes(0, 1)

                        # DeepSIC
                        if run_deepsic:
                            llr_cur_re_deepsic = llrs_mat_deepsic_list[iteration][:, :, re]
                            llr_cur_re_deepsic = llr_cur_re_deepsic.squeeze(-1)
                            llr_cur_re_deepsic = llr_cur_re_deepsic.reshape(int(tx_data.shape[0] / num_bits_data), n_users,
                                                                            num_bits_data).swapaxes(1, 2).reshape(
                                tx_data.shape[0], n_users)
                            llr_all_res_deepsic[:, re::conf.num_res] = llr_cur_re_deepsic.swapaxes(0, 1).cpu()

                        if run_mhsa:
                            llr_cur_re_mhsa = llrs_mat_mhsa_list[iteration][:, :, re]
                            llr_cur_re_mhsa = llr_cur_re_mhsa.squeeze(-1)
                            llr_cur_re_mhsa = llr_cur_re_mhsa.reshape(int(tx_data.shape[0] / num_bits_data), n_users,
                                                                      num_bits_data).swapaxes(1, 2).reshape(
                                tx_data.shape[0], n_users)
                            llr_all_res_mhsa[:, re::conf.num_res] = llr_cur_re_mhsa.swapaxes(0, 1).cpu()

                        # DeepRx
                        if run_deeprx:
                            llr_cur_re_deeprx = llrs_mat_deeprx[:, :, re, :]
                            llr_cur_re_deeprx = llr_cur_re_deeprx.squeeze(-1)
                            llr_cur_re_deeprx = llr_cur_re_deeprx.reshape(int(tx_data.shape[0] / num_bits_data), n_users,
                                                                          num_bits_data).swapaxes(1, 2).reshape(
                                tx_data.shape[0], n_users)
                            llr_all_res_deeprx[:, re::conf.num_res] = llr_cur_re_deeprx.swapaxes(0, 1).cpu()

                    num_slots = int(np.floor(llr_all_res.shape[1] / ldpc_n))
                    crc_count = 0
                    crc_count_tdfdcnn = 0
                    crc_count_lmmse = 0
                    crc_count_sphere = 0
                    crc_count_deeprx = 0
                    crc_count_deepsic = 0
                    crc_count_mhsa = 0
                    for slot in range(num_slots):
                        decodedwords = codec.decode(llr_all_res[:, slot * ldpc_n:(slot + 1) * ldpc_n])
                        crc_out = crc.decode(decodedwords)
                        crc_count += (~crc_out).numpy().astype(int).sum()
                        if conf.run_tdfdcnn:
                            decodedwords_tdfdcnn = codec.decode(llr_all_res_tdfdcnn[:, slot * ldpc_n:(slot + 1) * ldpc_n])
                            crc_out_tdfdcnn = crc.decode(decodedwords_tdfdcnn)
                            crc_count_tdfdcnn += (~crc_out_tdfdcnn).numpy().astype(int).sum()

                        decodedwords_lmmse = codec.decode(llr_all_res_lmmse[:, slot * ldpc_n:(slot + 1) * ldpc_n])
                        crc_out_lmmse = crc.decode(decodedwords_lmmse)
                        crc_count_lmmse += (~crc_out_lmmse).numpy().astype(int).sum()
                        if run_sphere:
                            decodedwords_sphere = codec.decode(llr_all_res_sphere[:, slot * ldpc_n:(slot + 1) * ldpc_n])
                            crc_out_sphere = crc.decode(decodedwords_sphere)
                            crc_count_sphere += (~crc_out_sphere).numpy().astype(int).sum()
                        if run_deeprx:
                            decodedwords_deeprx = codec.decode(llr_all_res_deeprx[:, slot * ldpc_n:(slot + 1) * ldpc_n])
                            crc_out_deeprx = crc.decode(decodedwords_deeprx)
                            crc_count_deeprx += (~crc_out_deeprx).numpy().astype(int).sum()
                        if run_deepsic:
                            decodedwords_deepsic = codec.decode(
                                llr_all_res_deepsic[:, slot * ldpc_n:(slot + 1) * ldpc_n])
                            crc_out_deepsic = crc.decode(decodedwords_deepsic)
                            crc_count_deepsic += (~crc_out_deepsic).numpy().astype(int).sum()
                        if run_mhsa:
                            decodedwords_mhsa = codec.decode(llr_all_res_mhsa[:, slot * ldpc_n:(slot + 1) * ldpc_n])
                            crc_out_mhsa = crc.decode(decodedwords_mhsa)
                            crc_count_mhsa += (~crc_out_mhsa).numpy().astype(int).sum()
                    bler_list[iteration] = crc_count / (num_slots * n_users)
                    total_bler_list[iteration].append(bler_list[iteration])

                    if conf.run_tdfdcnn:
                        bler_tdfdcnn_list[iteration] = crc_count_tdfdcnn / (num_slots * n_users)
                        total_bler_tdfdcnn_list[iteration].append(bler_tdfdcnn_list[iteration])
                    else:
                        bler_tdfdcnn_list[iteration] = 0
                        total_bler_tdfdcnn_list[iteration].append(bler_tdfdcnn_list[iteration])

                    if run_deepsic:
                        bler_deepsic_list[iteration] = crc_count_deepsic / (num_slots * n_users)
                        total_bler_deepsic_list[iteration].append(bler_deepsic_list[iteration])
                    else:
                        bler_deepsic_list[iteration] = 0
                        total_bler_deepsic_list[iteration].append(bler_deepsic_list[iteration])

                    if run_mhsa:
                        bler_mhsa_list[iteration] = crc_count_mhsa / (num_slots * n_users)
                        total_bler_mhsa_list[iteration].append(bler_mhsa_list[iteration])
                    else:
                        bler_mhsa_list[iteration] = 0
                        total_bler_mhsa_list[iteration].append(bler_mhsa_list[iteration])

                    if iteration == 0:
                        bler_lmmse = crc_count_lmmse / (num_slots * n_users)
                        total_bler_lmmse.append(bler_lmmse)
                        if run_sphere:
                            bler_sphere = crc_count_sphere / (num_slots * n_users)
                            total_bler_sphere.append(bler_sphere)
                        else:
                            bler_sphere = 0
                            total_bler_sphere.append(bler_sphere)

                        if run_deeprx:
                            bler_deeprx = crc_count_deeprx / (num_slots * n_users)
                            total_bler_deeprx.append(bler_deeprx)
                        else:
                            bler_deeprx = 0
                            total_bler_deeprx.append(bler_deeprx)


            else:
                for iteration in range(iterations):
                    bler_list[iteration] = 0
                    total_bler_list[iteration].append(0)
                    bler_deepsic_list[iteration] = 0
                    total_bler_deepsic_list[iteration].append(0)
                    total_bler_tdfdcnn_list[iteration].append(0)
                    total_bler_mhsa_list[iteration].append(0)
                bler_lmmse = 0
                total_bler_lmmse.append(0)
                total_bler_sphere.append(0)
                total_bler_deeprx.append(0)

            if PLOT_MI:
                for iteration in range(iterations):
                    mi = calc_mi(tx_data.cpu(), llrs_mat_list[iteration].cpu(), num_bits_data, n_users, num_res)
                    total_mi_list[iteration].append(mi)
                mi_deeprx = calc_mi(tx_data.cpu(), llrs_mat_deeprx.cpu(), num_bits_data, n_users, num_res)
                total_mi_deeprx.append(mi_deeprx)
                for iteration in range(iters_e2e_disp):
                    mi_e2e = calc_mi(tx_data.cpu(), llrs_mat_e2e_list[iteration].cpu(), num_bits_data, n_users, num_res)
                    total_mi_e2e_list[iteration].append(mi_e2e)
                mi_lmmse = calc_mi(tx_data.cpu(), llrs_mat_lmmse, num_bits_data, n_users, num_res)
                total_mi_lmmse.append(mi_lmmse)
            else:
                mi = 0
                mi_e2e = 0
                mi_deeprx = 0
                mi_lmmse = 0

            ber_list = [None] * iterations
            for iteration in range(iterations):
                # ber_list[iteration] = ber_sum[iteration] / (num_res - 2*conf.kernel_size)
                ber_list[iteration] = ber_sum[iteration] / num_res
                total_ber_list[iteration].append(ber_list[iteration])

            ber_e2e_list = [None] * iters_e2e_disp
            for iteration in range(iters_e2e_disp):
                ber_e2e_list[iteration] = ber_sum_e2e[iteration] / num_res
                total_ber_e2e_list[iteration].append(ber_e2e_list[iteration])

            ber_lmmse = ber_sum_lmmse / num_res
            ber_sphere = ber_sum_sphere / num_res
            ber_deeprx = ber_sum_deeprx / num_res

            ber_tdfdcnn_list = [None] * iterations
            for iteration in range(iterations):
                ber_tdfdcnn_list[iteration] = ber_sum_tdfdcnn[iteration] / num_res
                total_ber_tdfdcnn_list[iteration].append(ber_tdfdcnn_list[iteration])

            ber_deepsic_list = [None] * iterations
            for iteration in range(iterations):
                # ber_deepsic_list[iteration] = ber_sum_deepsic[iteration]  / (num_res - 2*conf.kernel_size)
                ber_deepsic_list[iteration] = ber_sum_deepsic[iteration] / num_res
                total_ber_deepsic_list[iteration].append(ber_deepsic_list[iteration])

            ber_deepsicmb_list = [None] * iterations
            for iteration in range(iterations):
                # ber_deepsicmb_list[iteration] = ber_sum_deepsicmb[iteration]  / (num_res - 2*conf.kernel_size)
                ber_deepsicmb_list[iteration] = ber_sum_deepsicmb[iteration] / num_res
                total_ber_deepsicmb_list[iteration].append(ber_deepsicmb_list[iteration])

            ber_deepstag_list = [None] * iterations * 2
            for iteration in range(iterations * 2):
                # ber_deepstag_list[iteration] = ber_sum_deepstag[iteration]  / (num_res - 2*conf.kernel_size)
                ber_deepstag_list[iteration] = ber_sum_deepstag[iteration] / num_res
                total_ber_deepstag_list[iteration].append(ber_deepstag_list[iteration])

            ber_mhsa_list = [None] * iterations
            for iteration in range(iterations):
                ber_mhsa_list[iteration] = ber_sum_mhsa[iteration] / num_res
                total_ber_mhsa_list[iteration].append(ber_mhsa_list[iteration])

            total_ber_deeprx.append(ber_deeprx)
            total_ber_lmmse.append(ber_lmmse)
            total_ber_sphere.append(ber_sphere)
            print(f'SNR={snr_cur}dB, Final SNR={Final_SNR}dB')
            print(f'current ESCNN: {block_ind, float(ber_list[iterations - 1]), mi}')
            if conf.mcs > -1:
                print(f'current ESCNN BLER: {block_ind, float(bler_list[iterations - 1]), mi}')
            if conf.run_e2e:
                print(f'curr DeepSICe2e: {block_ind, float(ber_e2e_list[iters_e2e_disp - 1]), mi_e2e}')
            if run_deeprx:
                print(f'current DeepRx: {block_ind, ber_deeprx.item(), mi_deeprx}')
                if conf.mcs > -1:
                    print(f'current DeepRx BLER: {block_ind, float(bler_deeprx), mi}')

            if conf.run_tdfdcnn:
                print(f'current TDFDCNN: {block_ind, float(ber_tdfdcnn_list[iterations - 1])}')
                if conf.mcs > -1:
                    print(f'current DeepSIC BLER: {block_ind, float(bler_tdfdcnn_list[iterations - 1])}')

            if run_deepsic:
                print(f'current DeepSIC: {block_ind, float(ber_deepsic_list[iterations - 1])}')
                if conf.mcs > -1:
                    print(f'current DeepSIC BLER: {block_ind, float(bler_deepsic_list[iterations - 1])}')
            if run_mhsa:
                print(f'current MHSA: {block_ind, float(ber_mhsa_list[iterations - 1])}')
                if conf.mcs > -1:
                    print(f'current MHSA BLER: {block_ind, float(bler_mhsa_list[iterations - 1])}')

            if conf.run_deepsicmb:
                print(f'current DeepSICMB: {block_ind, float(ber_deepsicmb_list[iterations - 1])}')
            if conf.run_deepstag:
                print(f'current DeepSTAG: {block_ind, float(ber_deepstag_list[iterations * 2 - 1])}')
            if mod_data == 4:
                print(f'current lmmse: {block_ind, ber_lmmse.item(), mi_lmmse}')
            else:
                print(f'current lmmse: {block_ind, ber_lmmse}')

            if conf.mcs > -1:
                print(f'current lmmse BLER: {block_ind, float(bler_lmmse), mi}')

            if run_sphere:
                print(f'current sphere: {block_ind, ber_sphere.item()}')
                if conf.mcs > -1:
                    print(f'current sphere BLER: {block_ind, float(bler_sphere), mi}')

        cfo_str = 'cfo=' + str(conf.cfo) + ' scs'

        fig_lmmse = plot_loss_and_LLRs([0] * len(train_loss_vect), [0] * len(val_loss_vect),
                                       torch.from_numpy(llrs_mat_lmmse),
                                       snr_cur, "lmmse", 0, train_samples, val_samples, mod_text, cfo_str, ber_lmmse,
                                       ber_lmmse, 0)


        if run_sphere:
            fig_sphere = plot_loss_and_LLRs([0] * len(train_loss_vect), [0] * len(val_loss_vect),
                                           torch.from_numpy(llrs_mat_sphere),
                                           snr_cur, "Sphere", 0, train_samples, val_samples, mod_text, cfo_str,
                                           ber_sphere, ber_lmmse, 0)

        if run_deeprx:
            fig_deeprx = plot_loss_and_LLRs(train_loss_vect_deeprx, val_loss_vect_deeprx, llrs_mat_deeprx, snr_cur,
                                            "DeepRx", 3,
                                            train_samples, val_samples, mod_text, cfo_str, ber_deeprx, ber_lmmse, 0)
        if conf.run_tdcnn:
            fig_tdcnn = plot_loss_and_LLRs(train_loss_vect_tdcnn, val_loss_vect_tdcnn, torch.zeros_like(llrs_mat_list[0]), snr_cur,
                                            "TDCNN", 3,
                                            train_samples, val_samples, mod_text, cfo_str, ber_deeprx, ber_lmmse, 0)
        if conf.run_tdfdcnn:
            for iteration in range(iterations):
                fig_tdfdcnn = plot_loss_and_LLRs(train_loss_vect_tdfdcnn, val_loss_vect_tdfdcnn, llrs_mat_list[iteration], snr_cur,
                                               "ESCNN",
                                               conf.kernel_size, train_samples, val_samples, mod_text, cfo_str,
                                               ber_list[iteration],
                                               ber_lmmse, iteration)

        if run_deepsic:
            for iteration in range(iterations):
                fig_deepsic = plot_loss_and_LLRs(train_loss_vect_deepsic, val_loss_vect_deepsic,
                                                 llrs_mat_deepsic_list[iteration], snr_cur, "DeepSIC", 3,
                                                 train_samples, val_samples, mod_text, cfo_str,
                                                 ber_deepsic_list[iteration], ber_lmmse, conf.iterations)
        if run_mhsa:
            for iteration in range(iterations):
                fig_mhsa = plot_loss_and_LLRs(train_loss_vect_mhsa, val_loss_vect_mhsa, llrs_mat_mhsa_list[iteration],
                                              snr_cur, "MHSA", 3,
                                              train_samples, val_samples, mod_text, cfo_str, ber_mhsa_list[iteration],
                                              ber_lmmse, conf.iterations)

        if conf.run_deepsicmb:
            for iteration in range(iterations):
                fig_deepsicmb = plot_loss_and_LLRs(train_loss_vect_deepsicmb, val_loss_vect_deepsicmb,
                                                   llrs_mat_deepsicmb_list[iteration], snr_cur, "DeepSICMB", 3,
                                                   train_samples, val_samples, mod_text, cfo_str,
                                                   ber_deepsicmb_list[iteration], ber_lmmse, conf.iterations)
        if conf.run_deepstag:
            for iteration in range(iterations * 2):
                fig_deepstag = plot_loss_and_LLRs(train_loss_vect_deepstag, val_loss_vect_deepstag,
                                                  llrs_mat_deepstag_list[iteration], snr_cur, "DeepSTAG", 3,
                                                  train_samples, val_samples, mod_text, cfo_str,
                                                  ber_deepstag_list[iteration], ber_lmmse, conf.iterations)
        if conf.run_e2e:
            for iteration in range(iters_e2e_disp):
                fig_e2e = plot_loss_and_LLRs(train_loss_vect_e2e, val_loss_vect_e2e, llrs_mat_e2e_list[iteration],
                                             snr_cur,
                                             "DeepSICe2e", conf.kernel_size, train_samples, val_samples, mod_text,
                                             cfo_str,
                                             ber_e2e_list[iteration], ber_lmmse, iteration)
        for iteration in range(iterations):
            fig_escnn = plot_loss_and_LLRs(train_loss_vect, val_loss_vect, llrs_mat_list[iteration], snr_cur, "ESCNN",
                                             conf.kernel_size, train_samples, val_samples, mod_text, cfo_str,
                                             ber_list[iteration],
                                             ber_lmmse, iteration)
            if conf.use_film:
                fig_film = plot_loss_and_LLRs(train_loss_vect_film, val_loss_vect_film, llrs_mat_list[iteration], snr_cur,
                                               "FILM",
                                               conf.kernel_size, train_samples, val_samples, mod_text, cfo_str,
                                               ber_list[iteration],
                                               ber_lmmse, iteration)

        data = {
            "SNR_range": SNR_range[:len(total_ber_lmmse)],
            "total_ber_lmmse": total_ber_lmmse,
            "total_ber_deeprx": total_ber_deeprx,
            "total_ber_sphere": total_ber_sphere,
        }

        # Add total_ber from total_ber_list with suffix _1, _2, ...
        for i in range(conf.iterations):
            data[f"total_ber_{i + 1}"] = total_ber_list[i]

        if conf.run_e2e:
            for i in range(iters_e2e_disp):
                data[f"total_ber_e2e_{i + 1}"] = total_ber_e2e_list[i]

        if run_deepsic:
            for i in range(conf.iterations):
                data[f"total_ber_deepsic_{i + 1}"] = total_ber_deepsic_list[i]

        if conf.run_deepsicmb:
            for i in range(conf.iterations):
                data[f"total_ber_deepsicmb_{i + 1}"] = total_ber_deepsicmb_list[i]

        if conf.run_deepstag:
            for i in range(conf.iterations):
                data[f"total_ber_deepstag_{i + 1}"] = total_ber_deepstag_list[i]

        if conf.run_mhsa:
            for i in range(conf.iterations):
                data[f"total_ber_mhsa_{i + 1}"] = total_ber_mhsa_list[i]

        if conf.run_tdfdcnn:
            for i in range(conf.iterations):
                data[f"total_ber_tdfdcnn_{i + 1}"] = total_ber_tdfdcnn_list[i]

        df = pd.DataFrame(data)

        data_bler = {
            "SNR_range": SNR_range[:len(total_ber_lmmse)],
            "total_ber_lmmse": total_bler_lmmse,
            "total_ber_sphere": total_bler_sphere,
            "total_ber_deeprx": total_bler_deeprx,
        }

        for i in range(conf.iterations):
            data_bler[f"total_ber_{i + 1}"] = total_bler_list[i]

        if run_deepsic:
            for i in range(conf.iterations):
                data_bler[f"total_ber_deepsic_{i + 1}"] = total_bler_deepsic_list[i]

        if conf.run_tdfdcnn:
            for i in range(conf.iterations):
                data_bler[f"total_ber_tdfdcnn_{i + 1}"] = total_bler_tdfdcnn_list[i]

        if run_mhsa:
            for i in range(conf.iterations):
                data_bler[f"total_ber_mhsa_{i + 1}"] = total_bler_mhsa_list[i]

        df_bler = pd.DataFrame(data_bler)

        # print('\n'+title_string)
        title_string = (chan_text + ', ' + mod_text + ', #TRN=' + str(train_samples) + ", #REs=" + str(
            conf.num_res) + ', #UEs=' + str(n_users) + '\n ' +
                        cfo_str + ', Epo=' + str(epochs) + ', #iter=' + str(
                    iterations) + ', ker=' + str(conf.kernel_size) + ', Clip=' + str(
                    conf.clip_percentage_in_tx) + '%')

        # plot BER per RE for first iteration:
        if conf.ber_on_one_user >= 0:
            add_text = 'user ' + str(conf.ber_on_one_user)
        else:
            add_text = 'all users'

        colors = ['g', 'r', 'k', 'b', 'yellow', 'pink']
        fig, ax = plt.subplots()
        for iter in range(iterations):
            plt.plot(np.arange(conf.num_res), ber_per_re[iter, :], linestyle='-', color=colors[iter],
                     label='BER ' + add_text + ', iter' + str(iter))
        ax.legend()
        ax.set_title('ESCNN, ' + title_string)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        fig_escnn_per_re = fig

        fig, ax = plt.subplots()
        for iter in range(iterations):
            plt.plot(np.arange(conf.num_res), ber_per_re_deepsic[iter, :], linestyle='-', color=colors[iter],
                     label='BER ' + add_text + ', iter' + str(iter))
        ax.legend()
        ax.set_title('DeepSIC, ' + title_string)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        fig_deepsic_per_re = fig

        fig, ax = plt.subplots()
        for iter in range(iterations):
            plt.plot(np.arange(conf.num_res), ber_per_re_deepsicmb[iter, :], linestyle='-', color=colors[iter],
                     label='BER ' + add_text + ', iter' + str(iter))
        ax.legend()
        ax.set_title('DeepSICMB, ' + title_string)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        fig_deepsicmb_per_re = fig

        fig, ax = plt.subplots()
        for iter in range(iterations):
            plt.plot(np.arange(conf.num_res), ber_per_re_deepstag[iter, :], linestyle='-', color=colors[iter],
                     label='BER ' + add_text + ', iter' + str(iter))
        ax.legend()
        ax.set_title('DeepSTAG, ' + title_string)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        fig_deepstag_per_re = fig

        fig, ax = plt.subplots()
        for iter in range(iterations):
            plt.plot(np.arange(conf.num_res), ber_per_re_mhsa[iter, :], linestyle='-', color=colors[iter],
                     label='BER ' + add_text + ', iter' + str(iter))
        ax.legend()
        ax.set_title('MHSA, ' + title_string)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        fig_mhsa_per_re = fig

        fig, ax = plt.subplots()
        plt.plot(np.arange(conf.num_res), ber_per_re_lmmse, linestyle='-', color='g', label='BER ' + add_text)
        ax.legend()
        ax.set_title('lmmse, ' + title_string)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        fig_lmmse_per_re = fig

        title_string = title_string.replace("\n", "")
        title_string = title_string.replace(",", "")
        title_string = title_string.replace(" ", "_")
        # Add spatial correlation indicator
        corr_map = {'none': 'No', 'low': 'Lo', 'medium': 'Med', 'medium_a': 'MedA', 'high': 'Hi', 'custom': 'Cust'}
        corr_level = getattr(conf, 'spatial_correlation', 'none')
        title_string = title_string + '_C=' + corr_map.get(corr_level, 'No')
        title_string = title_string + '_n_ants=' + str(conf.n_ants)
        # title_string = title_string + '_FFT_size=' + str(FFT_size)
        # title_string = title_string + '_sep_pilots_deeprx=' + str(conf.separate_pilots)
        title_string = title_string + '_' + conf.which_augment
        if conf.mcs > -1:
            title_string = title_string + f'_Rc={code_rate:.2f}'
        # title_string = title_string + '_scale_' + str(conf.scale_input)
        title_string = title_string + '_FILM_' + str(conf.use_film)
        title_string = title_string + '_PDR_' + str(pilot_data_ratio)
        # title_string = title_string + '_ONV_' + str(conf.override_noise_var)
        title_string = title_string + '_64Q16perc_' + str(conf.make_64QAM_16QAM_percentage)
        title_string = title_string + '_incPrimeMod_' + str(int(conf.increase_prime_modulation))
        title_string = title_string + '_' + conf.cur_str
        title_string = title_string + '_seed=' + str(conf.channel_seed)
        title_string = title_string + '_SNR=' + str(conf.snr)
        title_string = formatted_date + title_string
        output_dir = os.path.join(os.getcwd(), '..', 'Scratchpad')
        file_path = os.path.abspath(os.path.join(output_dir, title_string) + ".csv")
        df.to_csv(file_path, index=False)
        if conf.mcs > -1:
            file_path_bler = os.path.abspath(os.path.join(output_dir, title_string) + "_bler.csv")
            df_bler.to_csv(file_path_bler, index=False)

        if snr_cur in conf.save_loss_plot_snr:
            title_string_cur = "escnn_" + title_string
            file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
            fig_escnn.savefig(file_path)
            if conf.use_film:
                title_string_cur = "film_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_film.savefig(file_path)

            title_string_cur = "escnn_per_re_" + title_string
            file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
            fig_escnn_per_re.savefig(file_path)

            title_string_cur = "lmmse_" + title_string
            file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
            fig_lmmse.savefig(file_path)
            title_string_cur = "lmmse_per_re_" + title_string
            file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
            fig_lmmse_per_re.savefig(file_path)

            if run_deepsic:
                title_string_cur = "deepsic_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepsic.savefig(file_path)
                title_string_cur = "deepsic_per_re_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepsic_per_re.savefig(file_path)
            if run_mhsa:
                title_string_cur = "mhsa_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_mhsa.savefig(file_path)
                title_string_cur = "mhsa_per_re_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_mhsa_per_re.savefig(file_path)
            if conf.run_deepsicmb:
                title_string_cur = "deepsicmb_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepsicmb.savefig(file_path)
                title_string_cur = "deepsicmb_per_re_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepsicmb_per_re.savefig(file_path)
            if conf.run_deepstag:
                title_string_cur = "deepstag_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepstag.savefig(file_path)
                title_string_cur = "deepstag_per_re_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepstag_per_re.savefig(file_path)
            if run_deeprx:
                title_string_cur = "deeprx_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deeprx.savefig(file_path)
            if conf.run_tdcnn:
                title_string_cur = "tdcnn_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_tdcnn.savefig(file_path)

            if conf.run_e2e:
                title_string_cur = "e2e" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_e2e.savefig(file_path)

            if conf.run_tdfdcnn:
                title_string_cur = "tdfdcnn" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_tdfdcnn.savefig(file_path)

    markers = ['o', '*', 'x', 'D', '+', 'o']
    dashes = [':', '-.', '--', '-', '-', '-']
    if PLOT_MI:
        for iteration in range(iterations):
            plt.semilogy(SNR_range, total_mi_list[iteration], linestyle=dashes[iteration], marker=markers[iteration],
                         color='g', label='ESCNN' + str(iteration))
        if run_deeprx:
            plt.semilogy(SNR_range, total_mi_deeprx, '-o', color='c', label='DeepRx')
        if mod_data == 4:
            plt.semilogy(SNR_range, total_mi_lmmse, '-o', color='r', label='lmmse')
        plt.xlabel('SNR (dB)')
        plt.ylabel('MI')
        title_string = (chan_text + ', ' + mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(
            val_samples) + ", #REs=" + str(
            conf.num_res) + ', #UEs=' + str(n_users) + '\n ' +
                        cfo_str + ', Epochs=' + str(epochs) + ', #iters_e2e=' + str(
                    conf.iters_e2e) + ', CNN kernel size=' + str(conf.kernel_size))
        plt.title(title_string, fontsize=10)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    snr_at_target_list = np.zeros(iterations)
    snr_at_target_e2e_list = np.zeros(iters_e2e)
    snr_at_target_deepsic_list = np.zeros(iterations)
    snr_at_target_tdfdcnn_list = np.zeros(iterations)
    snr_at_target_deepsicmb_list = np.zeros(iterations)
    snr_at_target_deepstag_list = np.zeros(iterations)
    snr_at_target_mhsa_list = np.zeros(iterations)

    if len(SNR_range) > 1:
        bler_target = 0.01
        for iteration in range(iterations):
            interp_func = interp1d(total_ber_list[iteration], SNR_range, kind='linear', fill_value="extrapolate")
            snr_at_target_list[iteration] = np.round(interp_func(bler_target), 1)
        if conf.run_e2e:
            for iteration in range(iters_e2e_disp):
                interp_func = interp1d(total_ber_e2e_list[iteration], SNR_range, kind='linear',
                                       fill_value="extrapolate")
                snr_at_target_e2e_list[iteration] = np.round(interp_func(bler_target), 1)

        if conf.run_tdfdcnn:
            for iteration in range(iterations):
                interp_func = interp1d(total_ber_tdfdcnn_list[iteration], SNR_range, kind='linear',
                                       fill_value="extrapolate")
                snr_at_target_tdfdcnn_list[iteration] = np.round(interp_func(bler_target), 1)


        if run_deepsic:
            for iteration in range(iterations):
                interp_func = interp1d(total_ber_deepsic_list[iteration], SNR_range, kind='linear',
                                       fill_value="extrapolate")
                snr_at_target_deepsic_list[iteration] = np.round(interp_func(bler_target), 1)
        if conf.run_deepsicmb:
            for iteration in range(iterations):
                interp_func = interp1d(total_ber_deepsicmb_list[iteration], SNR_range, kind='linear',
                                       fill_value="extrapolate")
                snr_at_target_deepsicmb_list[iteration] = np.round(interp_func(bler_target), 1)
        if conf.run_deepstag:
            for iteration in range(iterations):
                interp_func = interp1d(total_ber_deepstag_list[iteration], SNR_range, kind='linear',
                                       fill_value="extrapolate")
                snr_at_target_deepstag_list[iteration] = np.round(interp_func(bler_target), 1)
        if conf.run_mhsa:
            for iteration in range(iterations):
                interp_func = interp1d(total_ber_mhsa_list[iteration], SNR_range, kind='linear',
                                       fill_value="extrapolate")
                snr_at_target_mhsa_list[iteration] = np.round(interp_func(bler_target), 1)
        if run_deeprx:
            interp_func = interp1d(total_ber_deeprx, SNR_range, kind='linear', fill_value="extrapolate")
            snr_at_target_deeprx = np.round(interp_func(bler_target), 1)
        interp_func = interp1d(total_ber_lmmse, SNR_range, kind='linear', fill_value="extrapolate")
        snr_at_target_lmmse = np.round(interp_func(bler_target), 1)
        interp_func = interp1d(total_ber_sphere, SNR_range, kind='linear', fill_value="extrapolate")
        snr_at_target_sphere = np.round(interp_func(bler_target), 1)
    else:
        for iteration in range(iterations):
            snr_at_target_list[iteration] = float('inf')
        if conf.run_e2e:
            for iteration in range(iters_e2e_disp):
                snr_at_target_e2e_list[iteration] = float('inf')
        if run_deeprx:
            snr_at_target_deeprx = float('inf')
        if conf.run_tdfdcnn:
            for iteration in range(iterations):
                snr_at_target_tdfdcnn_list[iteration] = float('inf')
        if run_deepsic:
            for iteration in range(iterations):
                snr_at_target_deepsic_list[iteration] = float('inf')
        if conf.run_deepsicmb:
            for iteration in range(iterations):
                snr_at_target_deepsicmb_list[iteration] = float('inf')
        if conf.run_deepstag:
            for iteration in range(iterations):
                snr_at_target_deepstag_list[iteration] = float('inf')
        if run_mhsa:
            for iteration in range(iterations):
                snr_at_target_mhsa_list[iteration] = float('inf')
        snr_at_target_lmmse = float('inf')
        snr_at_target_sphere = float('inf')

    if not conf.plot_only_last_iteration:
        iterations_for_plot = list(range(iterations))
    else:
        iterations_for_plot = [iterations - 1]

    for iteration in iterations_for_plot:
        plt.semilogy(SNR_range, total_ber_list[iteration], linestyle=dashes[iteration], marker=markers[iteration],
                     color='g',
                     label='ESCNN' + str(iteration + 1) + ', SNR @1%=' + str(snr_at_target_list[iteration]))
    if conf.run_e2e:
        if conf.no_probs:
            e2e_text = 'NoProbs'
        else:
            e2e_text = 'Deepe2e'

        for iteration in range(iters_e2e_disp):
            plt.semilogy(SNR_range, total_ber_e2e_list[iteration], linestyle=dashes[iteration],
                         marker=markers[iteration], color='m',
                         label=e2e_text + str(iteration + 1) + ', SNR @1%=' + str(snr_at_target_e2e_list[iteration]))
    if run_deepsic:
        for iteration in range(iterations):
            plt.semilogy(SNR_range, total_ber_deepsic_list[iteration], linestyle=dashes[iteration],
                         marker=markers[iteration], color='orange',
                         label='DeepSIC' + str(iteration + 1) + ', SNR @1%=' + str(
                             snr_at_target_deepsic_list[iteration]))
    if conf.run_tdfdcnn:
        for iteration in range(iterations):
            plt.semilogy(SNR_range, total_ber_tdfdcnn_list[iteration], linestyle=dashes[iteration],
                         marker=markers[iteration], color='orange',
                         label='TDFDCNN' + str(iteration + 1) + ', SNR @1%=' + str(
                             snr_at_target_deepsic_list[iteration]))
    if conf.run_deepsicmb:
        for iteration in range(iterations):
            plt.semilogy(SNR_range, total_ber_deepsicmb_list[iteration], linestyle=dashes[iteration],
                         marker=markers[iteration], color='black',
                         label='DeepSICMB' + str(iteration + 1) + ', SNR @1%=' + str(
                             snr_at_target_deepsicmb_list[iteration]))
    if conf.run_deepstag:
        for iteration in range(iterations):
            plt.semilogy(SNR_range, total_ber_deepstag_list[iteration], linestyle=dashes[iteration],
                         marker=markers[iteration], color='pink',
                         label='DeepSTAG' + str(iteration + 1) + ', SNR @1%=' + str(
                             snr_at_target_deepstag_list[iteration]))
    if run_mhsa:
        for iteration in range(iterations):
            plt.semilogy(SNR_range, total_ber_mhsa_list[iteration], linestyle=dashes[iteration],
                         marker=markers[iteration], color='orange',
                         label='MHSA' + str(iteration + 1) + ', SNR @1%=' + str(snr_at_target_mhsa_list[iteration]))
    if run_deeprx:
        plt.semilogy(SNR_range, total_ber_deeprx, '-o', color='c',
                     label='DeepRx,   SNR @1%=' + str(snr_at_target_deeprx))

    plt.semilogy(SNR_range, total_ber_lmmse, '-o', color='r', label='lmmse,    SNR @1%=' + str(snr_at_target_lmmse))
    plt.semilogy(SNR_range, total_ber_sphere, '-o', color='brown',
                 label=modulator_text + ',        SNR @1%=' + str(snr_at_target_sphere))

    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    title_string = (chan_text + ', ' + mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(
        val_samples) + ", #REs=" + str(
        conf.num_res) + ', #UEs=' + str(n_users) + '\n ' +
                    cfo_str + ', Epochs=' + str(epochs) + ', #iterations=' + str(
                iterations) + ', CNN kernel size=' + str(conf.kernel_size) + ', Clip=' + str(
                conf.clip_percentage_in_tx) + '%')
    plt.title(title_string, fontsize=10)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Look at the weights:
    # print(escnn_trainer.detector[0][0].shared_backbone.fc.weight)
    # print(escnn_trainer.detector[1][0].instance_heads[0].fc1.weight[0])

    return total_ber_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    args = parser.parse_args()

    # Reload config singleton with the provided config file (or default)
    conf.reload_config(args.config)

    # Now conf has the updated config, proceed as before
    assert not (
                conf.no_probs and conf.iters_e2e != 1 and conf.full_e2e == True), "Assert: No probs only works with 1 iteration or with full e2e"
    assert not (
                conf.no_probs and conf.iters_e2e != 1 and conf.full_e2e == True), "Assert: No probs only works with 1 iteration or with full e2e"


    start_time = time.time()
    num_bits_data, _ = get_mcs(conf.mcs)
    num_bits_data = int(num_bits_data)

    if conf.mod_pilot > 0:
        num_bits_pilot = int(np.log2(conf.mod_pilot))
    else:
        num_bits_pilot = num_bits_data

    escnn_trainer = ESCNNTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    deepsice2e_trainer = DeepSICe2eTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    deeprx_trainer = DeepRxTrainer(conf.num_res, conf.n_users, conf.n_ants)
    deepsic_trainer = DeepSICTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    deepsicmb_trainer = DeepSICMBTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    deepstag_trainer = DeepSTAGTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    mhsa_trainer = MHSATrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    tdcnn_trainer = TDCNNTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    tdfdcnn_trainer = TDFDCNNTrainer(num_bits_pilot, conf.n_users, conf.n_ants)

    # For measuring number of parameters
    # def iter_modules(obj):
    #     """Recursively yield nn.Modules from nested lists/tuples."""
    #     import torch.nn as nn
    #     if isinstance(obj, nn.Module):
    #         yield obj
    #     elif isinstance(obj, (list, tuple)):
    #         for item in obj:
    #             yield from iter_modules(item)
    #
    #
    # detectors = escnn_trainer.detector
    # total_params = sum(
    #     p.numel()
    #     for module in iter_modules(detectors)  # your nested structure
    #     for p in module.parameters()
    #     if p.requires_grad
    # )
    # print(f"Parameters ESCNN: {total_params}")
    #
    # detectors = deeprx_trainer.detector
    # total_params = sum(
    #     p.numel()
    #     for module in iter_modules(detectors)  # your nested structure
    #     for p in module.parameters()
    #     if p.requires_grad
    # )
    # print(f"Parameters DeepRx: {total_params}")
    #
    # detectors = deepsic_trainer.detector
    # total_params = sum(
    #     p.numel()
    #     for module in iter_modules(detectors)  # your nested structure
    #     for p in module.parameters()
    #     if p.requires_grad
    # )
    # print(f"Parameters DeepSIC: {total_params}")
    #
    # detectors = deepsicmb_trainer.detector
    # total_params = sum(
    #     p.numel()
    #     for module in iter_modules(detectors)  # your nested structure
    #     for p in module.parameters()
    #     if p.requires_grad
    # )
    # print(f"Parameters DeepSICMB: {total_params}")
    #
    # detectors = deepstag_trainer.det_conv
    # total_params_conv = sum(
    #     p.numel()
    #     for module in iter_modules(detectors)  # your nested structure
    #     for p in module.parameters()
    #     if p.requires_grad
    # )
    # detectors = deepstag_trainer.det_re
    # total_params_re = sum(
    #     p.numel()
    #     for module in iter_modules(detectors)  # your nested structure
    #     for p in module.parameters()
    #     if p.requires_grad
    # )
    #
    # print(f"Parameters DeepSTAG: {total_params_conv+total_params_re}")

    # def iter_modules(obj):
    #     """Recursively yield nn.Modules from nested lists/tuples."""
    #     import torch.nn as nn
    #     if isinstance(obj, nn.Module):
    #         yield obj
    #     elif isinstance(obj, (list, tuple)):
    #         for item in obj:
    #             yield from iter_modules(item)
    #
    #
    # detectors = escnn_trainer.detector
    # total_params = sum(
    #     p.numel()
    #     for module in iter_modules(detectors)  # your nested structure
    #     for p in module.parameters()
    #     if p.requires_grad
    # )
    # print(f"Parameters ESCNN: {total_params}")
    #
    # detectors = mhsa_trainer.detector
    # total_params = sum(
    #     p.numel()
    #     for module in iter_modules(detectors)  # your nested structure
    #     for p in module.parameters()
    #     if p.requires_grad
    # )
    # print(f"Parameters MHSA: {total_params}")

    run_evaluate(escnn_trainer, deepsice2e_trainer, deeprx_trainer, deepsic_trainer, deepsicmb_trainer,
                 deepstag_trainer, mhsa_trainer, tdcnn_trainer, tdfdcnn_trainer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    days = int(elapsed_time // (24 * 3600))
    elapsed_time %= 24 * 3600
    hours = int(elapsed_time // 3600)
    elapsed_time %= 3600
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")