
import os
from pathlib import Path
import time
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.detectors.deepsice2e.deepsice2e_trainer import DeepSICe2eTrainer
from python_code.detectors.deeprx.deeprx_trainer import DeepRxTrainer
from python_code.detectors.deepsicsb.deepsicsb_trainer import DeepSICSBTrainer


from typing import List

import numpy as np
import torch
from python_code import DEVICE, conf
from python_code.utils.metrics import calculate_ber
import matplotlib.pyplot as plt
from python_code.utils.constants import (IS_COMPLEX, TRAIN_PERCENTAGE, CFO_COMP, GENIE_CFO,
                                         FFT_size, FIRST_CP, CP, NUM_SYMB_PER_SLOT, NUM_SAMPLES_PER_SLOT, PLOT_MI,
                                         PLOT_CE_ON_DATA, N_ANTS)

import commpy.modulation as mod

from python_code.channel.modulator import BPSKModulator, QPSKModulator, QAM16Modulator
import pandas as pd

from python_code.channel.channel_dataset import ChannelModelDataset
from scipy.stats import entropy
from scipy.interpolate import interp1d

from python_code.detectors.sphere.sphere_decoder import SphereDecoder

from datetime import datetime
from scipy.io import savemat

import argparse




os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

base_dir = Path.home() / "Projects" / "Scratchpad"



def entropy_with_bin_width(data, bin_width):
    """Estimate entropy using histogram binning with a specified bin width."""
    min_x, max_x = np.min(data), np.max(data)
    bins = np.arange(min_x, max_x + bin_width, bin_width)  # Define bin edges
    hist, _ = np.histogram(data, bins=bins, density=True)  # Compute histogram
    hist = hist[hist > 0]  # Remove zero probabilities
    return entropy(hist, base=2)  # Compute entropy


def calc_mi(tx_data: np.ndarray, llrs_mat: np.ndarray, num_bits: int, n_users: int, num_res: int) -> np.ndarray:
    llr_1 = llrs_mat[:, :, :, :].squeeze(-1)
    llr_2 = llr_1.reshape(int(tx_data.shape[0] / num_bits), n_users, num_bits, num_res)
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
                       val_samples, mod_text, cfo_str, ber, ber_legacy, ber_legacy_genie, iteration):
    num_res = conf.num_res
    p_len = conf.epochs * (iteration + 1)
    if detector == 'DeepSIC':
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
        snr_cur) + ", #REs=" + str(num_res) + ', Interf=' + str(conf.interf_factor) + ', #UEs=' + str(
        conf.n_users) + '\n ' +
                    cfo_str + ', Epochs=' + str(conf.epochs) + iters_txt + ', CNN kernel size=' + str(kernel_size))

    axes[0].set_title(title_string, fontsize=8)
    axes[0].legend()
    axes[0].grid()

    axes[1].hist(llrs_mat.cpu().flatten(), bins=30, color='blue', edgecolor='black', alpha=0.7)
    if (detector == 'DeepSIC') or (detector == 'DeepSICe2e'):
        axes[1].set_xlabel('LLRs iteration ' + str(iteration + 1))
    else:
        axes[1].set_xlabel('LLRs')
    axes[1].set_ylabel('#Values')
    axes[1].grid()
    text = 'BER ' + detector + ':' + str(f"{ber:.4f}") + '\
             BER legacy:' + str(f"{ber_legacy:.4f}") + '\
             BER legacy genie:' + (f"{ber_legacy_genie:.4f}")
    # axes[2].text(0.5, 0.5, text, fontsize=12, ha="center", va="center")
    # axes[2].axis('off')
    fig.text(0.15, 0.02, text, ha="left", va="center", fontsize=8)
    # fig.text(0.15, 0.02, text, ha="left", va="center", fontsize=12)
    plt.tight_layout()
    plt.show()


def run_evaluate(deepsic_trainer, deepsice2e_trainer, deeprx_trainer, deepsicsb_trainer=None) -> List[float]:
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
    mod_pilot = conf.mod_pilot
    num_bits = int(np.log2(mod_pilot))
    n_users = conf.n_users
    iterations = conf.iterations
    iters_e2e = conf.iters_e2e
    epochs = conf.epochs

    if mod_pilot == 2:
        mod_text = 'BPSK'
    elif mod_pilot == 4:
        mod_text = 'QPSK'
    else:
        mod_text = [str(mod_pilot) + 'QAM']
        mod_text = mod_text[0]

    if conf.TDL_model[0] == 'N':
        chan_text = 'Flat'
    else:
        chan_text = 'TDL-'+ conf.TDL_model + '-' + str(int(round(float(conf.delay_spread) * 1e9)))

    now = datetime.now()
    formatted_date = now.strftime("%Y_%m_%d_%H_%M_")

    if conf.full_e2e:
        iters_e2e_disp = 1
    else:
        iters_e2e_disp = iters_e2e

    total_ber_list = [[] for _ in range(iterations)]
    total_ber_e2e_list = [[] for _ in range(iters_e2e_disp)]
    total_ber_deeprx = []
    total_ber_legacy = []
    total_ber_sphere = []
    total_ber_deepsicsb = []

    if PLOT_CE_ON_DATA:
        total_ber_legacy_ce_on_data = []
    total_ber_legacy_genie = []

    SNR_range = list(range(conf.snr, conf.snr + conf.num_snrs, conf.snr_step))
    total_mi_list = [[] for _ in range(iterations)]
    total_mi_e2e_list = [[] for _ in range(iters_e2e_disp)]
    total_mi_deeprx = []
    if mod_pilot == 4:
        total_mi_legacy = []
    Final_SNR = conf.snr + conf.num_snrs - 1
    for snr_cur in SNR_range:
        ber_sum = np.zeros(iterations)
        ber_per_re = np.zeros((iterations, conf.num_res))
        ber_sum_e2e = np.zeros(iters_e2e_disp)
        ber_sum_deeprx = 0
        ber_sum_legacy = 0
        ber_per_re_legacy = np.zeros(conf.num_res)
        ber_sum_sphere = 0
        ber_sum_deepsicsb = 0
        if PLOT_CE_ON_DATA:
            ber_sum_legacy_ce_on_data = 0
        ber_sum_legacy_genie = 0
        deepsic_trainer._initialize_detector(num_bits, n_users)  # For reseting the weights
        deepsice2e_trainer._initialize_detector(num_bits, n_users)
        deeprx_trainer._initialize_detector(num_bits, n_users)
        if conf.run_deepsicsb and deepsicsb_trainer is not None:
            deepsicsb_trainer._initialize_detector(num_bits, n_users)

        pilot_size = get_next_divisible(conf.pilot_size, num_bits * NUM_SYMB_PER_SLOT)
        pilot_chunk = int(pilot_size / np.log2(mod_pilot))
        channel_dataset = ChannelModelDataset(block_length=conf.block_length,
                                              pilots_length=pilot_size,
                                              blocks_num=conf.blocks_num,
                                              num_res=conf.num_res,
                                              fading_in_channel=conf.fading_in_channel,
                                              spatial_in_channel=conf.spatial_in_channel,
                                              delayspread_in_channel=conf.delayspread_in_channel,
                                              clip_percentage_in_tx=conf.clip_percentage_in_tx,
                                              cfo=conf.cfo,
                                              go_to_td=conf.go_to_td,
                                              cfo_and_clip_in_rx=conf.cfo_and_clip_in_rx,
                                              kernel_size=conf.kernel_size,
                                              n_users=n_users)

        transmitted_words, received_words, received_words_ce, hs, s_orig_words = channel_dataset.__getitem__(
            snr_list=[snr_cur], num_bits=num_bits, n_users
            =n_users, mod_pilot=mod_pilot)

        # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        # REs = np.arange(conf.num_res)
        # axes[0].plot(REs, np.abs(hs[0,0,0,:]), linestyle='-', color='b', label='Channel')
        # axes[0].set_ylabel('abs(channel)')
        # axes[0].grid()
        # axes[1].plot(REs, np.unwrap(np.angle((hs[0,0,0,:]))), linestyle='-', color='b', label='Channel')
        # axes[1].set_xlabel('REs')
        # axes[1].set_ylabel('angle(channel)')
        # axes[1].grid()
        # axes[0].set_title('Channel with ' + str(conf.num_res) + ' REs')
        # plt.show()

        train_samples = int(pilot_size * TRAIN_PERCENTAGE / 100)
        val_samples = pilot_size - train_samples

        # detect sequentially
        for block_ind in range(conf.blocks_num):
            print('*' * 20)
            # get current word and channel
            tx, h, rx, rx_ce, s_orig = transmitted_words[block_ind], hs[block_ind], received_words[block_ind], \
                received_words_ce[block_ind], s_orig_words[block_ind]

            if (conf.cfo != 0) & (CFO_COMP != 'NONE'):
                pointer = 0
                NUM_SLOTS = int(s_orig.shape[0] / NUM_SYMB_PER_SLOT)
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

                if (CFO_COMP == 'ON_CE'):
                    cfo_phase = 2 * np.pi * cfo_est * n / FFT_size  # CFO phase shift
                else:
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

                if (CFO_COMP == 'ON_CE'):
                    cfo_comp_vect = cfo_comp_vect[pilot_chunk:]
                else:  # 'ON_Y'
                    for i in range(s_orig.shape[0]):
                        rx[i, :, :] = rx[i, :, :] * cfo_comp_vect[i]
                        rx_ce[:, i, :, :] = rx_ce[:, i, :, :] * cfo_comp_vect[i]

            # Interleave real and imaginary parts of Rx into a real tensor
            if IS_COMPLEX:
                real_part = rx.real
                imag_part = rx.imag
                rx_real = torch.empty((rx.shape[0], rx.shape[1] * 2, rx.shape[2]), dtype=torch.float32)
                rx_real[:, 0::2, :] = real_part  # Real parts in even rows
                rx_real[:, 1::2, :] = imag_part  # Imaginary parts in odd rows
            else:
                rx_real = rx

            # split words into data and pilot part
            tx_pilot, tx_data = tx[:pilot_size], tx[pilot_size:]
            rx_pilot, rx_data = rx_real[:pilot_chunk], rx_real[pilot_chunk:]

            # online training main function
            if deepsic_trainer.is_online_training:
                if conf.train_on_ce_no_pilots:
                    H_all = torch.zeros(N_ANTS, conf.num_res, dtype=torch.complex64)
                    for re in range(conf.num_res):
                        rx_pilot_cur = rx[:pilot_chunk, :, re]
                        row_means = torch.mean(rx_pilot_cur, axis=1, keepdims=True)
                        row_means[row_means == 0] = 1e-12
                        H_all[:, re] = torch.mean(rx_pilot_cur / row_means, axis=0)

                    real_part = H_all.real
                    imag_part = H_all.imag
                    H_all_real = torch.empty((H_all.shape[0] * 2, H_all.shape[1]), dtype=torch.float32)
                    H_all_real[0::2, :] = real_part  # Real parts in even rows
                    H_all_real[1::2, :] = imag_part  # Imaginary parts in odd rows

                    H_repeated = H_all_real.unsqueeze(0).repeat(rx_pilot.shape[0], 1, 1)
                    rx_pilot_and_H = torch.cat((rx_pilot, H_repeated), dim=1)
                    train_loss_vect, val_loss_vect = deepsic_trainer._online_training(tx_pilot,
                                                                                      rx_pilot_and_H.to('cpu'),
                                                                                      num_bits, n_users, iterations,
                                                                                      epochs)
                elif conf.use_data_as_pilots:
                    H_all = torch.zeros(s_orig.shape[0], N_ANTS * conf.n_users, conf.num_res, dtype=torch.complex64)
                    for re in range(conf.num_res):
                        H = torch.zeros(s_orig.shape[0], N_ANTS, conf.n_users, dtype=torch.complex64)
                        for user in range(n_users):
                            s_orig_pilot = s_orig[:, user, re]
                            rx_pilot_ce_cur = rx_ce[user, :, :, re]

                            H[:, :, user] = (s_orig_pilot[:, None].conj() / (
                                    torch.abs(s_orig_pilot[:, None]) ** 2)) * rx_pilot_ce_cur  # shape: [56, 4]
                        H_all[:, :, re] = H.reshape(H.shape[0], N_ANTS * conf.n_users)
                    real_part = H_all.real
                    imag_part = H_all.imag
                    H_all_real = torch.empty((H_all.shape[0], H_all.shape[1] * 2, H_all.shape[2]),
                                             dtype=torch.float32)
                    H_all_real[:, 0::2, :] = real_part  # Real parts in even rows
                    H_all_real[:, 1::2, :] = imag_part  # Imaginary parts in odd rows
                    rx_pilot_and_H = torch.cat((rx_pilot, H_all_real[:pilot_chunk]), dim=1)

                    train_loss_vect, val_loss_vect = deepsic_trainer._online_training(tx_pilot,
                                                                                      rx_pilot_and_H.to('cpu'),
                                                                                      num_bits, n_users, iterations,epochs)
                else:
                    if not conf.enable_two_stage_train:
                        train_loss_vect, val_loss_vect = deepsic_trainer._online_training(tx_pilot, rx_pilot, num_bits,
                                                                                          n_users, iterations, epochs)
                    else:
                        tx_pilot_cur = tx_pilot[:int(pilot_chunk*num_bits/2),:,:]
                        rx_pilot_cur = rx_pilot[:int(pilot_chunk / 2), :, :]
                        train_loss_vect_1, val_loss_vect_1 = deepsic_trainer._online_training(tx_pilot_cur, rx_pilot_cur, num_bits,
                                                                                          n_users, iterations, int(epochs/2))
                        tx_pilot_cur = tx_pilot[int(pilot_chunk*num_bits/2):,:,:]
                        rx_pilot_cur = rx_pilot[int(pilot_chunk / 2):, :, :]
                        train_loss_vect_2, val_loss_vect_2 = deepsic_trainer._online_training(tx_pilot_cur, rx_pilot_cur, num_bits,
                                                                                          n_users, iterations, int(epochs/2))
                        train_loss_vect = train_loss_vect_1 + train_loss_vect_2
                        val_loss_vect = val_loss_vect_1 + val_loss_vect_2
                        pass


                if conf.train_on_ce_no_pilots:
                    H_repeated = H_all_real.unsqueeze(0).repeat(rx_data.shape[0], 1, 1)
                    rx_data_and_H = torch.cat((rx_data, H_repeated), dim=1)
                    detected_word_list, llrs_mat_list = deepsic_trainer._forward(rx_data_and_H, num_bits, n_users,
                                                                                 iterations)
                elif conf.use_data_as_pilots:
                    rx_data_and_H = torch.cat((rx_data, H_all_real[pilot_chunk:]), dim=1)
                    detected_word_list, llrs_mat_list = deepsic_trainer._forward(rx_data_and_H, num_bits, n_users,
                                                                                 iterations)
                else:
                    detected_word_list, llrs_mat_list = deepsic_trainer._forward(rx_data, num_bits, n_users, iterations)

            if conf.run_e2e:
                if deepsice2e_trainer.is_online_training:
                    train_loss_vect_e2e, val_loss_vect_e2e = deepsice2e_trainer._online_training(tx_pilot, rx_pilot,
                                                                                                 num_bits, n_users,
                                                                                                 iters_e2e, epochs)
                    detected_word_e2e_list, llrs_mat_e2e_list = deepsice2e_trainer._forward(rx_data, num_bits, n_users,
                                                                                            iters_e2e)

            if conf.run_deeprx:
                if deeprx_trainer.is_online_training:
                    train_loss_vect_deeprx, val_loss_vect_deeprx = deeprx_trainer._online_training(tx_pilot, rx_pilot,
                                                                                                   num_bits, n_users,
                                                                                                   iterations, epochs)
                    detected_word_deeprx, llrs_mat_deeprx = deeprx_trainer._forward(rx_data, num_bits, n_users,
                                                                                    iterations)
            if conf.run_deepsicsb and deepsicsb_trainer is not None:
                if deepsicsb_trainer.is_online_training:
                    train_loss_vect_deepsicsb, val_loss_vect_deepsicsb = deepsicsb_trainer._online_training(
                        tx_pilot, rx_pilot, num_bits, n_users, iterations, epochs)
                    detected_word_deepsicsb, llrs_mat_deepsicsb = deepsicsb_trainer._forward(rx_data, num_bits, n_users,
                                                                                         iterations)
            # CE Based
            # train_loss_vect = [0] * epochs
            # val_loss_vect = [0] * epochs
            rx_data_c = rx[pilot_chunk:].cpu()
            llrs_mat_legacy = np.zeros((rx_data_c.shape[0], num_bits * n_users, num_res, 1))

            for re in range(conf.num_res):
                # Regular CE
                H = torch.zeros_like(h[:, :, re])
                for user in range(n_users):
                    rx_pilot_ce_cur = rx_ce[user, :pilot_chunk, :, re]
                    s_orig_pilot = s_orig[:pilot_chunk, user, re]
                    H[:, user] = 1 / s_orig_pilot.shape[0] * (s_orig_pilot[:, None].conj() / (
                            torch.abs(s_orig_pilot[:, None]) ** 2) * rx_pilot_ce_cur).sum(dim=0)
                H = H.cpu().numpy()

                H_Ht = H @ H.T.conj()
                H_Ht_inv = np.linalg.pinv(H_Ht)
                H_pi = torch.tensor(H.T.conj() @ H_Ht_inv)
                equalized = torch.zeros(rx_data_c.shape[0], tx_data.shape[1], dtype=torch.cfloat)
                for i in range(rx_data_c.shape[0]):
                    equalized[i, :] = torch.matmul(H_pi, rx_data_c[i, :, re])
                detected_word_legacy = np.zeros((int(equalized.shape[0] * np.log2(mod_pilot)), equalized.shape[1]))

                if mod_pilot == 2:
                    for i in range(equalized.shape[1]):
                        detected_word_legacy[:, i] = torch.from_numpy(
                            BPSKModulator.demodulate(-torch.sign(equalized[:, i].real).numpy()))
                elif mod_pilot == 4:
                    # qam = mod.QAMModem(mod_pilot)
                    for user in range(n_users):
                        detected_word_legacy[:, user], llr_out = QPSKModulator.demodulate(equalized[:, user].numpy())
                        llrs_mat_legacy[:, (user * num_bits):((user + 1) * num_bits), re, :] = llr_out.reshape(
                            int(llr_out.shape[0] / num_bits), num_bits, 1)
                elif mod_pilot == 16:
                    for user in range(n_users):
                        detected_word_legacy[:, user], llr_out = QAM16Modulator.demodulate(equalized[:, user].numpy())
                        llrs_mat_legacy[:, (user * num_bits):((user + 1) * num_bits), re, :] = llr_out.reshape(
                            int(llr_out.shape[0] / num_bits), num_bits, 1)
                else:
                    print('Unknown modulator')

                # Sphere:
                if conf.sphere_radius == 'inf':
                    radius = np.inf
                    modulator_text = 'MAP'
                else:
                    radius = float(conf.sphere_radius)
                    modulator_text = 'Sphere, Radius=' + str(conf.sphere_radius)

                if (conf.n_users == 4) & (N_ANTS == 4):
                    detected_word_sphere = SphereDecoder(H, rx_data_c[:, :, re].numpy(), radius)
                else:
                    detected_word_sphere = np.zeros_like(detected_word_legacy)

                if PLOT_CE_ON_DATA:
                    # CE on data
                    H = torch.zeros_like(rx_ce[:, pilot_chunk:, :, re])
                    H_pilot = torch.zeros_like(rx_ce[:, :pilot_chunk, :, re])
                    for user in range(n_users):
                        s_orig_data_pilot = s_orig[:pilot_chunk, user, re]
                        rx_data_ce_cur_pilot = rx_ce[user, :pilot_chunk, :, re]
                        H_pilot[user, :, :] = (s_orig_data_pilot[:, None].conj() / (
                                torch.abs(s_orig_data_pilot[:, None]) ** 2) * rx_data_ce_cur_pilot)
                        s_orig_data = s_orig[pilot_chunk:, user, re]
                        rx_data_ce_cur = rx_ce[user, pilot_chunk:, :, re]
                        H[user, :, :] = (
                                s_orig_data[:, None].conj() / (torch.abs(s_orig_data[:, None]) ** 2) * rx_data_ce_cur)

                    H = H.cpu().numpy()

                    equalized = torch.zeros(rx_data_c.shape[0], tx_data.shape[1], dtype=torch.cfloat)
                    for i in range(rx_data_c.shape[0]):
                        H_cur = H[:, i, :].T
                        H_Ht = H_cur @ H_cur.T.conj()
                        H_Ht_inv = np.linalg.pinv(H_Ht)
                        H_pi = torch.tensor(H_cur.T.conj() @ H_Ht_inv)
                        equalized[i, :] = torch.matmul(H_pi, rx_data_c[i, :, re])
                    detected_word_legacy_ce_on_data = torch.zeros(int(equalized.shape[0] * np.log2(mod_pilot)),
                                                                  equalized.shape[1])
                    if mod_pilot > 2:
                        qam = mod.QAMModem(mod_pilot)
                        for i in range(equalized.shape[1]):
                            detected_word_legacy_ce_on_data[:, i] = torch.from_numpy(
                                qam.demodulate(equalized[:, i].numpy(), 'hard'))
                    else:
                        for i in range(equalized.shape[1]):
                            detected_word_legacy_ce_on_data[:, i] = torch.from_numpy(
                                BPSKModulator.demodulate(-torch.sign(equalized[:, i].real).numpy()))

                # plot phases:
                # plt.plot(np.unwrap(np.angle(H[0,:,0])), linestyle='-', color='g', label='Channel Estimation')
                # plt.plot(np.unwrap(np.angle(cfo_comp_vect)), linestyle='-', color='b', label='cfo_comp_vect')
                # plt.legend()
                # plt.show()

                # GENIE
                equalized = torch.zeros(rx_data_c.shape[0], tx_data.shape[1], dtype=torch.cfloat)
                for i in range(rx_data_c.shape[0]):
                    H_genie = h[:, :, re].cpu().numpy()
                    if (conf.cfo != 0) & (CFO_COMP == 'ON_CE'):
                        H_genie = H_genie * cfo_comp_vect[i]
                    H_Ht = H_genie @ H_genie.T.conj()
                    H_Ht_inv = np.linalg.pinv(H_Ht)
                    H_pi = torch.tensor(H_genie.T.conj() @ H_Ht_inv)
                    equalized[i, :] = torch.matmul(H_pi, rx_data_c[i, :, re])
                detected_word_legacy_genie = torch.zeros(int(equalized.shape[0] * np.log2(mod_pilot)),
                                                         equalized.shape[1])
                if mod_pilot > 2:
                    qam = mod.QAMModem(mod_pilot)
                    for i in range(equalized.shape[1]):
                        detected_word_legacy_genie[:, i] = torch.from_numpy(
                            qam.demodulate(equalized[:, i].numpy(), 'hard'))
                else:
                    for i in range(equalized.shape[1]):
                        detected_word_legacy_genie[:, i] = torch.from_numpy(
                            BPSKModulator.demodulate(-torch.sign(equalized[:, i].real).numpy()))

                ###############################################################
                # llr_for_mi = torch.stack([equalized.real , equalized.imag], dim=1).flatten()
                # tx_data_for_mi = tx_data[:, :rx.shape[1], re]
                # H_y = entropy_with_bin_width(llr_for_mi.numpy(), 0.005)
                # # H_y_x calculation
                # # x=0
                # zero_indexes = np.where(tx_data_for_mi.cpu() == 0)[0]
                # H_y_x_0 = entropy_with_bin_width(llr_for_mi[zero_indexes].numpy(), 0.005)
                # # x=1
                # one_indexes = np.where(tx_data_for_mi.cpu() == 1)[0]
                # H_y_x_1 = entropy_with_bin_width(llr_for_mi[one_indexes].numpy(), 0.005)
                #
                # H_y_x = 0.5 * H_y_x_0 + 0.5 * H_y_x_1
                #
                # mi = H_y - H_y_x
                # mi = np.maximum(mi, 0)
                # pass
                ###############################################################

                # calculate accuracy
                target = tx_data[:, :rx.shape[1], re]
                for iteration in range(iterations):
                    detected_word_cur_re = detected_word_list[iteration][:, :, re, :]
                    detected_word_cur_re = detected_word_cur_re.squeeze(-1)
                    detected_word_cur_re = detected_word_cur_re.reshape(int(tx_data.shape[0] / num_bits), n_users,
                                                                        num_bits).swapaxes(1, 2).reshape(
                        tx_data.shape[0], n_users)

                    if conf.ber_on_one_user >= 0:
                        ber = calculate_ber(detected_word_cur_re[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                            target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                    else:
                        ber = calculate_ber(detected_word_cur_re.cpu(), target.cpu(), num_bits)

                    ber_sum[iteration] += ber
                    ber_per_re[iteration, re] = ber
                if conf.run_e2e:
                    for iteration in range(iters_e2e_disp):
                        detected_word_cur_re_e2e = detected_word_e2e_list[iteration][:, :, re]
                        detected_word_cur_re_e2e = detected_word_cur_re_e2e.squeeze(-1)
                        detected_word_cur_re_e2e = detected_word_cur_re_e2e.reshape(int(tx_data.shape[0] / num_bits),
                                                                                    n_users,
                                                                                    num_bits).swapaxes(1, 2).reshape(
                            tx_data.shape[0],
                            n_users)
                        if conf.ber_on_one_user >= 0:
                            ber_e2e = calculate_ber(
                                detected_word_cur_re_e2e[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                        else:
                            ber_e2e = calculate_ber(detected_word_cur_re_e2e.cpu(), target.cpu(), num_bits)
                        ber_sum_e2e[iteration] += ber_e2e

                if conf.run_deeprx:
                    detected_word_cur_re_deeprx = detected_word_deeprx[:, :, re, :]
                    detected_word_cur_re_deeprx = detected_word_cur_re_deeprx.squeeze(-1)
                    detected_word_cur_re_deeprx = detected_word_cur_re_deeprx.reshape(int(tx_data.shape[0] / num_bits),
                                                                                      n_users,
                                                                                      num_bits).swapaxes(1, 2).reshape(
                        tx_data.shape[0],
                        n_users)
                    if conf.ber_on_one_user >= 0:
                        ber_deeprx = calculate_ber(
                            detected_word_cur_re_deeprx[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                            target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                    else:
                        ber_deeprx = calculate_ber(detected_word_cur_re_deeprx.cpu(), target.cpu(), num_bits)

                if conf.run_deepsicsb and deepsicsb_trainer is not None:
                    detected_word_cur_re_deepsicsb = detected_word_deepsicsb[:, :, re]
                    detected_word_cur_re_deepsicsb = detected_word_cur_re_deepsicsb.reshape(
                        int(tx_data.shape[0] / num_bits), n_users, num_bits).swapaxes(1, 2).reshape(tx_data.shape[0], n_users)

                if conf.ber_on_one_user >= 0:
                    ber_legacy = calculate_ber(
                        torch.from_numpy(detected_word_legacy[:, conf.ber_on_one_user]).unsqueeze(-1),
                        target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                    ber_sphere = calculate_ber(
                        torch.from_numpy(detected_word_sphere[:, conf.ber_on_one_user]).unsqueeze(-1),
                        target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                    ber_deepsicsb = calculate_ber(detected_word_cur_re_deepsicsb[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                        target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                else:
                    ber_legacy = calculate_ber(torch.from_numpy(detected_word_legacy), target.cpu(), num_bits)
                    ber_sphere = calculate_ber(torch.from_numpy(detected_word_sphere), target.cpu(), num_bits)
                    ber_deepsicsb = calculate_ber(detected_word_cur_re_deepsicsb.cpu(), target.cpu(), num_bits)

                ber_per_re_legacy[re] = ber_legacy

                if PLOT_CE_ON_DATA:
                    if conf.ber_on_one_user >= 0:
                        ber_legacy_ce_on_data = calculate_ber(
                            detected_word_legacy_ce_on_data[:conf.ber_on_one_user].unsqueeze(-1).cpu(),
                            target[:conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                    else:
                        ber_legacy_ce_on_data = calculate_ber(detected_word_legacy_ce_on_data.cpu(), target.cpu(),
                                                              num_bits)

                if conf.ber_on_one_user >= 0:
                    ber_legacy_genie = calculate_ber(
                        detected_word_legacy_genie[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                        target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                else:
                    ber_legacy_genie = calculate_ber(detected_word_legacy_genie.cpu(), target.cpu(), num_bits)

                if conf.run_deeprx:
                    ber_sum_deeprx += ber_deeprx
                ber_sum_legacy += ber_legacy
                ber_sum_sphere += ber_sphere
                if PLOT_CE_ON_DATA:
                    ber_sum_legacy_ce_on_data += ber_legacy_ce_on_data
                ber_sum_legacy_genie += ber_legacy_genie

                if conf.run_deepsicsb and deepsicsb_trainer is not None:
                    ber_sum_deepsicsb += ber_deepsicsb

            if PLOT_MI:
                for iteration in range(iterations):
                    mi = calc_mi(tx_data.cpu(), llrs_mat_list[iteration].cpu(), num_bits, n_users, num_res)
                    total_mi_list[iteration].append(mi)
                mi_deeprx = calc_mi(tx_data.cpu(), llrs_mat_deeprx.cpu(), num_bits, n_users, num_res)
                total_mi_deeprx.append(mi_deeprx)
                for iteration in range(iters_e2e_disp):
                    mi_e2e = calc_mi(tx_data.cpu(), llrs_mat_e2e_list[iteration].cpu(), num_bits, n_users, num_res)
                    total_mi_e2e_list[iteration].append(mi_e2e)
                mi_legacy = calc_mi(tx_data.cpu(), llrs_mat_legacy, num_bits, n_users, num_res)
                total_mi_legacy.append(mi_legacy)
            else:
                mi = 0
                mi_e2e = 0
                mi_deeprx = 0
                mi_legacy = 0

            ber_list = [None] * iterations
            for iteration in range(iterations):
                ber_list[iteration] = ber_sum[iteration] / num_res
                total_ber_list[iteration].append(ber_list[iteration])

            if conf.run_e2e:
                ber_e2e_list = [None] * iters_e2e_disp
                for iteration in range(iters_e2e_disp):
                    ber_e2e_list[iteration] = ber_sum_e2e[iteration] / num_res
                    total_ber_e2e_list[iteration].append(ber_e2e_list[iteration])

            ber_deeprx = ber_sum_deeprx / num_res
            ber_legacy = ber_sum_legacy / num_res
            ber_sphere = ber_sum_sphere / num_res
            if PLOT_CE_ON_DATA:
                ber_legacy_ce_on_data = ber_sum_legacy_ce_on_data / num_res
            ber_legacy_genie = ber_sum_legacy_genie / num_res
            ber_deepsicsb = ber_sum_deepsicsb / num_res

            total_ber_deeprx.append(ber_deeprx)
            total_ber_legacy.append(ber_legacy)
            total_ber_sphere.append(ber_sphere)
            if PLOT_CE_ON_DATA:
                total_ber_legacy_ce_on_data.append(ber_legacy_ce_on_data)
            total_ber_legacy_genie.append(ber_legacy_genie)
            total_ber_deepsicsb.append(ber_deepsicsb)
            print(f'SNR={snr_cur}dB, Final SNR={Final_SNR}dB')
            print(f'current DeepSIC: {block_ind, float(ber_list[iterations - 1]), mi}')
            if conf.run_e2e:
                print(f'curr DeepSICe2e: {block_ind, float(ber_e2e_list[iters_e2e_disp - 1]), mi_e2e}')
            if conf.run_deeprx:
                print(f'current DeepRx: {block_ind, ber_deeprx.item(), mi_deeprx}')
            if conf.run_deepsicsb:
                print(f'current DeepSICSB: {block_ind, ber_deepsicsb.item()}')
            if mod_pilot == 4:
                print(f'current legacy: {block_ind, ber_legacy.item(), mi_legacy}')
            else:
                print(f'current legacy: {block_ind, ber_legacy}')
            print(f'current sphere: {block_ind, ber_sphere.item()}')
            if PLOT_CE_ON_DATA:
                print(f'current legacy ce on data: {block_ind, ber_legacy_ce_on_data}')
            if conf.TDL_model[0] == 'N':
                print(f'current legacy genie: {block_ind, ber_legacy_genie.item()}')
        if conf.cfo != 0:
            if conf.cfo_and_clip_in_rx:
                cfo_str = 'cfo in Rx=' + str(conf.cfo) + ' scs'
            else:
                cfo_str = 'cfo in Tx=' + str(conf.cfo) + ' scs'
        else:
            cfo_str = 'cfo=0'

        plot_loss_and_LLRs([0] * len(train_loss_vect), [0] * len(val_loss_vect), torch.from_numpy(llrs_mat_legacy),
                           snr_cur, "Legacy", 0, train_samples, val_samples, mod_text, cfo_str, ber_legacy, ber_legacy,
                           ber_legacy_genie, 0)
        if conf.run_deeprx:
            plot_loss_and_LLRs(train_loss_vect_deeprx, val_loss_vect_deeprx, llrs_mat_deeprx, snr_cur, "DeepRx", 3,
                               train_samples, val_samples, mod_text, cfo_str, ber_deeprx, ber_legacy, ber_legacy_genie,
                               0)
        if conf.run_e2e:
            for iteration in range(iters_e2e_disp):
                plot_loss_and_LLRs(train_loss_vect_e2e, val_loss_vect_e2e, llrs_mat_e2e_list[iteration], snr_cur,
                                   "DeepSICe2e", conf.kernel_size, train_samples, val_samples, mod_text, cfo_str,
                                   ber_e2e_list[iteration], ber_legacy, ber_legacy_genie, iteration)
        for iteration in range(iterations):
            plot_loss_and_LLRs(train_loss_vect, val_loss_vect, llrs_mat_list[iteration], snr_cur, "DeepSIC",
                               conf.kernel_size, train_samples, val_samples, mod_text, cfo_str, ber_list[iteration],
                               ber_legacy, ber_legacy_genie, iteration)

        df = pd.DataFrame(
            {"SNR_range": SNR_range[:len(total_ber_legacy)], "total_ber_1": total_ber_list[0],
             "total_ber_deeprx": total_ber_deeprx,
             "total_ber_sphere": total_ber_sphere,
             "total_ber_deepsicsb": total_ber_deepsicsb,
             "total_ber_legacy": total_ber_legacy, "total_ber_legacy_genie": total_ber_legacy_genie}, )
        if conf.iterations == 2:
            df = pd.DataFrame(
                {"SNR_range": SNR_range[:len(total_ber_legacy)], "total_ber_1": total_ber_list[0],
                 "total_ber_2": total_ber_list[1],
                 "total_ber_deeprx": total_ber_deeprx,
                 "total_ber_sphere": total_ber_sphere,
                 "total_ber_deepsicsb": total_ber_deepsicsb,
                 "total_ber_legacy": total_ber_legacy, "total_ber_legacy_genie": total_ber_legacy_genie}, )
        elif conf.iterations == 3:
            df = pd.DataFrame(
                {"SNR_range": SNR_range[:len(total_ber_legacy)], "total_ber_1": total_ber_list[0],
                 "total_ber_2": total_ber_list[1], "total_ber_3": total_ber_list[2],
                 "total_ber_deeprx": total_ber_deeprx,
                 "total_ber_sphere": total_ber_sphere,
                 "total_ber_deepsicsb": total_ber_deepsicsb,
                 "total_ber_legacy": total_ber_legacy, "total_ber_legacy_genie": total_ber_legacy_genie}, )
        # print('\n'+title_string)
        title_string = (chan_text  + ', ' + mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ", #REs=" + str(
            conf.num_res) + ', Interf=' + str(conf.interf_factor) + ', #UEs=' + str(n_users) + '\n ' +
                        cfo_str + ', Epochs=' + str(epochs) + ', #iterations=' + str(
                    iterations) + ', CNN kernel size=' + str(conf.kernel_size) + ', Clip=' + str(
                    conf.clip_percentage_in_tx) + '%')

        # plot BER per RE for first iteration:
        if conf.ber_on_one_user >= 0:
            add_text = 'user ' + str(conf.ber_on_one_user)
        else:
            add_text = 'all users'

        colors = ['g', 'r', 'k', 'b']
        for iter in range(iterations):
            plt.plot(np.arange(conf.num_res), ber_per_re[iter, :], linestyle='-', color=colors[iter], label='BER ' + add_text+ ', iter'+str(iter))
        plt.legend()
        plt.title('DeepSIC, ' + title_string)
        plt.tight_layout()
        plt.grid()
        plt.show()

        plt.plot(np.arange(conf.num_res), ber_per_re_legacy, linestyle='-', color='g', label='BER ' + add_text)
        plt.legend()
        plt.title('Legacy, ' + title_string)
        plt.tight_layout()
        plt.grid()
        plt.show()

        title_string = title_string.replace("\n", "")
        title_string = title_string.replace(",", "")
        title_string = title_string.replace(" ", "_")
        title_string = title_string + '_two_stage=' + str(conf.two_stage_train)
        title_string = title_string + '_seed=' + str(conf.channel_seed)
        title_string = title_string + '_three_layers=' + str(conf.channel_seed)
        title_string = title_string + '_SNR=' + str(conf.snr)
        title_string = formatted_date + title_string
        output_dir = os.path.join(os.getcwd(), '..', 'Scratchpad')
        file_path = os.path.abspath(os.path.join(output_dir, title_string + ".csv"))
        df.to_csv(file_path, index=False)

        # for iteration in range(iterations):
        #     np.save('C:\\Projects\\Misc\\total_ber_deepseek'+str(iteration)+'.npy', np.array(total_ber_list[iteration]))
        # if conf.run_deeprx:
        #     np.save('C:\\Projects\\Misc\\total_ber_deeprx.npy', np.array(total_ber_deeprx))
        # np.save('C:\\Projects\\Misc\\total_ber_legacy.npy', np.array(total_ber_legacy))
        # np.save('C:\\Projects\\Misc\\total_ber_legacy_genie.npy', np.array(total_ber_legacy_genie))

        # np.save('C:\\Projects\\Misc\\tx_data_-10dB_QPSK.npy', tx_data.cpu())
        # np.save('C:\\Projects\\Misc\\llrs_mat_-10dB_QPSK.npy', llrs_mat.cpu())

    markers = ['o', '*', 'x', 'D', '+', 'o']
    dashes = [':', '-.', '--', '-', '-', '-']
    if PLOT_MI:
        for iteration in range(iterations):
            plt.semilogy(SNR_range, total_mi_list[iteration], linestyle=dashes[iteration], marker=markers[iteration],
                         color='g', label='DeeSIC' + str(iteration))
        if conf.run_deeprx:
            plt.semilogy(SNR_range, total_mi_deeprx, '-o', color='c', label='DeepRx')
        if mod_pilot == 4:
            plt.semilogy(SNR_range, total_mi_legacy, '-o', color='r', label='Legacy')
        plt.xlabel('SNR (dB)')
        plt.ylabel('MI')
        title_string = (chan_text  + ', ' + mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ", #REs=" + str(
            conf.num_res) + ', Interf=' + str(conf.interf_factor) + ', #UEs=' + str(n_users) + '\n ' +
                        cfo_str + ', Epochs=' + str(epochs) + ', #iters_e2e=' + str(
                    conf.iters_e2e) + ', CNN kernel size=' + str(conf.kernel_size))
        plt.title(title_string, fontsize=10)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    snr_at_target_list = np.zeros(iterations)
    snr_at_target_e2e_list = np.zeros(iters_e2e)
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
        if conf.run_deeprx:
            interp_func = interp1d(total_ber_deeprx, SNR_range, kind='linear', fill_value="extrapolate")
            snr_at_target_deeprx = np.round(interp_func(bler_target), 1)
        if conf.run_deepsicsb and deepsicsb_trainer is not None:
            interp_func = interp1d(total_ber_deepsicsb, SNR_range, kind='linear', fill_value="extrapolate")
            snr_at_target_deepsicsb = np.round(interp_func(bler_target), 1)
        interp_func = interp1d(total_ber_legacy, SNR_range, kind='linear', fill_value="extrapolate")
        snr_at_target_legacy = np.round(interp_func(bler_target), 1)
        interp_func = interp1d(total_ber_sphere, SNR_range, kind='linear', fill_value="extrapolate")
        snr_at_target_sphere = np.round(interp_func(bler_target), 1)
        interp_func = interp1d(total_ber_legacy_genie, SNR_range, kind='linear', fill_value="extrapolate")
        snr_at_target_legacy_genie = np.round(interp_func(bler_target), 1)
        if PLOT_CE_ON_DATA:
            interp_func = interp1d(total_ber_legacy_ce_on_data, SNR_range, kind='linear', fill_value="extrapolate")
            snr_at_target_legacy_ce_on_data = np.round(interp_func(bler_target), 1)
    else:
        for iteration in range(iterations):
            snr_at_target_list[iteration] = float('inf')
        if conf.run_e2e:
            for iteration in range(iters_e2e_disp):
                snr_at_target_e2e_list[iteration] = float('inf')
        if conf.run_deeprx:
            snr_at_target_deeprx = float('inf')
        if conf.run_deepsicsb and deepsicsb_trainer is not None:
            snr_at_target_deepsicsb = float('inf')
        snr_at_target_legacy = float('inf')
        snr_at_target_sphere = float('inf')
        snr_at_target_legacy_genie = float('inf')
        snr_at_target_legacy_ce_on_data = float('inf')

    # OryEger - plot only the last iterations
    if not (conf.plot_only_last_iteration):
        iterations_for_plot = list(range(iterations))
    else:
        iterations_for_plot = [iterations - 1]

    for iteration in iterations_for_plot:
        plt.semilogy(SNR_range, total_ber_list[iteration], linestyle=dashes[iteration], marker=markers[iteration],
                     color='g',
                     label='DeepSIC' + str(iteration + 1) + ', SNR @1%=' + str(snr_at_target_list[iteration]))
    if conf.run_e2e:
        if conf.no_probs_e2e:
            e2e_text = 'NoProbs'
        else:
            e2e_text = 'Deepe2e'

        for iteration in range(iters_e2e_disp):
            plt.semilogy(SNR_range, total_ber_e2e_list[iteration], linestyle=dashes[iteration],
                         marker=markers[iteration], color='m',
                         label=e2e_text + str(iteration + 1) + ', SNR @1%=' + str(snr_at_target_e2e_list[iteration]))
    if conf.run_deeprx:
        plt.semilogy(SNR_range, total_ber_deeprx, '-o', color='c',
                     label='DeepRx,   SNR @1%=' + str(snr_at_target_deeprx))
    if conf.run_deepsicsb and deepsicsb_trainer is not None:
        plt.semilogy(SNR_range, total_ber_deepsicsb, '-o', color='orange',
                     label='DeepSICSB, SNR @1%=' + str(snr_at_target_deepsicsb))

    plt.semilogy(SNR_range, total_ber_legacy, '-o', color='r', label='Legacy,    SNR @1%=' + str(snr_at_target_legacy))
    plt.semilogy(SNR_range, total_ber_sphere, '-o', color='brown',
                 label=modulator_text + ',        SNR @1%=' + str(snr_at_target_sphere))

    if PLOT_CE_ON_DATA:
        plt.semilogy(SNR_range, total_ber_legacy_ce_on_data, '-o', color='b',
                     label='CE Data,   SNR @1%=' + str(snr_at_target_legacy_ce_on_data))

    if conf.plot_genie:
        plt.semilogy(SNR_range, total_ber_legacy_genie, '-o', color='k',
                     label='Legacy Genie, SNR @1%=' + str(snr_at_target_legacy_genie))
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    title_string = (chan_text  + ', ' + mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ", #REs=" + str(
        conf.num_res) + ', Interf=' + str(conf.interf_factor) + ', #UEs=' + str(n_users) + '\n ' +
                    cfo_str + ', Epochs=' + str(epochs) + ', #iterations=' + str(
                iterations) + ', CNN kernel size=' + str(conf.kernel_size) + ', Clip=' + str(
                conf.clip_percentage_in_tx) + '%')
    plt.title(title_string, fontsize=10)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Look at the weights:
    # print(deepsic_trainer.detector[0][0].shared_backbone.fc.weight)
    # print(deepsic_trainer.detector[1][0].instance_heads[0].fc1.weight[0])

    return total_ber_list


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    args = parser.parse_args()

    # Reload config singleton with the provided config file (or default)
    conf.reload_config(args.config)

    # Now conf has the updated config, proceed as before
    assert not (conf.separate_nns and conf.mod_pilot <= 4), "Assert: Can't use separate nns with QPSK"
    assert not (conf.no_probs_e2e and conf.iters_e2e != 1 and conf.full_e2e == True), "Assert: No probs only works with 1 iteration or with full e2e"
    assert not (conf.separate_nns and conf.mod_pilot <= 4), "Assert: Can't use separate nns with QPSK"
    assert not (
                conf.no_probs_e2e and conf.iters_e2e != 1 and conf.full_e2e == True), "Assert: No probs only works with 1 iteration or with full e2e"

    start_time = time.time()
    num_bits = int(np.log2(conf.mod_pilot))
    deepsic_trainer = DeepSICTrainer(num_bits, conf.n_users)
    deepsice2e_trainer = DeepSICe2eTrainer(num_bits, conf.n_users)
    deeprx_trainer = DeepRxTrainer(conf.num_res, conf.n_users)
    deepsicsb_trainer = DeepSICSBTrainer(conf.num_res, conf.n_users)
    print(deepsic_trainer)
    run_evaluate(deepsic_trainer, deepsice2e_trainer, deeprx_trainer, deepsicsb_trainer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    days = int(elapsed_time // (24 * 3600))
    elapsed_time %= 24 * 3600
    hours = int(elapsed_time // 3600)
    elapsed_time %= 3600
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")
