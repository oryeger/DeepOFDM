

import os
from pathlib import Path
import time
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.detectors.deepsice2e.deepsice2e_trainer import DeepSICe2eTrainer
from python_code.detectors.deeprx.deeprx_trainer import DeepRxTrainer
from python_code.detectors.deepsicsb.deepsicsb_trainer import DeepSICSBTrainer
from python_code.detectors.deepsicmb.deepsicmb_trainer import DeepSICMBTrainer
from python_code.detectors.deepstag.deepstag_trainer import DeepSTAGTrainer


from typing import List

import numpy as np
import torch
from python_code import conf
from python_code.utils.metrics import calculate_ber
import matplotlib.pyplot as plt
from python_code.utils.constants import (IS_COMPLEX, TRAIN_PERCENTAGE, CFO_COMP, GENIE_CFO,
                                         FFT_size, FIRST_CP, CP, NUM_SYMB_PER_SLOT, NUM_SAMPLES_PER_SLOT, PLOT_MI,
                                         PLOT_CE_ON_DATA)

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

from python_code.coding.mcs_table import get_mcs


from python_code.coding.ldpc_wrapper import LDPC5GCodec
from python_code.coding.crc_wrapper import CRC5GCodec




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
    if conf.enable_two_stage_train:
        p_len = p_len * 2
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
                    cfo_str + ', Epochs=' + str(conf.epochs) + iters_txt + ', CNN kernel size=' + str(kernel_size) + ', two_stage=' + str(conf.enable_two_stage_train))

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
    return fig


def run_evaluate(deepsic_trainer, deepsice2e_trainer, deeprx_trainer, deepsicsb_trainer=None, deepsicmb_trainer=None, deepstag_trainer=None) -> List[float]:
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
    n_ants = conf.n_ants
    iterations = conf.iterations
    iters_e2e = conf.iters_e2e
    epochs = conf.epochs
    half_kernel = int(np.ceil(conf.kernel_size/2))

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
    total_bler_list = [[] for _ in range(iterations)]
    total_ber_e2e_list = [[] for _ in range(iters_e2e_disp)]
    total_ber_deepsicsb_list = [[] for _ in range(iterations)]
    total_ber_deepsicmb_list = [[] for _ in range(iterations)]
    total_ber_deepstag_list = [[] for _ in range(iterations*2)]
    total_ber_deeprx = []
    total_ber_legacy = []
    total_bler_legacy = []
    total_ber_sphere = []
    total_bler_sphere = []

    if conf.mcs > -1:
        qm, code_rate = get_mcs(conf.mcs)
        assert (np.log2(mod_pilot) == qm), "Assert: MCS and modulation don't fit"
        ldpc_n = int(conf.num_res * NUM_SYMB_PER_SLOT * qm)
        ldpc_k = int(ldpc_n*code_rate)
    else:
        ldpc_n = 0
        ldpc_k = 0



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

    if mod_pilot == 2:
        constellation_factor = 1
    elif mod_pilot == 4:
        constellation_factor = 2
    elif mod_pilot == 16:
        constellation_factor = 10
    elif mod_pilot == 64:
        constellation_factor = 42
    elif mod_pilot == 256:
        constellation_factor = 170

    for snr_cur in SNR_range:
        ber_sum = np.zeros(iterations)
        ber_per_re = np.zeros((iterations, conf.num_res))
        ber_per_re_deepsicsb = np.zeros((iterations, conf.num_res))
        ber_per_re_deepsicmb = np.zeros((iterations, conf.num_res))
        ber_per_re_deepstag = np.zeros((iterations*2, conf.num_res))
        ber_sum_e2e = np.zeros(iters_e2e_disp)
        ber_sum_deeprx = 0
        ber_sum_legacy = 0
        ber_per_re_legacy = np.zeros(conf.num_res)
        ber_sum_sphere = 0
        ber_sum_deepsicsb = np.zeros(iterations)
        ber_sum_deepsicmb = np.zeros(iterations)
        ber_sum_deepstag = np.zeros(iterations*2)
        if PLOT_CE_ON_DATA:
            ber_sum_legacy_ce_on_data = 0
        ber_sum_legacy_genie = 0
        deepsic_trainer._initialize_detector(num_bits, n_users, n_ants)  # For reseting the weights
        deepsice2e_trainer._initialize_detector(num_bits, n_users, n_ants)
        deeprx_trainer._initialize_detector(num_bits, n_users, n_ants)
        if conf.run_deepsicsb and deepsicsb_trainer is not None:
            deepsicsb_trainer._initialize_detector(num_bits, n_users, n_ants)

        if conf.run_deepsicmb and deepsicmb_trainer is not None:
            deepsicmb_trainer._initialize_detector(num_bits, n_users, n_ants)

        if conf.run_deepstag and deepstag_trainer is not None:
            deepstag_trainer._initialize_detector(num_bits, n_users, n_ants)

        pilot_size = get_next_divisible(conf.pilot_size, num_bits * NUM_SYMB_PER_SLOT)
        pilot_chunk = int(pilot_size / np.log2(mod_pilot))

        noise_var = 10 ** (-0.1 * snr_cur) * constellation_factor

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
            noise_var_list=[noise_var], num_bits=num_bits, n_users
            =n_users, mod_pilot=mod_pilot, ldpc_k=ldpc_k, ldpc_n=ldpc_n)

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

            rx_c = rx.cpu()
            llrs_mat_legacy_for_aug = np.zeros((rx_c.shape[0], num_bits * n_users, num_res, 1))
            llrs_mat_sphere_for_aug = np.zeros((rx_c.shape[0], num_bits * n_users, num_res, 1))
            detected_word_legacy_for_aug = np.zeros((int(rx_c.shape[0] * np.log2(mod_pilot)), n_users,num_res))
            detected_word_sphere_for_aug = np.zeros((int(rx_c.shape[0] * np.log2(mod_pilot)), n_users,num_res))
            # for re in range(conf.num_res):
            for re in range(conf.num_res):
                # Regular CE
                H = torch.zeros_like(h[:, :, re])
                for user in range(n_users):
                    if not(conf.separate_pilots):
                        rx_pilot_ce_cur = rx_ce[user, :pilot_chunk, :, re]
                        s_orig_pilot = s_orig[:pilot_chunk, user, re]
                        H[:, user] = 1 / s_orig_pilot.shape[0] * (s_orig_pilot[:, None].conj() / (
                                torch.abs(s_orig_pilot[:, None]) ** 2) * rx_pilot_ce_cur).sum(dim=0)
                    else:
                        rx_pilot_ce_cur = rx_ce[user, user:pilot_chunk:n_users, :, re]
                        s_orig_pilot = s_orig[user:pilot_chunk:n_users, user, re]
                        H[:, user] = 1 / s_orig_pilot.shape[0] * (s_orig_pilot[:, None].conj() / (
                                torch.abs(s_orig_pilot[:, None]) ** 2) * rx_pilot_ce_cur).sum(dim=0)

                H = torch.tensor(H, dtype=torch.complex128)
                I_users = torch.eye(n_users, dtype=H.dtype, device=H.device)
                W = torch.linalg.inv(H.T.conj() @ H + noise_var * I_users) @ H.T.conj()
                bias = (W@H).diag().real
                W = W.cpu()
                bias = bias.cpu()
                equalized = torch.zeros(rx_c.shape[0], n_users, dtype=torch.cfloat)
                for i in range(rx_c.shape[0]):
                    equalized[i, :] = torch.matmul(W, rx_c[i, :, re])/bias

                postEqSINR = bias/(1-bias)

                # signal_var = np.sum(np.abs(H[:, user]) ** 2)
                # postEqSINR[0] = signal_var / noise_var
                # postEqSINR[1] = signal_var / noise_var
                # postEqSINR[2] = signal_var / noise_var
                # postEqSINR[3] = signal_var / noise_var

                if mod_pilot == 2:
                    for i in range(equalized.shape[1]):
                        detected_word_legacy_for_aug[:, i,re] = torch.from_numpy(
                            BPSKModulator.demodulate(-torch.sign(equalized[:, i].real).numpy()))
                elif mod_pilot == 4:
                    # qam = mod.QAMModem(mod_pilot)
                    for user in range(n_users):
                        detected_word_legacy_for_aug[:, user,re], llr_out = QPSKModulator.demodulate(equalized[:, user].numpy())
                        llrs_mat_legacy_for_aug[:, (user * num_bits):((user + 1) * num_bits), re, :] = llr_out.reshape(
                            int(llr_out.shape[0] / num_bits), num_bits, 1) * postEqSINR[user].numpy()

                elif mod_pilot == 16:
                    for user in range(n_users):
                        detected_word_legacy_for_aug[:, user,re], llr_out = QAM16Modulator.demodulate(equalized[:, user].numpy())
                        llrs_mat_legacy_for_aug[:, (user * num_bits):((user + 1) * num_bits), re, :] = llr_out.reshape(
                            int(llr_out.shape[0] / num_bits), num_bits, 1) * postEqSINR[user].numpy()
                else:
                    print('Unknown modulator')

                if conf.run_sphere:
                    # start = time.time()
                    llr_out, detected_word_sphere_for_aug[:, :,re]  = SphereDecoder(H, rx_c[:, :, re].numpy(), noise_var, conf.sphere_radius)
                    # end = time.time()
                    # print(f"SphereDecoder took {end - start:.4f} seconds")
                else:
                    llr_out = np.zeros((rx_c.shape[0]*num_bits,n_users))
                    detected_word_sphere_for_aug[:, :, re] = np.zeros((rx_c.shape[0]*num_bits,n_users))

                for user in range(n_users):
                    llrs_mat_sphere_for_aug[:, (user * num_bits):((user + 1) * num_bits), re, :] = -llr_out[:,user].reshape(
                        int(llr_out[:,user].shape[0] / num_bits), num_bits, 1)


            llrs_mat_legacy = llrs_mat_legacy_for_aug[pilot_chunk:, :, :, :]
            llrs_mat_sphere = llrs_mat_sphere_for_aug[pilot_chunk:, :, :, :]
            if conf.sphere_augment and conf.run_sphere:
                probs_for_aug = torch.sigmoid(torch.tensor(llrs_mat_sphere_for_aug, dtype=torch.float32))
            else:
                probs_for_aug = torch.sigmoid(torch.tensor(llrs_mat_legacy_for_aug, dtype=torch.float32))

            # online training main function
            if deepsic_trainer.is_online_training:
                if conf.train_on_ce_no_pilots:
                    H_all = torch.zeros(conf.n_ants, conf.num_res, dtype=torch.complex64)
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
                                                                                      epochs, False, probs_for_aug)
                elif conf.use_data_as_pilots:
                    H_all = torch.zeros(s_orig.shape[0], conf.n_ants * conf.n_users, conf.num_res, dtype=torch.complex64)
                    for re in range(conf.num_res):
                        H = torch.zeros(s_orig.shape[0], conf.n_ants, conf.n_users, dtype=torch.complex64)
                        for user in range(n_users):
                            s_orig_pilot = s_orig[:, user, re]
                            rx_pilot_ce_cur = rx_ce[user, :, :, re]

                            H[:, :, user] = (s_orig_pilot[:, None].conj() / (
                                    torch.abs(s_orig_pilot[:, None]) ** 2)) * rx_pilot_ce_cur  # shape: [56, 4]
                        H_all[:, :, re] = H.reshape(H.shape[0], conf.n_ants * conf.n_users)
                    real_part = H_all.real
                    imag_part = H_all.imag
                    H_all_real = torch.empty((H_all.shape[0], H_all.shape[1] * 2, H_all.shape[2]),
                                             dtype=torch.float32)
                    H_all_real[:, 0::2, :] = real_part  # Real parts in even rows
                    H_all_real[:, 1::2, :] = imag_part  # Imaginary parts in odd rows
                    rx_pilot_and_H = torch.cat((rx_pilot, H_all_real[:pilot_chunk]), dim=1)

                    train_loss_vect, val_loss_vect = deepsic_trainer._online_training(tx_pilot,
                                                                                      rx_pilot_and_H.to('cpu'),
                                                                                      num_bits, n_users, iterations,epochs, False, probs_for_aug[:pilot_chunk])
                else:
                    if not conf.enable_two_stage_train:
                        train_loss_vect, val_loss_vect = deepsic_trainer._online_training(tx_pilot, rx_pilot, num_bits,
                                                                                          n_users, iterations, epochs, False, probs_for_aug[:pilot_chunk])
                    else:
                        tx_pilot_cur = tx_pilot[:int(pilot_chunk*num_bits/2),:,:]
                        rx_pilot_cur = rx_pilot[:int(pilot_chunk / 2), :, :]
                        train_loss_vect_1, val_loss_vect_1 = deepsic_trainer._online_training(tx_pilot_cur, rx_pilot_cur, num_bits,
                                                                                          n_users, iterations, epochs, True, probs_for_aug[:pilot_chunk])

                        tx_pilot_cur = tx_pilot[int(pilot_chunk*num_bits/2):,:,:]
                        rx_pilot_cur = rx_pilot[int(pilot_chunk / 2):, :, :]
                        train_loss_vect_2, val_loss_vect_2 = deepsic_trainer._online_training(tx_pilot_cur, rx_pilot_cur, num_bits,
                                                                                          n_users, iterations, epochs, False, probs_for_aug[:pilot_chunk])
                        train_loss_vect = train_loss_vect_1 + train_loss_vect_2
                        val_loss_vect = val_loss_vect_1 + val_loss_vect_2


                if conf.train_on_ce_no_pilots:
                    H_repeated = H_all_real.unsqueeze(0).repeat(rx_data.shape[0], 1, 1)
                    rx_data_and_H = torch.cat((rx_data, H_repeated), dim=1)
                    detected_word_list, llrs_mat_list = deepsic_trainer._forward(rx_data_and_H, num_bits, n_users,
                                                                                 iterations, probs_for_aug[pilot_chunk:])
                elif conf.use_data_as_pilots:
                    rx_data_and_H = torch.cat((rx_data, H_all_real[pilot_chunk:]), dim=1)
                    detected_word_list, llrs_mat_list = deepsic_trainer._forward(rx_data_and_H, num_bits, n_users,
                                                                                 iterations, probs_for_aug[pilot_chunk:])
                else:
                    detected_word_list, llrs_mat_list = deepsic_trainer._forward(rx_data, num_bits, n_users, iterations, probs_for_aug[pilot_chunk:])

            if conf.run_e2e:
                if deepsice2e_trainer.is_online_training:
                    for _ in range(conf.num_trainings):
                        train_loss_vect_e2e, val_loss_vect_e2e = deepsice2e_trainer._online_training(tx_pilot, rx_pilot,
                                                                                                     num_bits, n_users,
                                                                                                     iters_e2e, epochs, False, torch.empty(0))
                    detected_word_e2e_list, llrs_mat_e2e_list = deepsice2e_trainer._forward(rx_data, num_bits, n_users,
                                                                                            iters_e2e, torch.empty(0))

            if conf.run_deeprx:
                if deeprx_trainer.is_online_training:
                    for _ in range(conf.num_trainings):
                        train_loss_vect_deeprx, val_loss_vect_deeprx = deeprx_trainer._online_training(tx_pilot, rx_pilot,
                                                                                                       num_bits, n_users,
                                                                                                       iterations, epochs, False, torch.empty(0))
                    detected_word_deeprx, llrs_mat_deeprx = deeprx_trainer._forward(rx_data, num_bits, n_users,
                                                                                    iterations, torch.empty(0))
            if conf.run_deepsicsb and deepsicsb_trainer is not None:
                if deepsicsb_trainer.is_online_training:
                    for _ in range(conf.num_trainings):
                        train_loss_vect_deepsicsb, val_loss_vect_deepsicsb = deepsicsb_trainer._online_training(
                            tx_pilot, rx_pilot, num_bits, n_users, iterations, epochs, False, torch.empty(0))
                    detected_word_deepsicsb_list, llrs_mat_deepsicsb_list = deepsicsb_trainer._forward(rx_data, num_bits, n_users,
                                                                                         iterations, torch.empty(0))
            if conf.run_deepsicmb and deepsicmb_trainer is not None:
                if deepsicmb_trainer.is_online_training:
                    for _ in range(conf.num_trainings):
                        train_loss_vect_deepsicmb, val_loss_vect_deepsicmb = deepsicmb_trainer._online_training(
                            tx_pilot, rx_pilot, num_bits, n_users, iterations, epochs, False, torch.empty(0))
                    detected_word_deepsicmb_list, llrs_mat_deepsicmb_list = deepsicmb_trainer._forward(rx_data, num_bits, n_users,
                                                                                         iterations, torch.empty(0))
            if conf.run_deepstag and deepstag_trainer is not None:
                if deepstag_trainer.is_online_training:
                    for _ in range(conf.num_trainings):
                        train_loss_vect_deepstag, val_loss_vect_deepstag = deepstag_trainer._online_training(
                            tx_pilot, rx_pilot, num_bits, n_users, iterations, epochs, False, torch.empty(0))
                    detected_word_deepstag_list, llrs_mat_deepstag_list = deepstag_trainer._forward(rx_data, num_bits, n_users,
                                                                                         iterations, torch.empty(0))
            # CE Based
            rx_data_c = rx[pilot_chunk:].cpu()

            for re in range(conf.num_res):
                # Regular CE
                detected_word_legacy = detected_word_legacy_for_aug[pilot_size:, :, re]
                detected_word_sphere = detected_word_sphere_for_aug[pilot_size:, :, re]

                # Sphere:
                if conf.sphere_radius == 'inf':
                    radius = np.inf
                    modulator_text = 'MAP'
                else:
                    radius = float(conf.sphere_radius)
                    modulator_text = 'Sphere, Radius=' + str(conf.sphere_radius)

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
                        W = torch.tensor(H_cur.T.conj() @ H_Ht_inv)
                        equalized[i, :] = torch.matmul(W, rx_data_c[i, :, re])
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
                    W = torch.tensor(H_genie.T.conj() @ H_Ht_inv)
                    equalized[i, :] = torch.matmul(W, rx_data_c[i, :, re])
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

                    # if (re>=half_kernel) & (re<=conf.num_res-half_kernel-1):
                    #     ber_sum[iteration] += ber
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
                    for iteration in range(iterations):
                        detected_word_cur_re_deepsicsb = detected_word_deepsicsb_list[iteration][:, :, re]
                        detected_word_cur_re_deepsicsb = detected_word_cur_re_deepsicsb.squeeze(-1)
                        detected_word_cur_re_deepsicsb = detected_word_cur_re_deepsicsb.reshape(int(tx_data.shape[0] / num_bits),
                                                                                    n_users,
                                                                                    num_bits).swapaxes(1, 2).reshape(
                            tx_data.shape[0],
                            n_users)
                        if conf.ber_on_one_user >= 0:
                            ber_deepsicsb = calculate_ber(
                                detected_word_cur_re_deepsicsb[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                        else:
                            ber_deepsicsb = calculate_ber(detected_word_cur_re_deepsicsb.cpu(), target.cpu(), num_bits)
                        # if (re >= half_kernel) & (re <= conf.num_res - half_kernel - 1):
                        #     ber_sum_deepsicsb[iteration] += ber_deepsicsb
                        ber_sum_deepsicsb[iteration] += ber_deepsicsb
                        ber_per_re_deepsicsb[iteration, re] = ber_deepsicsb


                if conf.run_deepsicmb and deepsicmb_trainer is not None:
                    for iteration in range(iterations):
                        detected_word_cur_re_deepsicmb = detected_word_deepsicmb_list[iteration][:, :, re]
                        detected_word_cur_re_deepsicmb = detected_word_cur_re_deepsicmb.squeeze(-1)
                        detected_word_cur_re_deepsicmb = detected_word_cur_re_deepsicmb.reshape(int(tx_data.shape[0] / num_bits),
                                                                                    n_users,
                                                                                    num_bits).swapaxes(1, 2).reshape(
                            tx_data.shape[0],
                            n_users)
                        if conf.ber_on_one_user >= 0:
                            ber_deepsicmb = calculate_ber(
                                detected_word_cur_re_deepsicmb[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                        else:
                            ber_deepsicmb = calculate_ber(detected_word_cur_re_deepsicmb.cpu(), target.cpu(), num_bits)
                        # if (re >= half_kernel) & (re <= conf.num_res - half_kernel - 1):
                        #     ber_sum_deepsicmb[iteration] += ber_deepsicmb
                        ber_sum_deepsicmb[iteration] += ber_deepsicmb
                        ber_per_re_deepsicmb[iteration, re] = ber_deepsicmb


                if conf.run_deepstag and deepstag_trainer is not None:
                    for iteration in range(iterations*2):
                        detected_word_cur_re_deepstag = detected_word_deepstag_list[iteration][:, :, re]
                        detected_word_cur_re_deepstag = detected_word_cur_re_deepstag.squeeze(-1)
                        detected_word_cur_re_deepstag = detected_word_cur_re_deepstag.reshape(int(tx_data.shape[0] / num_bits),
                                                                                    n_users,
                                                                                    num_bits).swapaxes(1, 2).reshape(
                            tx_data.shape[0],
                            n_users)
                        if conf.ber_on_one_user >= 0:
                            ber_deepstag = calculate_ber(
                                detected_word_cur_re_deepstag[:, conf.ber_on_one_user].unsqueeze(-1).cpu(),
                                target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                        else:
                            ber_deepstag = calculate_ber(detected_word_cur_re_deepstag.cpu(), target.cpu(), num_bits)
                        # if (re >= half_kernel) & (re <= conf.num_res - half_kernel - 1):
                        #     ber_sum_deepstag[iteration] += ber_deepstag
                        ber_sum_deepstag[iteration] += ber_deepstag
                        ber_per_re_deepstag[iteration, re] = ber_deepstag


                if conf.ber_on_one_user >= 0:
                    ber_legacy = calculate_ber(
                        torch.from_numpy(detected_word_legacy[:, conf.ber_on_one_user]).unsqueeze(-1),
                        target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                    ber_sphere = calculate_ber(
                        torch.from_numpy(detected_word_sphere[:, conf.ber_on_one_user]).unsqueeze(-1),
                        target[:, conf.ber_on_one_user].unsqueeze(-1).cpu(), num_bits)
                else:
                    ber_legacy = calculate_ber(torch.from_numpy(detected_word_legacy), target.cpu(), num_bits)
                    ber_sphere = calculate_ber(torch.from_numpy(detected_word_sphere), target.cpu(), num_bits)

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
                    # if (re >= half_kernel) & (re <= conf.num_res - half_kernel - 1):
                    #     ber_sum_deeprx += ber_deeprx
                    ber_sum_deeprx += ber_deeprx
                ber_sum_legacy += ber_legacy
                ber_sum_sphere += ber_sphere
                if PLOT_CE_ON_DATA:
                    ber_sum_legacy_ce_on_data += ber_legacy_ce_on_data
                ber_sum_legacy_genie += ber_legacy_genie

            # LDPC decoding
            bler_list = [None] * iterations
            if conf.mcs>-1:
                if ldpc_k > 3824:
                    crc_length = 24
                else:
                    crc_length = 16
                codec = LDPC5GCodec(k=(ldpc_k+crc_length), n=ldpc_n)
                crc = CRC5GCodec(crc_length)
                for iteration in range(iterations):
                    llr_all_res = np.zeros((n_users,int(tx_data.shape[0]*conf.num_res)))
                    llr_all_res_legacy = np.zeros((n_users,int(tx_data.shape[0]*conf.num_res)))
                    llr_all_res_sphere = np.zeros((n_users,int(tx_data.shape[0]*conf.num_res)))
                    for re in range(conf.num_res):
                        # DeepSIC
                        llr_cur_re = llrs_mat_list[iteration][:, :, re, :]
                        llr_cur_re = llr_cur_re.squeeze(-1)
                        llr_cur_re = llr_cur_re.reshape(int(tx_data.shape[0] / num_bits), n_users,
                                                                        num_bits).swapaxes(1, 2).reshape(
                        tx_data.shape[0], n_users)
                        llr_all_res[:,re::conf.num_res] = llr_cur_re.swapaxes(0, 1).cpu()

                        # Legacy
                        llr_cur_re_legacy = llrs_mat_legacy[:, :, re, :]
                        llr_cur_re_legacy = llr_cur_re_legacy.squeeze(-1)
                        llr_cur_re_legacy = llr_cur_re_legacy.reshape(int(tx_data.shape[0] / num_bits), n_users,
                                                                        num_bits).swapaxes(1, 2).reshape(
                        tx_data.shape[0], n_users)
                        llr_all_res_legacy[:,re::conf.num_res] = llr_cur_re_legacy.swapaxes(0, 1)

                        # Sphere
                        llr_cur_re_sphere = llrs_mat_sphere[:, :, re, :]
                        llr_cur_re_sphere = llr_cur_re_sphere.squeeze(-1)
                        llr_cur_re_sphere = llr_cur_re_sphere.reshape(int(tx_data.shape[0] / num_bits), n_users,
                                                                        num_bits).swapaxes(1, 2).reshape(
                        tx_data.shape[0], n_users)
                        llr_all_res_sphere[:,re::conf.num_res] = llr_cur_re_sphere.swapaxes(0, 1)


                    num_slots = int(np.floor(llr_all_res.shape[1] / ldpc_n))
                    crc_count = 0
                    crc_count_legacy = 0
                    crc_count_sphere = 0
                    for slot in range(num_slots):
                        decodedwords = codec.decode(llr_all_res[:, slot * ldpc_n:(slot + 1) * ldpc_n])
                        crc_out = crc.decode(decodedwords)
                        crc_count += (~crc_out).numpy().astype(int).sum()
                        decodedwords_legacy = codec.decode(llr_all_res_legacy[:, slot * ldpc_n:(slot + 1) * ldpc_n])
                        crc_out_legacy = crc.decode(decodedwords_legacy)
                        crc_count_legacy += (~crc_out_legacy).numpy().astype(int).sum()
                        decodedwords_sphere = codec.decode(llr_all_res_sphere[:, slot * ldpc_n:(slot + 1) * ldpc_n])
                        crc_out_sphere = crc.decode(decodedwords_sphere)
                        crc_count_sphere += (~crc_out_sphere).numpy().astype(int).sum()
                    bler_list[iteration] = crc_count / (num_slots * n_users)
                    total_bler_list[iteration].append(bler_list[iteration])
                    if iteration == 0:
                        bler_legacy = crc_count_legacy / (num_slots * n_users)
                        total_bler_legacy.append(bler_legacy)
                        bler_sphere = crc_count_sphere / (num_slots * n_users)
                        total_bler_sphere.append(bler_sphere)
            else:
                for iteration in range(iterations):
                    bler_list[iteration] = 0
                    total_bler_list[iteration].append(0)
                bler_legacy = 0
                total_bler_legacy.append(0)
                total_bler_sphere.append(0)

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
                # ber_list[iteration] = ber_sum[iteration] / (num_res - 2*conf.kernel_size)
                ber_list[iteration] = ber_sum[iteration] / num_res
                total_ber_list[iteration].append(ber_list[iteration])

            ber_e2e_list = [None] * iters_e2e_disp
            for iteration in range(iters_e2e_disp):
                ber_e2e_list[iteration] = ber_sum_e2e[iteration] / num_res
                total_ber_e2e_list[iteration].append(ber_e2e_list[iteration])

            # ber_deeprx = ber_sum_deeprx / (num_res - 2*conf.kernel_size)
            ber_deeprx = ber_sum_deeprx / num_res
            ber_legacy = ber_sum_legacy / num_res
            ber_sphere = ber_sum_sphere / num_res
            if PLOT_CE_ON_DATA:
                ber_legacy_ce_on_data = ber_sum_legacy_ce_on_data / num_res
            ber_legacy_genie = ber_sum_legacy_genie / num_res

            ber_deepsicsb_list = [None] * iterations
            for iteration in range(iterations):
                # ber_deepsicsb_list[iteration] = ber_sum_deepsicsb[iteration]  / (num_res - 2*conf.kernel_size)
                ber_deepsicsb_list[iteration] = ber_sum_deepsicsb[iteration]  / num_res
                total_ber_deepsicsb_list[iteration].append(ber_deepsicsb_list[iteration])

            ber_deepsicmb_list = [None] * iterations
            for iteration in range(iterations):
                # ber_deepsicmb_list[iteration] = ber_sum_deepsicmb[iteration]  / (num_res - 2*conf.kernel_size)
                ber_deepsicmb_list[iteration] = ber_sum_deepsicmb[iteration]  / num_res
                total_ber_deepsicmb_list[iteration].append(ber_deepsicmb_list[iteration])

            ber_deepstag_list = [None] * iterations*2
            for iteration in range(iterations*2):
                # ber_deepstag_list[iteration] = ber_sum_deepstag[iteration]  / (num_res - 2*conf.kernel_size)
                ber_deepstag_list[iteration] = ber_sum_deepstag[iteration]  / num_res
                total_ber_deepstag_list[iteration].append(ber_deepstag_list[iteration])

            total_ber_deeprx.append(ber_deeprx)
            total_ber_legacy.append(ber_legacy)
            total_ber_sphere.append(ber_sphere)
            if PLOT_CE_ON_DATA:
                total_ber_legacy_ce_on_data.append(ber_legacy_ce_on_data)
            total_ber_legacy_genie.append(ber_legacy_genie)
            print(f'SNR={snr_cur}dB, Final SNR={Final_SNR}dB')
            print(f'current DeepSIC: {block_ind, float(ber_list[iterations - 1]), mi}')
            if conf.mcs>-1:
                print(f'current DeepSIC BLER: {block_ind, float(bler_list[iterations - 1]), mi}')
            if conf.run_e2e:
                print(f'curr DeepSICe2e: {block_ind, float(ber_e2e_list[iters_e2e_disp - 1]), mi_e2e}')
            if conf.run_deeprx:
                print(f'current DeepRx: {block_ind, ber_deeprx.item(), mi_deeprx}')
            if conf.run_deepsicsb and deepsicsb_trainer is not None:
                print(f'current DeepSICSB: {block_ind, float(ber_deepsicsb_list[iterations - 1])}')
            if conf.run_deepsicmb and deepsicmb_trainer is not None:
                print(f'current DeepSICMB: {block_ind, float(ber_deepsicmb_list[iterations - 1])}')
            if conf.run_deepstag and deepstag_trainer is not None:
                print(f'current DeepSTAG: {block_ind, float(ber_deepstag_list[iterations*2 - 1])}')
            if mod_pilot == 4:
                print(f'current legacy: {block_ind, ber_legacy.item(), mi_legacy}')
            else:
                print(f'current legacy: {block_ind, ber_legacy}')

            if conf.mcs>-1:
                print(f'current legacy BLER: {block_ind, float(bler_legacy), mi}')

            if conf.run_sphere:
                print(f'current sphere: {block_ind, ber_sphere.item()}')
                if conf.mcs>-1:
                    print(f'current sphere BLER: {block_ind, float(bler_sphere), mi}')

            if PLOT_CE_ON_DATA:
                print(f'current legacy ce on data: {block_ind, ber_legacy_ce_on_data}')
            if conf.TDL_model[0] == 'N':
                print(f'current legacy genie: {block_ind, ber_legacy_genie.item()}')
        cfo_str = 'cfo=' + str(conf.cfo) + ' scs'

        fig_legacy = plot_loss_and_LLRs([0] * len(train_loss_vect), [0] * len(val_loss_vect), torch.from_numpy(llrs_mat_legacy),
                           snr_cur, "Legacy", 0, train_samples, val_samples, mod_text, cfo_str, ber_legacy, ber_legacy,
                           ber_legacy_genie, 0)

        if conf.run_sphere:
            fig_legacy = plot_loss_and_LLRs([0] * len(train_loss_vect), [0] * len(val_loss_vect), torch.from_numpy(llrs_mat_sphere),
                               snr_cur, "Sphere", 0, train_samples, val_samples, mod_text, cfo_str, ber_sphere, ber_legacy,
                               ber_legacy_genie, 0)


        if conf.run_deeprx:
            fig_deeprx = plot_loss_and_LLRs(train_loss_vect_deeprx, val_loss_vect_deeprx, llrs_mat_deeprx, snr_cur, "DeepRx", 3,
                               train_samples, val_samples, mod_text, cfo_str, ber_deeprx, ber_legacy, ber_legacy_genie,
                               0)
        if conf.run_deepsicsb and deepsicsb_trainer is not None:
            for iteration in range(iterations):
                fig_deepsicsb = plot_loss_and_LLRs(train_loss_vect_deepsicsb, val_loss_vect_deepsicsb, llrs_mat_deepsicsb_list[iteration], snr_cur, "DeepSICSB", 3,
                                   train_samples, val_samples, mod_text, cfo_str, ber_deepsicsb_list[iteration], ber_legacy, ber_legacy_genie,
                                   conf.iterations)


        if conf.run_deepsicmb and deepsicmb_trainer is not None:
            for iteration in range(iterations):
                fig_deepsicmb = plot_loss_and_LLRs(train_loss_vect_deepsicmb, val_loss_vect_deepsicmb, llrs_mat_deepsicmb_list[iteration], snr_cur, "DeepSICMB", 3,
                                   train_samples, val_samples, mod_text, cfo_str, ber_deepsicmb_list[iteration], ber_legacy, ber_legacy_genie,
                                   conf.iterations)
        if conf.run_deepstag and deepstag_trainer is not None:
            for iteration in range(iterations*2):
                fig_deepstag = plot_loss_and_LLRs(train_loss_vect_deepstag, val_loss_vect_deepstag, llrs_mat_deepstag_list[iteration], snr_cur, "DeepSTAG", 3,
                                   train_samples, val_samples, mod_text, cfo_str, ber_deepstag_list[iteration], ber_legacy, ber_legacy_genie,
                                   conf.iterations)
        if conf.run_e2e:
            for iteration in range(iters_e2e_disp):
                fig_e2e = plot_loss_and_LLRs(train_loss_vect_e2e, val_loss_vect_e2e, llrs_mat_e2e_list[iteration], snr_cur,
                                   "DeepSICe2e", conf.kernel_size, train_samples, val_samples, mod_text, cfo_str,
                                   ber_e2e_list[iteration], ber_legacy, ber_legacy_genie, iteration)
        for iteration in range(iterations):
            fig_deepsic = plot_loss_and_LLRs(train_loss_vect, val_loss_vect, llrs_mat_list[iteration], snr_cur, "DeepSIC",
                               conf.kernel_size, train_samples, val_samples, mod_text, cfo_str, ber_list[iteration],
                               ber_legacy, ber_legacy_genie, iteration)

        data = {
            "SNR_range": SNR_range[:len(total_ber_legacy)],
            "total_ber_legacy": total_ber_legacy,
            "total_ber_legacy_genie": total_ber_legacy_genie,
            "total_ber_deeprx": total_ber_deeprx,
            "total_ber_sphere": total_ber_sphere,
        }

        # Add total_ber from total_ber_list with suffix _1, _2, ...
        for i in range(conf.iterations):
            data[f"total_ber_{i + 1}"] = total_ber_list[i]

        for i in range(iters_e2e_disp):
            data[f"total_ber_e2e_{i + 1}"] = total_ber_e2e_list[i]

        if conf.run_deepsicsb and deepsicsb_trainer is not None:
            for i in range(conf.iterations):
                data[f"total_ber_deepsicsb_{i + 1}"] = total_ber_deepsicsb_list[i]

        if conf.run_deepsicmb and deepsicmb_trainer is not None:
            for i in range(conf.iterations):
                data[f"total_ber_deepsicmb_{i + 1}"] = total_ber_deepsicmb_list[i]

        if conf.run_deepstag and deepstag_trainer is not None:
            for i in range(conf.iterations):
                data[f"total_ber_deepstag_{i + 1}"] = total_ber_deepstag_list[i]


        df = pd.DataFrame(data)


        data_bler = {
            "SNR_range": SNR_range[:len(total_ber_legacy)],
            "total_ber_legacy": total_bler_legacy,
            "total_ber_sphere": total_bler_sphere,
        }

        for i in range(conf.iterations):
            data_bler[f"total_ber_{i + 1}"] = total_bler_list[i]

        df_bler = pd.DataFrame(data_bler)

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

        colors = ['g', 'r', 'k', 'b', 'yellow', 'pink']
        fig, ax = plt.subplots()
        for iter in range(iterations):
            plt.plot(np.arange(conf.num_res), ber_per_re[iter, :], linestyle='-', color=colors[iter], label='BER ' + add_text+ ', iter'+str(iter))
        ax.legend()
        ax.set_title('DeepSIC, ' + title_string)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        fig_deepsic_per_re = fig


        fig, ax = plt.subplots()
        for iter in range(iterations):
            plt.plot(np.arange(conf.num_res), ber_per_re_deepsicsb[iter, :], linestyle='-', color=colors[iter], label='BER ' + add_text+ ', iter'+str(iter))
        ax.legend()
        ax.set_title('DeepSICSB, ' + title_string)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        fig_deepsicsb_per_re = fig

        fig, ax = plt.subplots()
        for iter in range(iterations):
            plt.plot(np.arange(conf.num_res), ber_per_re_deepsicmb[iter, :], linestyle='-', color=colors[iter], label='BER ' + add_text+ ', iter'+str(iter))
        ax.legend()
        ax.set_title('DeepSICMB, ' + title_string)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        fig_deepsicmb_per_re = fig

        fig, ax = plt.subplots()
        for iter in range(iterations):
            plt.plot(np.arange(conf.num_res), ber_per_re_deepstag[iter, :], linestyle='-', color=colors[iter], label='BER ' + add_text+ ', iter'+str(iter))
        ax.legend()
        ax.set_title('DeepSTAG, ' + title_string)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        fig_deepstag_per_re = fig

        fig, ax = plt.subplots()
        plt.plot(np.arange(conf.num_res), ber_per_re_legacy, linestyle='-', color='g', label='BER ' + add_text)
        ax.legend()
        ax.set_title('Legacy, ' + title_string)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        fig_legacy_per_re = fig

        title_string = title_string.replace("\n", "")
        title_string = title_string.replace(",", "")
        title_string = title_string.replace(" ", "_")
        title_string = title_string + '_n_ants=' + str(conf.n_ants)
        # title_string = title_string + '_FFT_size=' + str(FFT_size)
        # title_string = title_string + '_sep_pilots_deeprx=' + str(conf.separate_pilots)
        title_string = title_string + '_' + conf.cur_str
        title_string = title_string + '_seed=' + str(conf.channel_seed)
        title_string = title_string + '_SNR=' + str(conf.snr)
        title_string = formatted_date + title_string
        output_dir = os.path.join(os.getcwd(), '..', 'Scratchpad')
        file_path = os.path.abspath(os.path.join(output_dir, title_string) + ".csv")
        df.to_csv(file_path, index=False)
        if conf.mcs>-1:
            file_path_bler = os.path.abspath(os.path.join(output_dir, title_string) + "_bler.csv")
            df_bler.to_csv(file_path_bler, index=False)

        if snr_cur in conf.save_loss_plot_snr:
            title_string_cur = "deepsic_" + title_string
            file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
            fig_deepsic.savefig(file_path)
            title_string_cur = "deepsic_per_re_" + title_string
            file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
            fig_deepsic_per_re.savefig(file_path)

            title_string_cur = "legacy_" + title_string
            file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
            fig_legacy.savefig(file_path)
            title_string_cur = "legacy_per_re_" + title_string
            file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
            fig_legacy_per_re.savefig(file_path)


            if conf.run_deepsicsb and deepsicsb_trainer is not None:
                title_string_cur = "deepsicsb_"+title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepsicsb.savefig(file_path)
                title_string_cur = "deepsicsb_per_re_"+title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepsicsb_per_re.savefig(file_path)
            if conf.run_deepsicmb and deepsicmb_trainer is not None:
                title_string_cur = "deepsicmb_"+title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepsicmb.savefig(file_path)
                title_string_cur = "deepsicmb_per_re_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepsicmb_per_re.savefig(file_path)
            if conf.run_deepstag and deepstag_trainer is not None:
                title_string_cur = "deepstag_"+title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepstag.savefig(file_path)
                title_string_cur = "deepstag_per_re_" + title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deepstag_per_re.savefig(file_path)
            if conf.run_deeprx:
                title_string_cur = "deeprx_"+title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_deeprx.savefig(file_path)
            if conf.run_e2e:
                title_string_cur = "e2e"+title_string
                file_path = os.path.abspath(os.path.join(output_dir, title_string_cur) + ".jpg")
                fig_e2e.savefig(file_path)


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
    snr_at_target_deepsicsb_list = np.zeros(iterations)
    snr_at_target_deepsicmb_list = np.zeros(iterations)
    snr_at_target_deepstag_list = np.zeros(iterations)
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

        if conf.run_deepsicsb and deepsicsb_trainer is not None:
            for iteration in range(iterations):
                interp_func = interp1d(total_ber_deepsicsb_list[iteration], SNR_range, kind='linear',
                                       fill_value="extrapolate")
                snr_at_target_deepsicsb_list[iteration] = np.round(interp_func(bler_target), 1)
        if conf.run_deepsicmb and deepsicmb_trainer is not None:
            for iteration in range(iterations):
                interp_func = interp1d(total_ber_deepsicmb_list[iteration], SNR_range, kind='linear',
                                       fill_value="extrapolate")
                snr_at_target_deepsicmb_list[iteration] = np.round(interp_func(bler_target), 1)
        if conf.run_deepstag and deepstag_trainer is not None:
            for iteration in range(iterations):
                interp_func = interp1d(total_ber_deepstag_list[iteration], SNR_range, kind='linear',
                                       fill_value="extrapolate")
                snr_at_target_deepstag_list[iteration] = np.round(interp_func(bler_target), 1)
        if conf.run_deeprx:
            interp_func = interp1d(total_ber_deeprx, SNR_range, kind='linear', fill_value="extrapolate")
            snr_at_target_deeprx = np.round(interp_func(bler_target), 1)
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
            for iteration in range(iterations):
                snr_at_target_deepsicsb_list[iteration] = float('inf')
        if conf.run_deepsicmb and deepsicmb_trainer is not None:
            for iteration in range(iterations):
                snr_at_target_deepsicmb_list[iteration] = float('inf')
        if conf.run_deepstag and deepstag_trainer is not None:
            for iteration in range(iterations):
                snr_at_target_deepstag_list[iteration] = float('inf')
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
        if conf.no_probs:
            e2e_text = 'NoProbs'
        else:
            e2e_text = 'Deepe2e'

        for iteration in range(iters_e2e_disp):
            plt.semilogy(SNR_range, total_ber_e2e_list[iteration], linestyle=dashes[iteration],
                         marker=markers[iteration], color='m',
                         label=e2e_text + str(iteration + 1) + ', SNR @1%=' + str(snr_at_target_e2e_list[iteration]))
    if conf.run_deepsicsb and deepsicsb_trainer is not None:
        for iteration in range(iterations):
            plt.semilogy(SNR_range, total_ber_deepsicsb_list[iteration], linestyle=dashes[iteration],
                         marker=markers[iteration], color='orange',
                         label='DeepSICSB' + str(iteration + 1) + ', SNR @1%=' + str(snr_at_target_deepsicsb_list[iteration]))
    if conf.run_deepsicmb and deepsicmb_trainer is not None:
        for iteration in range(iterations):
            plt.semilogy(SNR_range, total_ber_deepsicmb_list[iteration], linestyle=dashes[iteration],
                         marker=markers[iteration], color='black',
                         label='DeepSICMB' + str(iteration + 1) + ', SNR @1%=' + str(snr_at_target_deepsicmb_list[iteration]))
    if conf.run_deepstag and deepstag_trainer is not None:
        for iteration in range(iterations):
            plt.semilogy(SNR_range, total_ber_deepstag_list[iteration], linestyle=dashes[iteration],
                         marker=markers[iteration], color='pink',
                         label='DeepSTAG' + str(iteration + 1) + ', SNR @1%=' + str(snr_at_target_deepstag_list[iteration]))
    if conf.run_deeprx:
        plt.semilogy(SNR_range, total_ber_deeprx, '-o', color='c',
                     label='DeepRx,   SNR @1%=' + str(snr_at_target_deeprx))

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
    assert not (conf.no_probs and conf.iters_e2e != 1 and conf.full_e2e == True), "Assert: No probs only works with 1 iteration or with full e2e"
    assert not (conf.separate_nns and conf.mod_pilot <= 4), "Assert: Can't use separate nns with QPSK"
    assert not (
                conf.no_probs and conf.iters_e2e != 1 and conf.full_e2e == True), "Assert: No probs only works with 1 iteration or with full e2e"

    start_time = time.time()
    num_bits = int(np.log2(conf.mod_pilot))
    deepsic_trainer = DeepSICTrainer(num_bits, conf.n_users, conf.n_ants)
    deepsice2e_trainer = DeepSICe2eTrainer(num_bits, conf.n_users, conf.n_ants)
    deeprx_trainer = DeepRxTrainer(conf.num_res, conf.n_users, conf.n_ants)
    deepsicsb_trainer = DeepSICSBTrainer(num_bits, conf.n_users, conf.n_ants)
    deepsicmb_trainer = DeepSICMBTrainer(num_bits, conf.n_users, conf.n_ants)
    deepstag_trainer = DeepSTAGTrainer(num_bits, conf.n_users, conf.n_ants)



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
    # detectors = deepsic_trainer.detector
    # total_params = sum(
    #     p.numel()
    #     for module in iter_modules(detectors)  # your nested structure
    #     for p in module.parameters()
    #     if p.requires_grad
    # )
    # print(f"Parameters DeepSIC: {total_params}")
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
    # detectors = deepsicsb_trainer.detector
    # total_params = sum(
    #     p.numel()
    #     for module in iter_modules(detectors)  # your nested structure
    #     for p in module.parameters()
    #     if p.requires_grad
    # )
    # print(f"Parameters DeepSICSB: {total_params}")
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


    run_evaluate(deepsic_trainer, deepsice2e_trainer, deeprx_trainer, deepsicsb_trainer, deepsicmb_trainer, deepstag_trainer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    days = int(elapsed_time // (24 * 3600))
    elapsed_time %= 24 * 3600
    hours = int(elapsed_time // 3600)
    elapsed_time %= 3600
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")