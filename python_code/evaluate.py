import os
import time
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.detectors.deeprx.deeprx_trainer import DeepRxTrainer
from typing import List

import numpy as np
import torch
from python_code import DEVICE, conf
from python_code.utils.metrics import calculate_ber
import matplotlib.pyplot as plt
from python_code.utils.constants import (IS_COMPLEX, TRAIN_PERCENTAGE, INTERF_FACTOR, CFO_COMP, GENIE_CFO,
                                         FFT_size, FIRST_CP, CP, NUM_SYMB_PER_SLOT, NUM_SAMPLES_PER_SLOT)

import commpy.modulation as mod

from python_code.channel.modulator import BPSKModulator
import pandas as pd

from python_code.channel.channel_dataset import  ChannelModelDataset
from scipy.stats import entropy
from scipy.interpolate import interp1d



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def entropy_with_bin_width(data, bin_width):
    """Estimate entropy using histogram binning with a specified bin width."""
    min_x, max_x = np.min(data), np.max(data)
    bins = np.arange(min_x, max_x + bin_width, bin_width)  # Define bin edges
    hist, _ = np.histogram(data, bins=bins, density=True)  # Compute histogram
    hist = hist[hist > 0]  # Remove zero probabilities
    return entropy(hist, base=2) # Compute entropy


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

def plot_loss_and_LLRs(train_loss_vect, val_loss_vect, llrs_mat, snr_cur, detector, kernel_size, train_samples, val_samples, mod_text, cfo_str, ber, ber_legacy, ber_legacy_genie):
    num_res = conf.num_res
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 4.8))
    epochs_vect = list(range(1, len(train_loss_vect) + 1))
    axes[0].plot(epochs_vect[0::conf.num_res], train_loss_vect[0::num_res], linestyle='-', color='b',
                 label='Training Loss')
    axes[0].plot(epochs_vect[0::num_res], val_loss_vect[0::conf.num_res], linestyle='-', color='r',
                 label='Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    title_string = (detector + ', ' + mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ', SNR=' + str(
        snr_cur) + ", #REs=" + str(num_res) + ', Interf=' + str(INTERF_FACTOR) + ', #UEs=' + str(
        conf.n_users) + '\n ' +
                    cfo_str + ', Epochs=' + str(conf.epochs) + ', #Iterations=' + str(
                conf.iterations) + ', CNN kernel size=' + str(kernel_size))

    axes[0].set_title(title_string, fontsize=8)
    axes[0].legend()
    axes[0].grid()

    axes[1].hist(llrs_mat.cpu().flatten(), bins=30, color='blue', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('LLRs')
    axes[1].set_ylabel('#Values')
    axes[1].grid()
    text = 'BER '+detector+':' + str(f"{ber:.4f}") + '\
             BER legacy:' + str(f"{ber_legacy:.4f}") + '\
             BER legacy genie:' + (f"{ber_legacy_genie:.4f}")
    # axes[2].text(0.5, 0.5, text, fontsize=12, ha="center", va="center")
    # axes[2].axis('off')
    fig.text(0.15, 0.02, text, ha="left", va="center", fontsize=8)
    # fig.text(0.15, 0.02, text, ha="left", va="center", fontsize=12)
    plt.tight_layout()
    plt.show()


def run_evaluate(deepsic_trainer, deeprx_trainer) -> List[float]:
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
    epochs = conf.epochs

    if mod_pilot == 2:
        mod_text = 'BPSK'
    elif mod_pilot == 4:
        mod_text = 'QPSK'
    else:
        mod_text = [str(mod_pilot) + 'QAM']
        mod_text = mod_text[0]

    total_ber = []
    total_ber_deeprx = []
    total_ber_legacy = []
    total_ber_legacy_ce_on_data = []
    total_ber_legacy_genie = []


    SNR_range = [conf.snr + i for i in range(conf.num_snrs)]
    total_mi = []
    total_mi_deeprx = []
    if mod_pilot == 4:
        total_mi_legacy = []
    Final_SNR = conf.snr + conf.num_snrs - 1
    for snr_cur in SNR_range:
        ber_sum = 0
        ber_sum_deeprx = 0
        ber_sum_legacy = 0
        ber_sum_legacy_ce_on_data = 0
        ber_sum_legacy_genie = 0
        deepsic_trainer._initialize_detector(num_bits, n_users)  # For reseting the weights
        deeprx_trainer._initialize_detector(num_bits, n_users)

        pilot_size = get_next_divisible(conf.pilot_size,num_bits*NUM_SYMB_PER_SLOT)
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
                                                   cfo_in_rx=conf.cfo_in_rx,
                                                   kernel_size=conf.kernel_size,
                                                   n_users=n_users)

        transmitted_words, received_words, received_words_ce, hs, s_orig_words = channel_dataset.__getitem__(snr_list=[snr_cur], num_bits=num_bits, n_users
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
                    cfo_est =  conf.cfo
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
                        cfo_est_vect[user] = -np.angle(grad_sum)*FFT_size/(2*np.pi*(FFT_size+CP))
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
                cfo_comp_vect = np.tile(cfo_comp_vect,NUM_SLOTS)

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
                train_loss_vect, val_loss_vect = deepsic_trainer._online_training(tx_pilot, rx_pilot, num_bits, n_users, iterations, epochs)
                # Zero CNN weights
                detected_word, llrs_mat = deepsic_trainer._forward(rx_data, num_bits, n_users, iterations)

            if deeprx_trainer.is_online_training:
                train_loss_vect_deeprx, val_loss_vect_deeprx = deeprx_trainer._online_training(tx_pilot, rx_pilot, num_bits, n_users, iterations, epochs)
                # Zero CNN weights
                detected_word_deeprx, llrs_mat_deeprx = deeprx_trainer._forward(rx_data, num_bits, n_users, iterations)



            # CE Based
            # train_loss_vect = [0] * epochs
            # val_loss_vect = [0] * epochs
            rx_data_c = rx[pilot_chunk:].cpu()
            llrs_mat_legacy = torch.zeros([rx_data_c.shape[0],num_bits*n_users,num_res,1])

            for re in range(conf.num_res):
                # Regular CE
                H = torch.zeros_like(h[:, :, re])
                for user in range(n_users):
                    s_orig_pilot = s_orig[:pilot_chunk, user, re]
                    rx_pilot_ce_cur = rx_ce[user, :pilot_chunk, :, re]
                    H[:, user] = 1 / s_orig_pilot.shape[0] * (s_orig_pilot[:, None].conj() / (
                                torch.abs(s_orig_pilot[:, None]) ** 2) * rx_pilot_ce_cur).sum(dim=0)
                H = H.cpu().numpy()

                H_Ht = H @ H.T.conj()
                H_Ht_inv = np.linalg.pinv(H_Ht)
                H_pi = torch.tensor(H.T.conj() @ H_Ht_inv)
                equalized = torch.zeros(rx_data_c.shape[0], tx_data.shape[1], dtype=torch.cfloat)
                for i in range(rx_data_c.shape[0]):
                    equalized[i, :] = torch.matmul(H_pi, rx_data_c[i, :, re])
                detected_word_legacy = torch.zeros(int(equalized.shape[0] * np.log2(mod_pilot)), equalized.shape[1])
                if mod_pilot > 2:
                    qam = mod.QAMModem(mod_pilot)
                    for i in range(equalized.shape[1]):
                        detected_word_legacy[:, i] = torch.from_numpy(qam.demodulate(equalized[:, i].numpy(), 'hard'))
                else:
                    for i in range(equalized.shape[1]):
                        detected_word_legacy[:, i] = torch.from_numpy(
                            BPSKModulator.demodulate(-torch.sign(equalized[:, i].real).numpy()))

                for user in range(n_users):
                    llrs_mat_legacy[:,(user*2):((user+1)*2),re, :] = torch.view_as_real(equalized[:,user]).unsqueeze(2)

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

                detected_word_cur_re = detected_word[:, :, re, :]
                detected_word_cur_re = detected_word_cur_re.squeeze(-1)
                detected_word_cur_re = detected_word_cur_re.reshape(int(tx_data.shape[0] / num_bits), n_users,
                                                                    num_bits).swapaxes(1, 2).reshape(tx_data.shape[0],
                                                                                                     n_users)

                detected_word_cur_re_deeprx = detected_word_deeprx[:, :, re, :]
                detected_word_cur_re_deeprx = detected_word_cur_re_deeprx.squeeze(-1)
                detected_word_cur_re_deeprx = detected_word_cur_re_deeprx.reshape(int(tx_data.shape[0] / num_bits), n_users,
                                                                    num_bits).swapaxes(1, 2).reshape(tx_data.shape[0],
                                                                                                     n_users)

                ber = calculate_ber(detected_word_cur_re.cpu(), target.cpu(),num_bits)
                ber_deeprx = calculate_ber(detected_word_cur_re_deeprx.cpu(), target.cpu(),num_bits)
                ber_legacy = calculate_ber(detected_word_legacy.cpu(), target.cpu(),num_bits)
                ber_legacy_ce_on_data = calculate_ber(detected_word_legacy_ce_on_data.cpu(), target.cpu(),num_bits)
                ber_legacy_genie = calculate_ber(detected_word_legacy_genie.cpu(), target.cpu(),num_bits)

                ber_sum += ber
                ber_sum_deeprx += ber_deeprx
                ber_sum_legacy += ber_legacy
                ber_sum_legacy_ce_on_data += ber_legacy_ce_on_data
                ber_sum_legacy_genie += ber_legacy_genie

            mi = calc_mi(tx_data.cpu(), llrs_mat.cpu(), num_bits, n_users, num_res)
            total_mi.append(mi)
            mi_deeprx = calc_mi(tx_data.cpu(), llrs_mat_deeprx.cpu(), num_bits, n_users, num_res)
            total_mi_deeprx.append(mi_deeprx)
            if mod_pilot == 4:
                mi_legacy = calc_mi(tx_data.cpu(), llrs_mat_legacy, num_bits, n_users, num_res)
                total_mi_legacy.append(mi_legacy)
            ber = ber_sum/num_res
            ber_deeprx = ber_sum_deeprx/num_res
            ber_legacy = ber_sum_legacy/num_res
            ber_legacy_ce_on_data = ber_sum_legacy_ce_on_data/num_res
            ber_legacy_genie = ber_sum_legacy_genie/num_res

            total_ber.append(ber_sum)
            total_ber_deeprx.append(ber_sum_deeprx)
            total_ber_legacy.append(ber_sum_legacy)
            total_ber_legacy_ce_on_data.append(ber_sum_legacy_ce_on_data)
            total_ber_legacy_genie.append(ber_sum_legacy_genie)
            print(f'SNR={snr_cur}dB, Final SNR={Final_SNR}dB')
            print(f'current DeepSIC: {block_ind, ber, mi}')
            print(f'current DeepRx: {block_ind, ber_deeprx, mi_deeprx}')
            if mod_pilot == 4:
                print(f'current legacy: {block_ind, ber_legacy, mi_legacy}')
            else:
                print(f'current legacy: {block_ind, ber_legacy}')
            print(f'current legacy ce on data: {block_ind, ber_legacy_ce_on_data}')
            print(
                f'current legacy genie: {block_ind, ber_legacy_genie}')  # print(f'Final BER: {sum(total_ber) / len(total_ber)}')
        if conf.cfo != 0:
            if conf.cfo_in_rx:
                cfo_str = 'cfo in Rx=' + str(conf.cfo) + ' scs'
            else:
                cfo_str = 'cfo in Tx=' + str(conf.cfo) + ' scs'
        else:
            cfo_str = 'cfo=0'

        if mod_pilot == 4:
            plot_loss_and_LLRs([0] * len(train_loss_vect_deeprx), [0] * len(val_loss_vect_deeprx), llrs_mat_legacy, snr_cur, "Legacy", 0, train_samples, val_samples, mod_text, cfo_str, ber_legacy, ber_legacy, ber_legacy_genie)
        plot_loss_and_LLRs(train_loss_vect_deeprx, val_loss_vect_deeprx, llrs_mat_deeprx, snr_cur, "DeepRx", 3, train_samples, val_samples, mod_text, cfo_str, ber_deeprx, ber_legacy, ber_legacy_genie)
        plot_loss_and_LLRs(train_loss_vect, val_loss_vect, llrs_mat, snr_cur, "DeepSIC", conf.kernel_size, train_samples, val_samples, mod_text, cfo_str, ber, ber_legacy, ber_legacy_genie)

        # np.save('C:\\Projects\\Misc\\tx_data_-10dB_QPSK.npy', tx_data.cpu())
        # np.save('C:\\Projects\\Misc\\llrs_mat_-10dB_QPSK.npy', llrs_mat.cpu())

    plt.semilogy(SNR_range, total_mi, '-x', color='g', label='DeeSIC')
    plt.semilogy(SNR_range, total_mi_deeprx, '-o', color='c', label='DeepRx')
    if mod_pilot == 4:
        plt.semilogy(SNR_range, total_mi_legacy, '-o', color='r', label='Legacy')
    plt.xlabel('SNR (dB)')
    plt.ylabel('MI')
    title_string = (mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ", #REs=" + str(
        conf.num_res) + ', Interf=' + str(INTERF_FACTOR) + ', #UEs=' + str(n_users) + '\n ' +
                    cfo_str + ', Epochs=' + str(epochs) + ', #Iterations=' + str(
                iterations) + ', CNN kernel size=' + str(conf.kernel_size))
    plt.title(title_string, fontsize=10)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    bler_target = 0.01
    interp_func = interp1d(total_ber, SNR_range, kind='linear', fill_value="extrapolate")
    snr_at_target = np.round(interp_func(bler_target), 1)
    interp_func = interp1d(total_ber_deeprx, SNR_range, kind='linear', fill_value="extrapolate")
    snr_at_target_deeprx = np.round(interp_func(bler_target), 1)
    interp_func = interp1d(total_ber_legacy, SNR_range, kind='linear', fill_value="extrapolate")
    snr_at_target_legacy = np.round(interp_func(bler_target), 1)
    interp_func = interp1d(total_ber_legacy_ce_on_data, SNR_range, kind='linear', fill_value="extrapolate")
    snr_at_target_legacy_ce_on_data = np.round(interp_func(bler_target), 1)

    plt.semilogy(SNR_range, total_ber, '-x', color='g', label='DeepSIC, SNR @1%='+str(snr_at_target))
    plt.semilogy(SNR_range, total_ber_deeprx, '-o', color='c', label='DeepRx,  SNR @1%='+str(snr_at_target_deeprx))
    plt.semilogy(SNR_range, total_ber_legacy, '-o', color='r', label='Legacy,   SNR @1%='+str(snr_at_target_legacy))
    plt.semilogy(SNR_range, total_ber_legacy_ce_on_data, '-o', color='b', label='CE Data,  SNR @1%='+str(snr_at_target_legacy_ce_on_data))

    # plt.semilogy(SNR_range, total_ber_legacy_genie, '-o', color='g', label='Legacy Genie')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    title_string = (mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ", #REs=" + str(
        conf.num_res) + ', Interf=' + str(INTERF_FACTOR) + ', #UEs=' + str(n_users) + '\n ' +
                    cfo_str + ', Epochs=' + str(epochs) + ', #Iterations=' + str(
                iterations) + ', CNN kernel size=' + str(conf.kernel_size))
    plt.title(title_string, fontsize=10)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame({"SNR_range": SNR_range, "total_ber": total_ber, "total_ber_deeprx": total_ber_deeprx,
                       "total_ber_legacy": total_ber_legacy, "total_ber_legacy_genie": total_ber_legacy_genie}, )
    # print('\n'+title_string)
    title_string = title_string.replace("\n", "")
    df.to_csv("C:\\Projects\\Scatchpad\\" + title_string + ".csv", index=False)
    # Look at teh weights:
    # print(deepsic_trainer.detector[0][0].shared_backbone.fc.weight)
    # print(deepsic_trainer.detector[1][0].instance_heads[0].fc1.weight[0])

    return total_ber


if __name__ == '__main__':
    start_time = time.time()
    deepsic_trainer = DeepSICTrainer(conf.num_res, conf.n_users)
    deeprx_trainer = DeepRxTrainer(conf.num_res, conf.n_users)
    print(deepsic_trainer)
    run_evaluate(deepsic_trainer,deeprx_trainer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    days = int(elapsed_time // (24 * 3600))
    elapsed_time %= 24 * 3600
    hours = int(elapsed_time // 3600)
    elapsed_time %= 3600
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")

