from python_code import conf
import torch
from python_code.channel.modulator import BPSKModulator, QPSKModulator, QAM16Modulator, QAM64Modulator
import numpy as np

def LmmseEqualize(rx_ce, rx_c, s_orig, ext_noise_var, pilot_chunk, re, H):
    noise_var = 0
    for user in range(conf.n_users):
        if not conf.separate_pilots:
            rx_pilot_ce_cur = rx_ce[user, :pilot_chunk, :, re]
            s_orig_pilot = s_orig[:pilot_chunk, user, re]
            LS_channel = (s_orig_pilot[:, None].conj() / (torch.abs(s_orig_pilot[:, None]) ** 2) * rx_pilot_ce_cur)
            H[:, user] = 1 / s_orig_pilot.shape[0] * LS_channel.sum(dim=0)
            noise_var = torch.mean(torch.abs(LS_channel - H[:, user])**2)
        else:
            rx_pilot_ce_cur = rx_ce[user, user:pilot_chunk:conf.n_users, :, re]
            s_orig_pilot = s_orig[user:pilot_chunk:conf.n_users, user, re]
            LS_channel = (s_orig_pilot[:, None].conj() / (torch.abs(s_orig_pilot[:, None]) ** 2) * rx_pilot_ce_cur)
            H[:, user] = 1 / s_orig_pilot.shape[0] * LS_channel.sum(dim=0)
            noise_var = torch.mean(torch.abs(LS_channel - H[:, user])**2)

    if conf.override_noise_var:
        noise_var = ext_noise_var

    I_users = torch.eye(conf.n_users, dtype=H.dtype, device=H.device)
    W = torch.linalg.inv(H.T.conj() @ H + noise_var * I_users) @ H.T.conj()
    bias = (W @ H).diag().real
    W = W.cpu()
    bias = bias.cpu()
    equalized = torch.zeros(rx_ce.shape[1], conf.n_users, dtype=torch.cfloat)
    for i in range(rx_ce.shape[1]):
        equalized[i, :] = torch.matmul(W, rx_c[i, :, re]) / bias

    postEqSINR = bias / (1 - bias)

    return equalized, postEqSINR

def LmmseDemod(equalized, postEqSINR, num_bits, re, llrs_mat_lmmse_for_aug, detected_word_lmmse_for_aug, skip):
    if num_bits == 1:
        for i in range(equalized.shape[1]):
            detected_word_lmmse_for_aug[:, i, re] = torch.from_numpy(
                BPSKModulator.demodulate(-torch.sign(equalized[:, i].real).numpy()))
    elif num_bits == 2:
        for user in range(conf.n_users):
            if skip!= 1:
                llr_out = np.zeros(detected_word_lmmse_for_aug.shape[0])
                detected_word_lmmse_for_aug[::skip, user, re], llr_out[::skip] = QPSKModulator.demodulate(
                    equalized[:, user].numpy())
                num_bits_int = int(skip*num_bits)
            else:
                detected_word_lmmse_for_aug[:, user, re], llr_out = QAM16Modulator.demodulate(
                    equalized[:, user].numpy())
                num_bits_int = num_bits

            llrs_mat_lmmse_for_aug[:, (user * num_bits_int):((user + 1) * num_bits_int), re, :] = llr_out.reshape(
                int(llr_out.shape[0] / num_bits_int), num_bits_int, 1) * postEqSINR[user].numpy()

    elif num_bits == 4:
        for user in range(conf.n_users):
            detected_word_lmmse_for_aug[:, user, re], llr_out = QAM16Modulator.demodulate(
                equalized[:, user].numpy())
            llrs_mat_lmmse_for_aug[:, (user * num_bits):((user + 1) * num_bits), re, :] = llr_out.reshape(
                int(llr_out.shape[0] / num_bits), num_bits, 1) * postEqSINR[user].numpy()

    elif num_bits == 6:
        for user in range(conf.n_users):
            detected_word_lmmse_for_aug[:, user, re], llr_out = QAM64Modulator.demodulate(equalized[:, user].numpy())
            llrs_mat_lmmse_for_aug[:, (user * num_bits):((user + 1) * num_bits), re, :] = llr_out.reshape(
                int(llr_out.shape[0] / num_bits), num_bits, 1) * postEqSINR[user].numpy()

    else:
        print('Unknown modulator')

    return detected_word_lmmse_for_aug, llrs_mat_lmmse_for_aug

