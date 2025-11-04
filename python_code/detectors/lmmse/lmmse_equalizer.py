from python_code import conf
import torch
from python_code.channel.modulator import BPSKModulator, QPSKModulator, QAM16Modulator, QAM64Modulator

def LmmseDemod(rx_ce, rx_c, s_orig, noise_var, pilot_chunk, re, num_bits, llrs_mat_lmmse_for_aug, detected_word_lmmse_for_aug, H):
    for user in range(conf.n_users):
        if not conf.separate_pilots:
            rx_pilot_ce_cur = rx_ce[user, :pilot_chunk, :, re]
            s_orig_pilot = s_orig[:pilot_chunk, user, re]
            H[:, user] = 1 / s_orig_pilot.shape[0] * (s_orig_pilot[:, None].conj() / (
                    torch.abs(s_orig_pilot[:, None]) ** 2) * rx_pilot_ce_cur).sum(dim=0)
        else:
            rx_pilot_ce_cur = rx_ce[user, user:pilot_chunk:conf.n_users, :, re]
            s_orig_pilot = s_orig[user:pilot_chunk:conf.n_users, user, re]
            H[:, user] = 1 / s_orig_pilot.shape[0] * (s_orig_pilot[:, None].conj() / (
                    torch.abs(s_orig_pilot[:, None]) ** 2) * rx_pilot_ce_cur).sum(dim=0)

    I_users = torch.eye(conf.n_users, dtype=H.dtype, device=H.device)
    W = torch.linalg.inv(H.T.conj() @ H + noise_var * I_users) @ H.T.conj()
    bias = (W @ H).diag().real
    W = W.cpu()
    bias = bias.cpu()
    equalized = torch.zeros(rx_ce.shape[1], conf.n_users, dtype=torch.cfloat)
    for i in range(rx_ce.shape[1]):
        equalized[i, :] = torch.matmul(W, rx_c[i, :, re]) / bias

    postEqSINR = bias / (1 - bias)

    if conf.mod_pilot == 2:
        for i in range(equalized.shape[1]):
            detected_word_lmmse_for_aug[:, i, re] = torch.from_numpy(
                BPSKModulator.demodulate(-torch.sign(equalized[:, i].real).numpy()))
    elif conf.mod_pilot == 4:
        for user in range(conf.n_users):
            detected_word_lmmse_for_aug[:, user, re], llr_out = QPSKModulator.demodulate(equalized[:, user].numpy())
            llrs_mat_lmmse_for_aug[:, (user * num_bits):((user + 1) * num_bits), re, :] = llr_out.reshape(
                int(llr_out.shape[0] / num_bits), num_bits, 1) * postEqSINR[user].numpy()

    elif conf.mod_pilot == 16:
        for user in range(conf.n_users):
            detected_word_lmmse_for_aug[:, user, re], llr_out = QAM16Modulator.demodulate(equalized[:, user].numpy())
            llrs_mat_lmmse_for_aug[:, (user * num_bits):((user + 1) * num_bits), re, :] = llr_out.reshape(
                int(llr_out.shape[0] / num_bits), num_bits, 1) * postEqSINR[user].numpy()

    elif conf.mod_pilot == 64:
        for user in range(conf.n_users):
            detected_word_lmmse_for_aug[:, user, re], llr_out = QAM64Modulator.demodulate(equalized[:, user].numpy())
            llrs_mat_lmmse_for_aug[:, (user * num_bits):((user + 1) * num_bits), re, :] = llr_out.reshape(
                int(llr_out.shape[0] / num_bits), num_bits, 1) * postEqSINR[user].numpy()

    else:
        print('Unknown modulator')

