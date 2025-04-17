import torch
from torch import nn

from python_code import conf

from python_code.utils.constants import N_ANTS
from python_code.utils.probs_utils import ensure_tensor_iterable

HIDDEN_BASE_SIZE = 16


class DeepSICe2eDetector(nn.Module):
    def __init__(self, num_bits, n_users):
        super(DeepSICe2eDetector, self).__init__()
        if conf.separate_nns:
            conv_num_channels =  int(num_bits/2)*n_users+N_ANTS*2
        else:
            conv_num_channels =  num_bits*n_users+N_ANTS*2
        hidden_size = HIDDEN_BASE_SIZE * num_bits

        if conf.full_e2e:
            self.num_layers = int(conf.n_users*conf.iters_e2e)
        else:
            self.num_layers = conf.n_users

        self.fc1 = nn.ModuleList([
            nn.Conv2d(in_channels=conv_num_channels, out_channels=hidden_size, kernel_size=(conf.kernel_size, 1),padding='same') for _ in range(self.num_layers)])
        if conf.separate_nns:
            self.fc2 = nn.ModuleList([
                nn.Conv2d(in_channels=hidden_size, out_channels=int(num_bits/2), kernel_size=(conf.kernel_size, 1),padding='same') for _ in range(self.num_layers)])
        else:
            self.fc2 = nn.ModuleList([
                nn.Conv2d(in_channels=hidden_size, out_channels=num_bits, kernel_size=(conf.kernel_size, 1),padding='same') for _ in range(self.num_layers)])

        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

    def forward(self, rx_prob, num_bits, iters_e2e):
        n_users = conf.n_users
        if conf.separate_nns:
            llrs_mat = torch.zeros(rx_prob.shape[0], n_users * int(num_bits/2), rx_prob.shape[2], rx_prob.shape[3],
                                   device=rx_prob.device)
        else:
            llrs_mat = torch.zeros(rx_prob.shape[0], n_users * num_bits, rx_prob.shape[2], rx_prob.shape[3],
                                   device=rx_prob.device)

        rx_prob_new = rx_prob.clone()  # Ensure it's a copy for updates

        for iteration in ensure_tensor_iterable(iters_e2e):
            for user in range(n_users):
                cur_index = iteration * n_users + user
                out1 = self.fc1[cur_index](rx_prob_new)
                out2 = self.activation1(out1)
                llrs = self.fc2[cur_index](out2)
                out3 = self.activation2(llrs)

                if conf.separate_nns:
                    index_start = N_ANTS * 2 + user * int(num_bits/2)
                    index_end = N_ANTS * 2 + (user + 1) * int(num_bits/2)
                    # Instead of in-place modification, use slicing and assignment outside the loop
                    rx_prob_new = rx_prob_new.clone()
                    rx_prob_new[:, index_start:index_end, :, :] = out3
                    index_start = user * int(num_bits/2)
                    index_end = (user + 1) * int(num_bits/2)
                    llrs_mat[:, index_start:index_end, :, :] = llrs
                else:
                    index_start = N_ANTS * 2 + user * num_bits
                    index_end = N_ANTS * 2 + (user + 1) * num_bits
                    rx_prob_new = rx_prob_new.clone()
                    rx_prob_new[:, index_start:index_end, :, :] = out3

                    index_start = user * num_bits
                    index_end = (user + 1) * num_bits
                    llrs_mat[:, index_start:index_end, :, :] = llrs

        out3 = rx_prob_new[:, N_ANTS * 2:, :, :].squeeze(-1)  # Output of final rx_prob
        llrs_mat = llrs_mat.squeeze(-1)
        return out3, llrs_mat