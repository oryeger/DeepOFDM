import torch
from torch import nn

from python_code import conf

from python_code.utils.probs_utils import ensure_tensor_iterable

HIDDEN_BASE_SIZE = 16


class DeepSICe2eDetector(nn.Module):
    def __init__(self, num_bits, n_users):
        super(DeepSICe2eDetector, self).__init__()
        if conf.no_probs:
            conv_num_channels = conf.n_ants*2
        else:
            if conf.separate_nns:
                conv_num_channels =  int(num_bits/2)*n_users+conf.n_ants*2
            else:
                conv_num_channels =  num_bits*n_users+conf.n_ants*2

        hidden_size = HIDDEN_BASE_SIZE * num_bits

        if conf.full_e2e:
            self.num_layers = int(conf.n_users*conf.iters_e2e)
        else:
            self.num_layers = conf.n_users

        if conf.scale_input:
            matrix_size = conv_num_channels*conf.num_res # conf.n_ants*2*conf.num_res
            self.scale = nn.ParameterList([
                nn.Parameter(torch.ones(matrix_size))
                for _ in range(self.num_layers)
            ])


        self.fc1 = nn.ModuleList([
            nn.Conv2d(in_channels=conv_num_channels, out_channels=hidden_size, kernel_size=(conf.kernel_size, 1),padding='same') for _ in range(self.num_layers)])
        if conf.separate_nns:
            self.fc2 = nn.ModuleList([
                nn.Conv2d(in_channels=hidden_size, out_channels=int(num_bits/2), kernel_size=(conf.kernel_size, 1),padding='same') for _ in range(self.num_layers)])
        else:
            self.fc2 = nn.ModuleList([
                nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(conf.kernel_size, 1),padding='same') for _ in range(self.num_layers)])

        self.fc3 = nn.ModuleList([
            nn.Conv2d(in_channels=hidden_size, out_channels=num_bits, kernel_size=(conf.kernel_size, 1), padding='same') for _ in range(self.num_layers)])



        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.Sigmoid()

    def forward(self, rx_prob, num_bits, iters_e2e):
        n_users = conf.n_users
        if conf.separate_nns:
            llrs_mat = torch.zeros(rx_prob.shape[0], n_users * int(num_bits/2), rx_prob.shape[2], rx_prob.shape[3],
                                   device=rx_prob.device)
        else:
            llrs_mat = torch.zeros(rx_prob.shape[0], n_users * num_bits, rx_prob.shape[2], rx_prob.shape[3],
                                   device=rx_prob.device)

        if conf.no_probs:
            if conf.separate_nns:
                rx_prob_new = torch.zeros(rx_prob.shape[0], n_users * int(num_bits / 2), rx_prob.shape[2], rx_prob.shape[3], device=rx_prob.device)
            else:
                rx_prob_new = torch.zeros(rx_prob.shape[0], n_users * num_bits, rx_prob.shape[2], rx_prob.shape[3], device=rx_prob.device)
        else:
            rx_prob_new = rx_prob.clone()  # Ensure it's a copy for updates

        for iteration in ensure_tensor_iterable(iters_e2e):
            for user in range(n_users):
                cur_index = iteration * n_users + user

                if conf.scale_input:
                    rx = rx_prob_new  # rx = rx_prob[:,:conf.n_ants*2,:,:]
                    rx_flattened = rx.reshape(rx.shape[0], rx.shape[1] * rx.shape[2], 1).squeeze(-1)
                    rx_out_flattened = rx_flattened * self.scale[cur_index]
                    rx_out = rx_out_flattened.unsqueeze(-1).reshape_as(rx)
                    out1 = self.activation1(self.fc1[cur_index](rx_out))
                else:
                    out1 = self.activation1(self.fc1[cur_index](rx_prob_new))

                out2 = self.activation2(self.fc2[cur_index](out1))
                llrs = self.fc3[cur_index](out2)
                out3 = self.activation3(llrs)

                if conf.no_probs:
                    if conf.separate_nns:
                        index_start = user * int(num_bits / 2)
                        index_end = (user + 1) * int(num_bits / 2)
                        rx_prob_new = rx_prob_new.clone()
                        rx_prob_new[:, index_start:index_end, :, :] = out3
                        llrs_mat[:, index_start:index_end, :, :] = llrs
                    else:
                        index_start = user * num_bits
                        index_end = (user + 1) * num_bits
                        rx_prob_new = rx_prob_new.clone()
                        rx_prob_new[:, index_start:index_end, :, :] = out3
                        llrs_mat[:, index_start:index_end, :, :] = llrs
                else:
                    if conf.separate_nns:
                        index_start = conf.n_ants * 2 + user * int(num_bits/2)
                        index_end = conf.n_ants * 2 + (user + 1) * int(num_bits/2)
                        # Instead of in-place modification, use slicing and assignment outside the loop
                        rx_prob_new = rx_prob_new.clone()
                        rx_prob_new[:, index_start:index_end, :, :] = out3
                        index_start = user * int(num_bits/2)
                        index_end = (user + 1) * int(num_bits/2)
                        llrs_mat[:, index_start:index_end, :, :] = llrs
                    else:
                        index_start = conf.n_ants * 2 + user * num_bits
                        index_end = conf.n_ants * 2 + (user + 1) * num_bits
                        rx_prob_new = rx_prob_new.clone()
                        rx_prob_new[:, index_start:index_end, :, :] = out3

                        index_start = user * num_bits
                        index_end = (user + 1) * num_bits
                        llrs_mat[:, index_start:index_end, :, :] = llrs

        if conf.no_probs:
            out3 = rx_prob_new.squeeze(-1)  # Output of final rx_prob
        else:
            out3 = rx_prob_new[:, conf.n_ants * 2:, :, :].squeeze(-1)  # Output of final rx_prob
        llrs_mat = llrs_mat.squeeze(-1)
        return out3, llrs_mat