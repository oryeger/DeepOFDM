import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

from python_code.utils.constants import N_ANTS

HIDDEN_BASE_SIZE = 16

class DeepSICDetector(nn.Module):
    def __init__(self, num_bits, n_users):
        super(DeepSICDetector, self).__init__()
        if conf.separate_nns:
            conv_num_channels =  int(num_bits/2)*n_users+N_ANTS*2
        else:
            if conf.half_probs:
                conv_num_channels =  int(num_bits+(num_bits/2)*(n_users-1)+N_ANTS*2)
            else:
                conv_num_channels =  int(num_bits*n_users+N_ANTS*2)
        hidden_size = HIDDEN_BASE_SIZE * num_bits
        matrix_size = N_ANTS*2*conf.num_res
        self.fc0 = nn.Linear(matrix_size, matrix_size, bias=False)

        with torch.no_grad():
            self.fc0.weight.copy_(torch.eye(matrix_size) + 1e-6 * torch.ones(matrix_size, matrix_size))

        self.fc1 = nn.Conv2d(in_channels=conv_num_channels, out_channels=hidden_size, kernel_size=(conf.kernel_size, 1),padding='same')
        if conf.separate_nns:
            self.fc2 = nn.Conv2d(in_channels=hidden_size, out_channels=int(num_bits/2), kernel_size=(conf.kernel_size, 1),padding='same')
        else:
            self.fc2 = nn.Conv2d(in_channels=hidden_size, out_channels=num_bits, kernel_size=(conf.kernel_size, 1),padding='same')
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()


    def forward(self, rx_prob):
        rx = rx_prob[:,:N_ANTS*2,:,:]
        rx_flattened = rx.reshape(rx.shape[0], rx.shape[1] * rx.shape[2], 1).squeeze(-1)
        rx_out_flattened = self.fc0(rx_flattened)
        rx_out = rx_out_flattened.unsqueeze(-1).reshape_as(rx)
        rx_prob_out = rx_prob.clone()
        rx_prob_out[:,:N_ANTS*2,:,:] =  rx_out

        out1 = self.fc1(rx_prob)
        out2 = self.activation1(out1)
        llrs = self.fc2(out2)
        out3 = self.activation2(llrs)
        return out3, llrs
