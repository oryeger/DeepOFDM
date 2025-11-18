import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

HIDDEN_BASE_SIZE = 16

class ESCNNDetector(nn.Module):
    def __init__(self, num_bits, n_users):
        super(ESCNNDetector, self).__init__()
        torch.manual_seed(42)
        if conf.no_probs:
            conv_num_channels = int(conf.n_ants * 2)
        else:
            conv_num_channels = int(num_bits * n_users + conf.n_ants * 2)

        hidden_size = HIDDEN_BASE_SIZE * num_bits
        if conf.scale_input:
            self.scale = nn.Parameter(torch.ones(conv_num_channels*conf.num_res))

        self.fc1 = nn.Conv2d(in_channels=conv_num_channels, out_channels=hidden_size, kernel_size=(conf.kernel_size, 1),padding='same')
        self.fc2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(conf.kernel_size, 1), padding='same')
        self.fc3 = nn.Conv2d(in_channels=hidden_size, out_channels=num_bits, kernel_size=(conf.kernel_size, 1), padding='same')

        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.Sigmoid()


    def forward(self, rx_prob):
        if conf.scale_input:
            rx = rx_prob
            rx_flattened = rx.reshape(rx.shape[0], rx.shape[1] * rx.shape[2], 1).squeeze(-1)
            rx_out_flattened = rx_flattened*self.scale
            rx_out = rx_out_flattened.unsqueeze(-1).reshape_as(rx)
            out1 = self.activation1(self.fc1(rx_out))
        else:
            out1 = self.activation1(self.fc1(rx_prob))

        out2 = self.activation2(self.fc2(out1))
        llrs = self.fc3(out2)
        out3 = self.activation3(llrs)
        return out3, llrs