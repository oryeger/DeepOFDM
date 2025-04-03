import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

from python_code.utils.constants import N_ANTS

HIDDEN_BASE_SIZE = 16

class DeepSICDetector(nn.Module):
    def __init__(self, num_bits, n_users):
        super(DeepSICDetector, self).__init__()
        conv_num_channels =  num_bits*n_users+N_ANTS*2
        hidden_size = HIDDEN_BASE_SIZE * num_bits

        self.fc1 = nn.Conv2d(in_channels=conv_num_channels, out_channels=hidden_size, kernel_size=(conf.kernel_size, 1),
                            padding='same')
        self.fc2 = nn.Conv2d(in_channels=hidden_size, out_channels=num_bits, kernel_size=(conf.kernel_size, 1),
                            padding='same')
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()


    def forward(self, rx_prob):
        # rx_prob_normalized = F.normalize(rx_prob, p=2, dim=1)
        out1 = self.fc1(rx_prob)
        out2 = self.activation1(out1)
        llrs = self.fc2(out2)
        out3 = self.activation2(llrs)
        return out3, llrs
