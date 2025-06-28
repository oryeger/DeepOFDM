import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

from python_code.utils.constants import N_ANTS

HIDDEN_BASE_SIZE = 64

class DeepSICSBDetector(nn.Module):
    """
    The DeepSIC Network Architecture

    ===========Architecture=========
    DeepSICNet(
      (fullyConnectedLayer): Linear(...)
      (reluLayer): ReLU()
      (fullyConnectedLayer2): Linear(...)
    ================================
    """

    def __init__(self, num_bits, n_users):
        super(DeepSICSBDetector, self).__init__()
        torch.manual_seed(42)
        hidden_size = HIDDEN_BASE_SIZE * num_bits
        base_rx_size = N_ANTS *2
        linear_input = base_rx_size + num_bits * (conf.n_users - 1)  # from DeepSIC paper
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_bits)
        self.activation2 = nn.Sigmoid()

    def _reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, rx: torch.Tensor):
        out0 = self.activation1(self.fc1(rx))
        llrs = self.fc2(out0)
        out1 = self.activation2(llrs)
        return out1, llrs

