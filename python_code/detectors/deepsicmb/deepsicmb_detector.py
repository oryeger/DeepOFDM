import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

HIDDEN_BASE_SIZE = 64

class DeepSICMBDetector(nn.Module):
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
        super(DeepSICMBDetector, self).__init__()
        torch.manual_seed(42)
        hidden_size = HIDDEN_BASE_SIZE * num_bits
        base_rx_size = conf.n_ants *2
        # OryEger
        # linear_input = base_rx_size + num_bits * (conf.n_users - 1)  # from DeepSIC paper
        linear_input =  (base_rx_size + num_bits * conf.n_users) * conf.kernel_size # from DeepSIC paper
        self.fc1 = nn.Linear(linear_input, hidden_size)
        # n_prob_effective = num_bits * conf.n_users
        # kernel_center = int(torch.ceil(torch.tensor(conf.kernel_size / 2-1)).item())
        # vec1 = torch.arange(kernel_center*base_rx_size, (kernel_center+1)*base_rx_size)
        # vec2 = torch.arange(conf.kernel_size*base_rx_size+kernel_center*n_prob_effective, conf.kernel_size*base_rx_size+(kernel_center+1)*n_prob_effective)
        # exclude_indexes = torch.cat((vec1, vec2))
        # all_indexes = torch.arange(linear_input)
        # mask = ~torch.isin(all_indexes, exclude_indexes)
        # remaining_indexes = all_indexes[mask]
        # with torch.no_grad():
        #     self.fc1.weight[:,remaining_indexes] = 1e-6
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

