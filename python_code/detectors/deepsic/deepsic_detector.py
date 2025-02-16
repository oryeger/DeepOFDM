import torch
from torch import nn
from python_code import conf

from python_code.utils.constants import N_ANTS, N_USERS, IS_COMPLEX, NUM_BITS

HIDDEN_BASE_SIZE = 16



class DeepSICDetector(nn.Module):
    """
    The DeepSIC Network Architecture

    ===========Architecture=========
    DeepSICNet(
      (fullyConnectedLayer): Linear(...)
      (reluLayer): ReLU()
      (fullyConnectedLayer2): Linear(...)
    ================================
    """

    def __init__(self):
        super(DeepSICDetector, self).__init__()
        hidden_size = HIDDEN_BASE_SIZE * NUM_BITS
        base_rx_size = N_ANTS
        if IS_COMPLEX:
            base_rx_size = base_rx_size * 2
        conv_num_channels =  NUM_BITS*N_USERS

        self.fc1 = nn.Conv2d(in_channels=conv_num_channels, out_channels=1,kernel_size=(conf.kernel_size, 1),padding='same')
        self.fc2 = nn.Linear(base_rx_size+1, hidden_size)
        self.fc3 = nn.Linear(hidden_size, NUM_BITS)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

    def _reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, probs: torch.Tensor, rx: torch.Tensor) -> torch.Tensor:
        out0 = self.fc1(probs)
        out0_flat = torch.flatten(out0)
        out0_cat = torch.cat((out0_flat.view(-1,1),rx), dim=1)
        out1 = self.fc2(out0_cat)
        out2 = self.activation1(out1)
        out3 = self.fc3(out2)
        out4 = self.activation2(out3)
        return out4

