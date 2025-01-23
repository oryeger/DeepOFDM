import torch
from torch import nn


from python_code.utils.constants import N_ANTS, N_USERS, IS_COMPLEX, NUM_BITS

HIDDEN_BASE_SIZE = 64



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
        linear_input = base_rx_size + NUM_BITS * (N_USERS - 1)  # from DeepSIC paper
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, NUM_BITS)
        self.activation2 = nn.Sigmoid()

    def _reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        out0 = self.activation1(self.fc1(rx))
        out1 = self.fc2(out0)
        out2 = self.activation2(out1)
        return out2

