import torch
from torch import nn
from python_code import conf

from python_code.utils.constants import N_ANTS, N_USERS, IS_COMPLEX, NUM_BITS

HIDDEN_BASE_SIZE = 16



class SharedDeepSICDetector(nn.Module):
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
        super(SharedDeepSICDetector, self).__init__()
        conv_num_channels =  NUM_BITS*N_USERS

        self.fc = nn.Conv2d(in_channels=conv_num_channels, out_channels=1,kernel_size=(conf.kernel_size, 1),padding='same')

    def _reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        return self.fc(probs)

class InstanceDeepSICDetector(nn.Module):
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
        super(InstanceDeepSICDetector, self).__init__()
        hidden_size = HIDDEN_BASE_SIZE * NUM_BITS
        base_rx_size = N_ANTS
        if IS_COMPLEX:
            base_rx_size = base_rx_size * 2
        self.fc1 = nn.Linear(base_rx_size+1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, NUM_BITS)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

    def _reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, rx: torch.Tensor, out0_flat: torch.Tensor) -> torch.Tensor:
        out0_cat = torch.cat((out0_flat.view(-1,1),rx), dim=1)
        out1 = self.fc1(out0_cat)
        out2 = self.activation1(out1)
        llrs = self.fc2(out2)
        out3 = self.activation2(llrs)
        return out3, llrs


class DeepSICDetector(nn.Module):
    def __init__(self, num_instances):
        super(DeepSICDetector, self).__init__()
        self.shared_backbone = SharedDeepSICDetector()  # Shared part
        self.instance_heads = nn.ModuleList([InstanceDeepSICDetector() for _ in range(num_instances)])  # Unique parts

    def forward(self, prob, rx, instance_id):
        # Pass through shared backbone
        shared_output = self.shared_backbone(prob)
        # Pass through the specific head for the given instance
        instance_output, llrs = self.instance_heads[instance_id](rx[:,:,instance_id],torch.squeeze(shared_output[:,:,instance_id,:]))
        return instance_output, llrs