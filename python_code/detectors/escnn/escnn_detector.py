import math
import torch
from torch import nn
from python_code import conf

HIDDEN_BASE_SIZE = 16
def inverse_sigmoid(p):
    return torch.log(p / (1 - p))

class ESCNNDetector(nn.Module):
    def __init__(self, num_bits, n_users):
        super().__init__()
        torch.manual_seed(42)

        if conf.no_probs:
            conv_num_channels = int(conf.n_ants * 2)
        else:
            conv_num_channels = int(num_bits * n_users + conf.n_ants * 2)

        hidden_size = HIDDEN_BASE_SIZE * num_bits

        if conf.scale_input:
            self.scale = nn.Parameter(torch.ones(conv_num_channels * conf.num_res))

        # ----- main path -----
        self.fc1 = nn.Conv2d(conv_num_channels, hidden_size, kernel_size=(conf.kernel_size, 1), padding='same')
        self.fc2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(conf.kernel_size, 1), padding='same')
        self.fc3 = nn.Conv2d(hidden_size, num_bits,     kernel_size=(conf.kernel_size, 1), padding='same')
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # ----- single mixing coefficient λ in (0,1) -----
        # λ = sigmoid(mix_raw); initialize λ=0.75 -> mix_raw=logit(0.75)=log(3)
        self.mix_raw = nn.Parameter(torch.tensor(math.log(3.0)))  # scalar

    def forward(self, rx_prob, num_bits, user):
        # optional input scaling (your original)
        if conf.scale_input:
            rx = rx_prob
            rx_flat = rx.reshape(rx.shape[0], rx.shape[1] * rx.shape[2], 1).squeeze(-1)
            rx_out_flat = rx_flat * self.scale
            x = rx_out_flat.unsqueeze(-1).reshape_as(rx)
        else:
            x = rx_prob

        # main path -> LLRs
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        llrs_main = self.fc3(h2)  # [B, num_bits, H, W]

        # skip path -> LLRs directly from input
        start = conf.n_ants*2 + user * num_bits
        end = conf.n_ants*2 + (user + 1) * num_bits
        llrs_skip = inverse_sigmoid(rx_prob[:,start:end,:,:])  # [B, num_bits, H, W]

        # single scalar mix λ in (0,1)
        lam = torch.sigmoid(self.mix_raw)  # scalar

        # llrs_total = lam * llrs_skip + (1.0 - lam) * llrs_main
        # llrs_total = 0.75 * llrs_skip + (1.0 - 0.75) * llrs_main
        llrs_total = 1 * llrs_skip + 0 * llrs_main

        probs = self.sigmoid(llrs_total)
        return probs, llrs_total
