import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

HIDDEN_BASE_SIZE = 16


class TDFDCNNDetector(nn.Module):
    def __init__(self, num_bits, n_users):
        """
        cond_dim:
            Dimension of conditioning vector for FiLM (e.g. SNR features).
            If None, FiLM is disabled and the model behaves exactly like
            the original TDFDCNNDetector.
        """
        super(TDFDCNNDetector, self).__init__()
        torch.manual_seed(42)

        # --------- original channel computation ----------
        if conf.no_probs:
            conv_num_channels = int(conf.n_ants * 2)
        else:
            conv_num_channels = int(num_bits * n_users + conf.n_ants * 2)

        hidden_size = HIDDEN_BASE_SIZE * num_bits

        # --------- original scaling parameter ------------
        if conf.scale_input:
            self.scale = nn.Parameter(torch.ones(conv_num_channels * conf.num_res))
        else:
            self.scale = None

        # --------- original conv layers ------------------
        self.fc1 = nn.Conv2d(
            in_channels=conv_num_channels,
            out_channels=hidden_size,
            kernel_size=(conf.kernel_size, 1),
            padding="same"
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=(conf.kernel_size, 1),
            padding="same"
        )
        self.fc3 = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=num_bits,
            kernel_size=(conf.kernel_size, 1),
            padding="same"
        )

        # --------- original activations ------------------
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.Sigmoid()

        # --------- FiLM integration (optional) -----------
        # cond_dim = conf.n_ants * 2 * conf.num_res
        cond_dim = conf.n_ants * 2
    def forward(self, rx_prob):
        """
        rx_prob: (B, C_in, H, W)
        cond_vec:   (B, cond_dim), required if FiLM is enabled
        Returns:
            soft_estimation, llrs   (same as original code)
        """
        # ---------- original scaling logic ---------------
        if conf.scale_input:
            rx = rx_prob
            # (B, C, H) -> flatten C*H
            rx_flattened = rx.reshape(rx.shape[0], rx.shape[1] * rx.shape[2], 1).squeeze(-1)
            rx_out_flattened = rx_flattened * self.scale
            rx_out = rx_out_flattened.unsqueeze(-1).reshape_as(rx)
            out1 = self.fc1(rx_out)
        else:
            out1 = self.fc1(rx_prob)

        out1 = self.activation1(out1)

        # Second conv + activation
        out2 = self.fc2(out1)

        out2 = self.activation2(out2)

        # Final conv head (LLRs) + sigmoid
        llrs = self.fc3(out2)
        soft_estimation = self.activation3(llrs)

        # IMPORTANT: keep same return signature as original
        return soft_estimation, llrs
