import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

HIDDEN_BASE_SIZE = 16


class FiLMLayer(nn.Module):
    """
    FiLM: Feature-wise Linear Modulation
    x:     (B, C, H, W)
    cond_vec:  (B, cond_dim)
    """
    def __init__(self, in_channels: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, in_channels)
        self.beta = nn.Linear(cond_dim, in_channels)

    def forward(self, x, cond_vec):
        # x: (B, C, H, W), cond_vec: (B, cond_dim)
        gamma = self.gamma(cond_vec).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta(cond_vec).unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        return gamma * x + beta


class ESCNNDetector(nn.Module):
    def __init__(self, num_bits, n_users):
        """
        cond_dim:
            Dimension of conditioning vector for FiLM (e.g. SNR features).
            If None, FiLM is disabled and the model behaves exactly like
            the original ESCNNDetector.
        """
        super(ESCNNDetector, self).__init__()
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
        if conf.use_film:
            # FiLM on the hidden feature maps (same C as fc1/fc2 outputs)
            self.film1 = FiLMLayer(hidden_size, cond_dim)
            self.film2 = FiLMLayer(hidden_size, cond_dim)
        else:
            self.film1 = None
            self.film2 = None

    def forward(self, rx_prob):
        """
        rx_prob: (B, C_in, H, W)
        cond_vec:   (B, cond_dim), required if FiLM is enabled
        Returns:
            soft_estimation, llrs   (same as original code)
        """
        if conf.use_film:
            # cond_slice = rx_prob[:, 0:conf.n_ants*2, :, :]
            # B = cond_slice.shape[0]
            # cond_vec = cond_slice.view(B, -1)  # Shape: (B, 384)

            cond_slice = rx_prob[:, 0:16, :, :]
            cond_pooled = F.adaptive_avg_pool2d(cond_slice, 1)
            cond_vec = cond_pooled.squeeze(-1).squeeze(-1)  # Shape: (B, 16)
        else:
            cond_vec = None
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

        # FiLM after first conv
        if conf.use_film:
            out1 = self.film1(out1, cond_vec)

        # Second conv + activation
        out2 = self.activation2(self.fc2(out1))

        # FiLM after second conv
        if conf.use_film:
            out2 = self.film2(out2, cond_vec)

        # Final conv head (LLRs) + sigmoid
        llrs = self.fc3(out2)
        soft_estimation = self.activation3(llrs)

        # IMPORTANT: keep same return signature as original
        return soft_estimation, llrs
