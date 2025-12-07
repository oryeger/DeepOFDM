import torch
from torch import nn
import torch.nn.functional as F
from python_code.utils.constants import FFT_size,NUM_SYMB_PER_SLOT
from python_code import DEVICE

from python_code import conf

HIDDEN_BASE_SIZE = 16

class ResNetBlockTD(nn.Module):
    """
    Time-domain ResNet block:
    Conv -> BN -> ReLU -> Conv -> BN -> skip connection -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super().__init__()

        # 'same' padding for odd kernel sizes (e.g., 3x3)
        padding = tuple(k // 2 for k in kernel_size)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If channel count changes, adapt the skip path with 1x1 conv
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.activation(out)
        return out


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

        conv_num_channels = 2*conf.n_ants

        # Three ResNet blocks:
        # 1st: input_channels  -> 64
        # 2nd: 64              -> 128
        # 3rd: 128             -> 256
        self.tdblock1 = ResNetBlockTD(conv_num_channels, 64, kernel_size=(3, 3))
        self.tdblock2 = ResNetBlockTD(64, 128, kernel_size=(3, 3))
        self.tdblock3 = ResNetBlockTD(128, 256, kernel_size=(3, 3))

        # Final projection back to input channel dimension
        self.td_out_conv = nn.Conv2d(
            in_channels=256,
            out_channels=conv_num_channels,
            kernel_size=1,
            bias=True,
        )


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
        NUM_SLOTS =  rx_prob.shape[0]//NUM_SYMB_PER_SLOT
        NUM_REs = rx_prob.shape[2]

        rx_fd = torch.zeros_like(rx_prob, device=DEVICE)

        s_t_matrix = torch.zeros((NUM_SLOTS, 2 * conf.n_ants, FFT_size, NUM_SYMB_PER_SLOT), dtype=torch.float32, device=DEVICE)
        for ant in range(conf.n_ants):
            for slot_num in range(NUM_SLOTS):
                for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                    cur_index = slot_num * NUM_SYMB_PER_SLOT + ofdm_symbol
                    cur_symbol = rx_prob[cur_index, 2 * ant, :] + 1j * rx_prob[cur_index, 2 * ant + 1, :]
                    s_t = torch.fft.ifft(torch.squeeze(cur_symbol), n=FFT_size)
                    s_t_matrix[slot_num, 2 * ant, :, ofdm_symbol] = torch.real(s_t)
                    s_t_matrix[slot_num, 2 * ant + 1, :, ofdm_symbol] = torch.imag(s_t)

        # ResNet blocks
        s_t_matrix = self.tdblock1(s_t_matrix)
        s_t_matrix = self.tdblock2(s_t_matrix)
        s_t_matrix = self.tdblock3(s_t_matrix)
        s_t_matrix = self.td_out_conv(s_t_matrix)  # raw regression output, no sigmoid

        for ant in range(conf.n_ants):
            index = 0
            for slot_num in range(NUM_SLOTS):
                for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                    rx_t = s_t_matrix[slot_num, 2 * ant, :, ofdm_symbol] + 1j * s_t_matrix[slot_num, 2 * ant + 1, :,
                                                                                ofdm_symbol]
                    rx_f = torch.fft.fft(rx_t, n=FFT_size)
                    rx_fd[index, 2 * ant, :, 0] = torch.real(rx_f[:NUM_REs])
                    rx_fd[index, 2 * ant + 1, :, 0] = torch.imag(rx_f[:NUM_REs])
                    index += 1

        # ---------- original scaling logic ---------------
        if conf.scale_input:
            rx = rx_fd
            # (B, C, H) -> flatten C*H
            rx_flattened = rx.reshape(rx.shape[0], rx.shape[1] * rx.shape[2], 1).squeeze(-1)
            rx_out_flattened = rx_flattened * self.scale
            rx_out = rx_out_flattened.unsqueeze(-1).reshape_as(rx)
            out1 = self.fc1(rx_out)
        else:
            out1 = self.fc1(rx_fd)

        out1 = self.activation1(out1)

        # Second conv + activation
        out2 = self.fc2(out1)

        out2 = self.activation2(out2)

        # Final conv head (LLRs) + sigmoid
        llrs = self.fc3(out2)
        soft_estimation = self.activation3(llrs)

        # IMPORTANT: keep same return signature as original
        return soft_estimation, llrs
