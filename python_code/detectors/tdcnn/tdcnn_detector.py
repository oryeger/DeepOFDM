import torch
from torch import nn

from python_code import conf


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


class TDCNNDetector(nn.Module):
    """
    Time-domain CNN for regression (L2 / MSE loss).
    - No separate stem: first ResNet block maps input_channels -> 64
    - Three ResNet blocks: (C_in -> 64) -> (64 -> 128) -> (128 -> 256)
    - Final conv projects 256 -> C_in
    - Output shape matches input shape.
    """
    def __init__(self, num_bits, n_users):
        super(TDCNNDetector, self).__init__()
        torch.manual_seed(42)

        # Number of input channels: probs + real/imag antennas (same logic as before)
        conv_num_channels = 2*conf.n_ants

        # Three ResNet blocks:
        # 1st: input_channels  -> 64
        # 2nd: 64              -> 128
        # 3rd: 128             -> 256
        self.block1 = ResNetBlockTD(conv_num_channels, 64, kernel_size=(3, 3))
        self.block2 = ResNetBlockTD(64, 128, kernel_size=(3, 3))
        self.block3 = ResNetBlockTD(128, 256, kernel_size=(3, 3))

        # Final projection back to input channel dimension
        self.out_conv = nn.Conv2d(
            in_channels=256,
            out_channels=conv_num_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, rx_td):
        """
        rx_td: time-domain input tensor.
        Expected shape: [batch, channels, T, F]
        - channels = conv_num_channels
        Returns:
            out: same shape as rx_td (for regression with L2 loss).
        """

        x = rx_td

        # ResNet blocks
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        # Final projection to same number of channels as input
        out = self.out_conv(out)  # raw regression output, no sigmoid

        # out has shape [B, conv_num_channels, T, F], i.e., same as input (in channels)
        return out
