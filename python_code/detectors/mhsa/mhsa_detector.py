import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

HIDDEN_BASE_SIZE = 16


class MHSADetector(nn.Module):
    """
    OFDM / MIMO Detector using Multi-Head Self-Attention (MHSA).
    API-compatible with ESCNNDetector (same constructor and forward signature).
    """

    def __init__(self, num_bits, n_users):
        super(MHSADetector, self).__init__()
        torch.manual_seed(42)

        if conf.mhsa_no_probs:
            input_channels = int(conf.n_ants * 2)
        else:
            input_channels = int(num_bits * n_users + conf.n_ants * 2)

        hidden_size = HIDDEN_BASE_SIZE * num_bits
        self.embed_dim = hidden_size
        self.num_heads = 4  # can be tuned
        self.seq_len = conf.num_res  # number of REs / subcarriers

        # Optional input scaling
        if conf.scale_input:
            self.scale = nn.Parameter(torch.ones(input_channels * conf.num_res))

        # Input projection (2D -> D)
        self.input_proj = nn.Linear(input_channels, self.embed_dim)

        # Learnable absolute positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(self.seq_len, self.embed_dim) * 0.02)

        # Multi-Head Self-Attention (batch_first)
        self.mhsa = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        # Feed-forward block
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # Output head (produces logits for each bit)
        self.fc_out = nn.Linear(self.embed_dim, num_bits)

        # Normalizations
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)

        # Activation to produce [0,1] probabilities (for compatibility)
        self.activation_final = nn.Sigmoid()


    def forward(self, rx_prob):
        """
        rx_prob: [B, C, N, 1]
          B - batch
          C - input channels (num features)
          N - num resource elements / subcarriers
        returns:
          out3 (sigmoid outputs), llrs (raw logits)
        """

        B, C, N, _ = rx_prob.shape
        assert N == self.seq_len, f"Expected N={self.seq_len}, got {N}"

        # ---- Optional input scaling ----
        if conf.scale_input:
            rx_flat = rx_prob.reshape(B, C * N)
            rx_scaled = rx_flat * self.scale
            rx_scaled = rx_scaled.reshape(B, C, N, 1)
            x = rx_scaled
        else:
            x = rx_prob

        # reshape to [B, N, C]
        x = x.squeeze(-1).permute(0, 2, 1)

        # ---- Input projection + positional embedding ----
        z = self.input_proj(x)  # [B, N, D]
        z = z + self.pos_emb.unsqueeze(0)

        # ---- MHSA block ----
        attn_out, _ = self.mhsa(z, z, z)
        z = self.ln1(z + attn_out)
        ffn_out = self.ffn(z)
        z = self.ln2(z + ffn_out)

        # ---- Output head ----
        llrs = self.fc_out(z)          # [B, N, num_bits]
        out3 = self.activation_final(llrs)

        # reshape to [B, num_bits, N, 1] for compatibility
        llrs = llrs.permute(0, 2, 1).unsqueeze(-1)
        out3 = out3.permute(0, 2, 1).unsqueeze(-1)

        return out3, llrs
