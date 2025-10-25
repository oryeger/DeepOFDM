import math
import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

HIDDEN_BASE_SIZE = 16


class MHSADetector(nn.Module):
    """
    OFDM / MIMO Detector using Multi-Head Self-Attention (MHSA).

    Key changes for stability:
      - Per-token (per-RE) normalization across channels before projection.
      - LayerNorm after input projection (ln_in).
      - Learnable positional scale (pos_scale) for sinusoidal PE.
      - Pre-norm Transformer block (LN -> sublayer -> residual).
      - Dropout on attention and FFN paths.
    """

    def __init__(self, num_bits, n_users, dropout=0.1, pe_base=1000.0):
        super().__init__()
        torch.manual_seed(42)

        if getattr(conf, "mhsa_no_probs", False):
            input_channels = int(conf.n_ants * 2)
        else:
            input_channels = int(num_bits * n_users + conf.n_ants * 2)

        hidden_size = HIDDEN_BASE_SIZE * num_bits
        self.embed_dim = hidden_size
        self.num_heads = 4
        self.seq_len = conf.num_res

        # Input projection (C -> D)
        self.input_proj = nn.Linear(input_channels, self.embed_dim, bias=True)

        # Normalize embeddings right after projection
        self.ln_in = nn.LayerNorm(self.embed_dim)

        # ---- Sinusoidal (fixed) positional encoding + learnable gain ----
        # Tip: use a smaller base (e.g., 1e3) for shorter OFDM sequences
        pe = self._build_sinusoidal_pos_emb(self.seq_len, self.embed_dim, base=pe_base)
        self.register_buffer("pos_emb", pe, persistent=False)  # [N, D]
        self.pos_scale = nn.Parameter(torch.tensor(1.0))       # learnable PE strength

        # Multi-Head Self-Attention (batch_first)
        self.mhsa = nn.MultiheadAttention(
            self.embed_dim, self.num_heads, batch_first=True, dropout=dropout
        )

        # Feed-forward block (use a modest expansion; GELU often works well)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        )

        # Output head (produces logits for each bit)
        self.fc_out = nn.Linear(self.embed_dim, num_bits)

        # Pre-norms for Transformer sublayers
        self.ln_attn = nn.LayerNorm(self.embed_dim)
        self.ln_ffn  = nn.LayerNorm(self.embed_dim)

        # Residual dropouts
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn  = nn.Dropout(dropout)

        # (Optional) final sigmoid for compatibility; prefer BCEWithLogitsLoss on raw logits when training.
        self.activation_final = nn.Sigmoid()

    @staticmethod
    def _build_sinusoidal_pos_emb(seq_len: int, dim: int, base: float = 1000.0) -> torch.Tensor:
        """
        Fixed sinusoidal positional encoding with exponential frequency spacing.
        'base' controls the slowest frequency span; for OFDM, smaller base (100–1000) is often better than 10000.
        """
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # [N, 1]
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(base) / dim))  # [dim/2]

        pe = torch.zeros(seq_len, dim, dtype=torch.float32)  # [N, D]
        angles = position * div_term  # [N, dim/2]
        pe[:, 0::2] = torch.sin(angles)  # even
        pe[:, 1::2] = torch.cos(angles)  # odd
        return pe

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

        # ----- Per-token normalization across channels -----
        # Move to [B, N, C] and normalize each token over channels.
        x = rx_prob.squeeze(-1).permute(0, 2, 1)  # [B, N, C]
        mu    = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True) + 1e-6
        x = (x - mu) / sigma                       # each RE ~ unit scale across channels

        # ----- Input projection + input LayerNorm + PE (with learnable gain) -----
        z = self.input_proj(x)                     # [B, N, D]
        z = self.ln_in(z)                          # keep embedding scale stable
        z = z + self.pos_scale * self.pos_emb.unsqueeze(0)  # [B, N, D]

        # ----- Pre-norm MHSA block -----
        z_norm = self.ln_attn(z)
        attn_out, _ = self.mhsa(z_norm, z_norm, z_norm)
        z = z + self.drop_attn(attn_out)

        # ----- Pre-norm FFN block -----
        z_norm = self.ln_ffn(z)
        ffn_out = self.ffn(z_norm)
        z = z + self.drop_ffn(ffn_out)

        # ----- Output head -----
        logits = self.fc_out(z)                    # [B, N, num_bits]
        probs  = self.activation_final(logits)     # keep only for inference/compat

        # reshape to [B, num_bits, N, 1] for compatibility
        llrs = logits.permute(0, 2, 1).unsqueeze(-1)
        out3 = probs.permute(0, 2, 1).unsqueeze(-1)

        return out3, llrs
