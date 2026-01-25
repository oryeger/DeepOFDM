import torch
import torch.nn as nn
import torch.nn.functional as F

from python_code import conf

HIDDEN_BASE_SIZE = 16


class JointLLRDetector(nn.Module):
    """
    Joint MIMO detector with:
    - ICI handling via frequency-domain CNN
    - All users processed jointly (no sequential loops)
    - LLR/probability feedback for iterative refinement or primary detector augmentation
    - Channel H conditioning via FiLM layers

    Can be used in multiple modes:
    1. Standalone: priors initialized to 0 (LLR=0 means prob=0.5)
    2. Augmented: priors from LMMSE/Sphere/other primary detector
    3. Iterative: output fed back as priors for multiple passes
    """
    def __init__(self, num_bits, n_users, n_ants):
        super(JointLLRDetector, self).__init__()
        self.n_ants = n_ants
        self.n_users = n_users
        self.num_bits = num_bits

        # Dimensions
        y_dim = 2 * n_ants                    # real/imag received signal
        h_dim = 2 * n_ants * n_users          # flattened channel matrix (real/imag)
        prior_dim = n_users * num_bits        # LLRs for all users

        # Hidden size - scale with num_bits like ESCNN
        hidden_size = HIDDEN_BASE_SIZE * num_bits
        self.hidden = hidden_size

        # === Stage 1: Input fusion (y + priors, conditioned on H) ===
        input_dim = y_dim + prior_dim         # received signal + prior LLRs

        self.fc_in = nn.Linear(input_dim, hidden_size)
        self.film1_gamma = nn.Linear(h_dim, hidden_size)
        self.film1_beta = nn.Linear(h_dim, hidden_size)

        self.fc_mid = nn.Linear(hidden_size, hidden_size)
        self.film2_gamma = nn.Linear(h_dim, hidden_size)
        self.film2_beta = nn.Linear(h_dim, hidden_size)

        # === Stage 2: ICI-aware CNN along frequency axis ===
        ici_kernel = conf.kernel_size
        n_ici_layers = 3

        ici_layers = []
        for i in range(n_ici_layers):
            ici_layers.extend([
                nn.Conv1d(hidden_size, hidden_size, kernel_size=ici_kernel,
                         padding=ici_kernel//2),
                nn.ReLU(),
            ])
        self.ici_cnn = nn.Sequential(*ici_layers)

        # === Stage 3: Joint output for all users ===
        self.fc_out1 = nn.Linear(hidden_size, hidden_size)
        self.fc_out2 = nn.Linear(hidden_size, prior_dim)  # LLRs for all users

    def forward(self, y, H, prior_llrs=None):
        """
        Args:
            y: (batch, n_symbols, 2*n_ants) - received signal (real/imag interleaved)
            H: (batch, n_symbols, 2*n_ants*n_users) - flattened channel (real/imag)
            prior_llrs: (batch, n_symbols, n_users*num_bits) - prior LLRs
                        If None, initialized to 0 (equivalent to prob=0.5)

        Returns:
            llrs: (batch, n_symbols, n_users*num_bits) - refined LLRs
            probs: (batch, n_symbols, n_users*num_bits) - probabilities
        """
        batch, n_sym, _ = y.shape
        device = y.device

        # Initialize priors to 0 (prob=0.5) if not provided
        if prior_llrs is None:
            prior_llrs = torch.zeros(batch, n_sym, self.n_users * self.num_bits,
                                     device=device)

        # === Stage 1: Fuse y + priors, conditioned on H ===
        # Flatten batch and symbol dimensions for parallel processing
        y_flat = y.reshape(batch * n_sym, -1)
        H_flat = H.reshape(batch * n_sym, -1)
        prior_flat = prior_llrs.reshape(batch * n_sym, -1)

        # Concatenate received signal with prior LLRs
        x = torch.cat([y_flat, prior_flat], dim=-1)

        # FiLM-conditioned layer 1
        gamma1 = self.film1_gamma(H_flat)
        beta1 = self.film1_beta(H_flat)
        x = F.relu(gamma1 * self.fc_in(x) + beta1)

        # FiLM-conditioned layer 2
        gamma2 = self.film2_gamma(H_flat)
        beta2 = self.film2_beta(H_flat)
        x = F.relu(gamma2 * self.fc_mid(x) + beta2)

        # Reshape for CNN: (batch, hidden, n_symbols)
        x = x.reshape(batch, n_sym, self.hidden).permute(0, 2, 1)

        # === Stage 2: ICI CNN ===
        x = self.ici_cnn(x)

        # === Stage 3: Joint output ===
        x = x.permute(0, 2, 1)  # (batch, n_sym, hidden)
        x = F.relu(self.fc_out1(x))
        llrs = self.fc_out2(x)  # (batch, n_sym, n_users*num_bits)

        probs = torch.sigmoid(llrs)

        return probs, llrs
