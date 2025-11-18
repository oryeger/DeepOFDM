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

        self.num_bits = num_bits  # store for FiLM shapes
        self.n_users = n_users

        if conf.no_probs:
            conv_num_channels = int(conf.n_ants * 2)
        else:
            conv_num_channels = int(num_bits * n_users + conf.n_ants * 2)

        hidden_size = HIDDEN_BASE_SIZE * num_bits
        self.hidden_size = hidden_size  # store for FiLM

        # ----- (optional) old static input scaling -----
        # Recommended: set conf.scale_input = False and let FiLM handle adaptation
        if conf.scale_input:
            self.scale = nn.Parameter(torch.ones(conv_num_channels * conf.num_res))

        # ----- main conv path -----
        self.fc1 = nn.Conv2d(conv_num_channels, hidden_size,
                             kernel_size=(conf.kernel_size, 1),
                             padding='same')
        self.fc2 = nn.Conv2d(hidden_size, hidden_size,
                             kernel_size=(conf.kernel_size, 1),
                             padding='same')
        self.fc3 = nn.Conv2d(hidden_size, num_bits,
                             kernel_size=(conf.kernel_size, 1),
                             padding='same')

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # ----- FiLM conditioning network -----
        # We use a simple 2D conditioning vector: [mean, std] of the current input block.
        # You can always replace this with richer features later.
        self.cond_dim = 2
        cond_hidden = 64
        total_channels = hidden_size + hidden_size + num_bits  # fc1 + fc2 + fc3

        self.cond_net = nn.Sequential(
            nn.Linear(self.cond_dim, cond_hidden),
            nn.ReLU(),
            nn.Linear(cond_hidden, 2 * total_channels)  # gamma_raw_all || beta_raw_all
        )

        # Init last layer of cond_net to zeros so we start with gamma ≈ 1, beta ≈ 0
        last = self.cond_net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        # ----- single mixing coefficient λ in (0,1) for skip path -----
        # λ = sigmoid(mix_raw); init λ = conf.init_skip_weight (e.g. 0.75)
        p = torch.tensor(conf.init_skip_weight)
        self.mix_raw = nn.Parameter(torch.log(p / (1 - p)))  # scalar

    def _build_cond_vec(self, x):
        """
        Build a simple conditioning vector from the current input/features.
        Here: per-sample [mean, std] over all elements.
        x: [B, C, H, W]
        returns: [B, 2]
        """
        mean = x.mean(dim=(1, 2, 3))
        std = x.std(dim=(1, 2, 3), unbiased=False)
        cond_vec = torch.stack([mean, std], dim=-1)
        return cond_vec

    def _film_params(self, cond_vec):
        """
        From cond_vec [B, cond_dim] produce gamma/beta for
        fc1, fc2, fc3 outputs (channel-wise).
        Returns:
          (gamma1, beta1), (gamma2, beta2), (gamma3, beta3)
          each with shape [B, C_l, 1, 1]
        """
        B = cond_vec.size(0)
        total_channels = self.hidden_size + self.hidden_size + self.num_bits

        gb_all = self.cond_net(cond_vec)            # [B, 2 * total_channels]
        gamma_raw_all, beta_raw_all = gb_all.chunk(2, dim=-1)

        # Start around gamma=1, beta=0
        gamma_all = 1.0 + gamma_raw_all
        beta_all = beta_raw_all

        # Split per layer
        g1, g2, g3 = torch.split(
            gamma_all,
            [self.hidden_size, self.hidden_size, self.num_bits],
            dim=-1
        )
        b1, b2, b3 = torch.split(
            beta_all,
            [self.hidden_size, self.hidden_size, self.num_bits],
            dim=-1
        )

        # Reshape to [B, C, 1, 1] for broadcasting over H,W
        g1 = g1.view(B, self.hidden_size, 1, 1)
        g2 = g2.view(B, self.hidden_size, 1, 1)
        g3 = g3.view(B, self.num_bits, 1, 1)
        b1 = b1.view(B, self.hidden_size, 1, 1)
        b2 = b2.view(B, self.hidden_size, 1, 1)
        b3 = b3.view(B, self.num_bits, 1, 1)

        return (g1, b1), (g2, b2), (g3, b3)

    def forward(self, rx_prob, num_bits, user):
        # Optional static input scaling (legacy)
        if conf.scale_input:
            rx = rx_prob.clone()
            if conf.which_augment == 'AUGMENT_LMMSE':
                rx[:, conf.n_ants * 2:, :, :] = torch.sigmoid(
                    rx_prob[:, conf.n_ants * 2:, :, :]
                )
            rx_flat = rx.reshape(rx.shape[0], rx.shape[1] * rx.shape[2], 1).squeeze(-1)
            rx_out_flat = rx_flat * self.scale
            x = rx_out_flat.unsqueeze(-1).reshape_as(rx)
        else:
            x = rx_prob

        # ---- Build FiLM conditioning parameters from current input ----
        cond_vec = self._build_cond_vec(x)  # [B, 2]
        (g1, b1), (g2, b2), (g3, b3) = self._film_params(cond_vec)

        # ---- main path with FiLM after each conv ----
        h1_lin = self.fc1(x)                 # [B, hidden_size, H, W]
        h1_film = g1 * h1_lin + b1
        h1 = self.act1(h1_film)

        h2_lin = self.fc2(h1)                # [B, hidden_size, H, W]
        h2_film = g2 * h2_lin + b2
        h2 = self.act2(h2_film)

        llrs_lin = self.fc3(h2)              # [B, num_bits, H, W]
        llrs_main = g3 * llrs_lin + b3       # FiLM on final layer too

        # ---- skip path -> LLRs directly from input ----
        start = conf.n_ants * 2 + user * self.num_bits
        end = conf.n_ants * 2 + (user + 1) * self.num_bits
        llrs_skip = rx_prob[:, start:end, :, :]  # [B, num_bits, H, W]

        # ---- single scalar mix λ in (0,1) ----
        lam = torch.sigmoid(self.mix_raw)  # scalar

        if conf.init_skip_weight != 0:
            llrs_total = lam * llrs_skip + (1.0 - lam) * llrs_main
        else:
            llrs_total = llrs_main

        probs = self.sigmoid(llrs_total)
        return probs, llrs_total
