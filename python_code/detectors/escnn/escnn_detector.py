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

        # Gamma → start at 1  (identity scaling)
        nn.init.constant_(self.gamma.weight, 0.0)
        nn.init.constant_(self.gamma.bias, 1.0)

        # Beta → start at ε = 1e-4  (small offset)
        nn.init.constant_(self.beta.weight, 0.0)
        nn.init.constant_(self.beta.bias, 1e-4)

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

        self.stage = "base"  # default

    def set_stage(self, stage: str):
        assert stage in ["base", "film", "scale_only"]
        self.stage = stage

        # backbone params: convs + scale
        backbone_params = []
        for m in [self.fc1, self.fc2, self.fc3]:
            backbone_params += list(m.parameters())
        if self.scale is not None:
            backbone_params.append(self.scale)

        # FiLM params
        film_params = []
        if self.film1 is not None:
            film_params += list(self.film1.parameters())
        if self.film2 is not None:
            film_params += list(self.film2.parameters())

        if stage == "base":
            # train backbone, freeze FiLM
            for p in backbone_params:
                p.requires_grad = True
            for p in film_params:
                p.requires_grad = False

        elif stage == "film":
            # freeze backbone, train FiLM
            for p in backbone_params:
                p.requires_grad = False
            for p in film_params:
                p.requires_grad = True

        elif stage == "scale_only":
            # freeze CNN (fc1, fc2, fc3), train only scale
            for m in [self.fc1, self.fc2, self.fc3]:
                for p in m.parameters():
                    p.requires_grad = False
            if self.scale is not None:
                self.scale.requires_grad = True
            for p in film_params:
                p.requires_grad = False

    def get_cnn_state_dict(self):
        """Extract only CNN layer parameters (fc1, fc2, fc3), excluding scale."""
        cnn_state = {}
        for name, param in self.state_dict().items():
            if name.startswith(('fc1.', 'fc2.', 'fc3.')):
                cnn_state[name] = param.clone()
        return cnn_state

    def load_cnn_from(self, source):
        """Load CNN parameters from another model or state dict, keeping scale untouched.

        Args:
            source: ESCNNDetector instance or state dict containing CNN params
        """
        if isinstance(source, ESCNNDetector):
            source_state = source.get_cnn_state_dict()
        elif isinstance(source, dict):
            source_state = {k: v.clone() for k, v in source.items()
                           if k.startswith(('fc1.', 'fc2.', 'fc3.'))}
        else:
            raise TypeError("source must be ESCNNDetector or state dict")

        # Load only the CNN parameters
        current_state = self.state_dict()
        current_state.update(source_state)
        self.load_state_dict(current_state)

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

            cond_slice = rx_prob[:, 0:2*conf.n_ants, :, :]
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

        # FiLM after first conv
        if conf.use_film and self.stage == "film":
            out1 = self.film1(out1, cond_vec)

        out1 = self.activation1(out1)

        # Second conv + activation
        out2 = self.fc2(out1)

        # FiLM after second conv
        if conf.use_film and self.stage == "film":
            out2 = self.film2(out2, cond_vec)

        out2 = self.activation2(out2)

        # Final conv head (LLRs) + sigmoid
        llrs = self.fc3(out2)
        soft_estimation = self.activation3(llrs)

        # IMPORTANT: keep same return signature as original
        return soft_estimation, llrs
