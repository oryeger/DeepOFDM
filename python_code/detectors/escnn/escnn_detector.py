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
        no_samples = getattr(conf, 'no_samples', False)
        if conf.no_probs and no_samples:
            raise ValueError("ESCNNDetector: no_probs and no_samples are mutually exclusive")
        if conf.no_probs:
            conv_num_channels = int(conf.n_ants * 2)              # samples only
        elif no_samples:
            conv_num_channels = int(num_bits * n_users)           # LLRs only
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

        # --------- dropout (applied after each hidden ReLU) ------
        dropout_rate = getattr(conf, 'escnn_dropout', 0.0)
        self.dropout1 = nn.Dropout2d(dropout_rate) if dropout_rate > 0.0 else None
        self.dropout2 = nn.Dropout2d(dropout_rate) if dropout_rate > 0.0 else None

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
        self._load_freeze_mode = "none"  # set by set_load_freeze, re-applied by set_stage below
        freeze_mode = getattr(conf, 'escnn_load_freeze', 'none')
        loading_weights = bool(getattr(conf, 'load_escnn_weights_tag', ''))
        # Bypass only applies from scratch (no loaded weights): scale/fc2 are skipped in forward().
        # When loading weights the frozen layers contribute their loaded values as before.
        self._bypass_scale = (not loading_weights) and freeze_mode in (
            "scale", "last_conv_only", "first_conv_only", "all")
        self._bypass_fc2 = (not loading_weights) and freeze_mode in (
            "second_conv", "scale_only", "last_conv_only", "first_conv_only", "first_conv_and_scale", "all")
        self.set_load_freeze(freeze_mode)

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

        # Re-apply any active load-freeze restriction on top - otherwise a transfer-learning
        # freeze set via set_load_freeze (e.g. 'scale' or 'last_conv') would get silently
        # undone the next time training runs and set_stage() re-enables everything.
        # (FiLM is deliberately left alone here so the separate stage="film" pass still works.)
        if self._load_freeze_mode == "scale" and self.scale is not None:
            self.scale.requires_grad = False
        elif self._load_freeze_mode == "last_conv":
            for p in self.fc3.parameters():
                p.requires_grad = False
        elif self._load_freeze_mode == "first_conv":
            for p in self.fc1.parameters():
                p.requires_grad = False
        elif self._load_freeze_mode == "second_conv":
            for p in self.fc2.parameters():
                p.requires_grad = False
        elif self._load_freeze_mode == "scale_only":
            for m in [self.fc1, self.fc2, self.fc3]:
                for p in m.parameters():
                    p.requires_grad = False
        elif self._load_freeze_mode == "last_conv_only":
            for m in [self.fc1, self.fc2]:
                for p in m.parameters():
                    p.requires_grad = False
            if self.scale is not None:
                self.scale.requires_grad = False
        elif self._load_freeze_mode == "first_conv_only":
            for m in [self.fc2, self.fc3]:
                for p in m.parameters():
                    p.requires_grad = False
            if self.scale is not None:
                self.scale.requires_grad = False
        elif self._load_freeze_mode == "first_conv_and_scale":
            for m in [self.fc2, self.fc3]:
                for p in m.parameters():
                    p.requires_grad = False
            # fc1 and scale stay trainable (set_stage('base') already enabled them)
        elif self._load_freeze_mode == "all":
            for p in self.parameters():
                p.requires_grad = False

    def set_load_freeze(self, mode: str):
        """
        Controls which parameters stay trainable. Applied both at construction (from
        conf.escnn_load_freeze, so it works even without loading weights) and again
        after loading saved weights for transfer learning.

        For scale / fc2: sets bypass flags so they are skipped entirely in forward()
        rather than contributing a frozen random transformation.
        For fc1 / fc3: freezes parameters (random init when called at construction,
        loaded weights when called after loading).

          'none'               - train everything
          'scale'              - bypass scale
          'first_conv'         - freeze fc1
          'second_conv'        - bypass fc2
          'last_conv'          - freeze fc3
          'scale_only'         - bypass scale+fc2, freeze fc1+fc3
          'last_conv_only'     - bypass scale+fc2, freeze fc1  (only fc3 trains)
          'first_conv_only'    - bypass scale+fc2, freeze fc3  (only fc1 trains)
          'first_conv_and_scale' - bypass fc2, freeze fc3  (fc1 and scale train)
          'all'                - bypass scale+fc2, freeze fc1+fc3
        """
        assert mode in ["none", "scale", "first_conv", "second_conv", "last_conv",
                         "scale_only", "last_conv_only", "first_conv_only",
                         "first_conv_and_scale", "all"]
        self._load_freeze_mode = mode
        for p in self.fc1.parameters():
            p.requires_grad = mode not in ("all", "scale_only", "last_conv_only", "first_conv")
        for p in self.fc2.parameters():
            p.requires_grad = mode not in ("all", "scale_only", "last_conv_only", "second_conv", "first_conv_only", "first_conv_and_scale")
        for p in self.fc3.parameters():
            p.requires_grad = mode not in ("all", "last_conv", "scale_only", "first_conv_only", "first_conv_and_scale")
        if self.scale is not None:
            self.scale.requires_grad = mode not in ("all", "scale", "last_conv_only", "first_conv_only")
        for film in (self.film1, self.film2):
            if film is not None:
                for p in film.parameters():
                    p.requires_grad = (mode == "none")

    def forward(self, rx_prob):
        """
        rx_prob: (B, C_in, H, W)
        cond_vec:   (B, cond_dim), required if FiLM is enabled
        Returns:
            soft_estimation, llrs   (same as original code)
        """
        if conf.use_film:
            # FiLM conditioning expects the first 2*n_ants channels to be samples
            # (rx). In no_samples mode the input has no samples, so FiLM is not
            # well-defined here.
            if getattr(conf, 'no_samples', False):
                raise ValueError("ESCNNDetector: use_film is incompatible with no_samples mode")
            cond_slice = rx_prob[:, 0:2*conf.n_ants, :, :]
            cond_pooled = F.adaptive_avg_pool2d(cond_slice, 1)
            cond_vec = cond_pooled.squeeze(-1).squeeze(-1)  # Shape: (B, 16)
        else:
            cond_vec = None
        # ---------- original scaling logic ---------------
        if conf.scale_input and not self._bypass_scale:
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
        if self.dropout1 is not None:
            out1 = self.dropout1(out1)

        # Second conv + activation (bypassed when fc2 is init-frozen)
        if not self._bypass_fc2:
            out2 = self.fc2(out1)
            if conf.use_film and self.stage == "film":
                out2 = self.film2(out2, cond_vec)
            out2 = self.activation2(out2)
            if self.dropout2 is not None:
                out2 = self.dropout2(out2)
        else:
            out2 = out1

        # Final conv head (LLRs) + sigmoid
        llrs = self.fc3(out2)
        soft_estimation = self.activation3(llrs)

        # IMPORTANT: keep same return signature as original
        return soft_estimation, llrs
