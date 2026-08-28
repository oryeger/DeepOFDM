"""Extended Kalman Filter tracking a subset of an nn.Module's trainable parameters via a
soft-syndrome (LDPC parity check) measurement - unsupervised, no ground-truth bits needed.
See ekf_syndrome.tex for the state-space model this implements:

    theta_i = f(theta_{i-1}) + w_i,     Q = sigma_q^2 I
    y_i     = h_i(theta_i) + v_i,       R = sigma_r^2 I,   y_i = 1 (all checks satisfied)

f is either a random walk (f(theta) = theta) or an AR(1) pull toward the pretrained checkpoint
(f(theta) = alpha*theta + (1-alpha)*theta_pretrained). Both are affine, so F = alpha*I is a
known constant - no autodiff needed in predict(). h_i is whatever measurement_fn the caller
supplies to update() (built from SyndromeLoss.p_vector in escnn_trainer.py); its Jacobian H_i
is obtained via torch.func.jacrev, not hand-derived.

The tracked state is "whatever net.parameters() has requires_grad=True" - callers control
scope entirely through their own freeze logic (e.g. ESCNNDetector.set_load_freeze); this
module has no notion of which parameters those are or what network they belong to.
"""
from typing import Callable, Dict

import torch
from torch import nn
from torch.func import jacrev

# Dense P is O(d^2) memory/compute per update; above this, escnn_load_freeze should be
# narrowed rather than tracking the whole network.
_DENSE_STATE_WARN_DIM = 4000


class EkfParamTracker:
    def __init__(self, net: nn.Module, dynamics: str = 'ar1', alpha: float = 0.99,
                 sigma_p0: float = 0.1, sigma_q: float = 0.01, sigma_r: float = 0.5,
                 jacobian_chunk_size: int = 16):
        if dynamics not in ('random_walk', 'ar1'):
            raise ValueError(f"EkfParamTracker: dynamics must be 'random_walk' or 'ar1', got {dynamics!r}")
        self.dynamics = dynamics
        self.alpha = float(alpha) if dynamics == 'ar1' else 1.0
        self.sigma_q2 = float(sigma_q) ** 2
        self.sigma_r2 = float(sigma_r) ** 2
        # jacrev vmaps this many measurement-vector rows through the backward pass at once;
        # None = all M rows at once (fastest, but memory ~ M * activation size - OOMs on
        # large M/d combinations). Lower this if update() runs out of memory.
        self.jacobian_chunk_size = jacobian_chunk_size

        self.param_names = [n for n, p in net.named_parameters() if p.requires_grad]
        if not self.param_names:
            raise ValueError("EkfParamTracker: net has no trainable parameters - check "
                              "escnn_load_freeze (EKF needs something left unfrozen to track)")
        params_by_name = dict(net.named_parameters())
        self._params = [params_by_name[n] for n in self.param_names]
        self.shapes = [p.shape for p in self._params]
        self.numels = [p.numel() for p in self._params]
        self.d = sum(self.numels)
        self._bound_net = net

        device, dtype = self._params[0].device, self._params[0].dtype
        theta0 = torch.cat([p.detach().flatten() for p in self._params]).to(device=device, dtype=dtype)
        self.theta_pretrained = theta0.clone()
        self.theta = theta0.clone()
        self.P = (float(sigma_p0) ** 2) * torch.eye(self.d, device=device, dtype=dtype)

        if self.d > _DENSE_STATE_WARN_DIM:
            print(f"[ekf] WARNING: tracked state dim d={self.d} - dense P is O(d^2) memory/compute "
                  f"per block. Consider a narrower escnn_load_freeze scope if this is too slow.",
                  flush=True)

    def _split(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        idx = 0
        for name, shape, numel in zip(self.param_names, self.shapes, self.numels):
            out[name] = theta[idx:idx + numel].view(shape)
            idx += numel
        return out

    @torch.no_grad()
    def _write_back(self):
        for p, val in zip(self._params, self._split(self.theta).values()):
            p.copy_(val)

    def rebind(self, net: nn.Module):
        """Re-point at a freshly (re)constructed module with the same trainable-param layout
        (evaluate.py rebuilds self.detector every SNR via _initialize_detector+load_weights,
        so the old module's parameters are gone). theta_pretrained is refreshed to that new
        checkpoint - each SNR may load a different one. theta/P, the online-tracked state,
        carry over unchanged and get pushed into the new module immediately, so this SNR
        starts from where the filter left off, not from the freshly-loaded checkpoint. No-op
        if already bound to this exact module (called once per block; only actually rebinds
        on the first block of a new SNR)."""
        if net is self._bound_net:
            return
        params_by_name = dict(net.named_parameters())
        self._params = [params_by_name[n] for n in self.param_names]
        self._bound_net = net
        self.theta_pretrained = torch.cat([p.detach().flatten() for p in self._params]).to(
            device=self.theta.device, dtype=self.theta.dtype)
        self._write_back()

    def predict(self):
        """theta_{i|i-1} = f(theta_{i-1|i-1}),  P_{i|i-1} = alpha^2 P_{i-1|i-1} + Q."""
        a = self.alpha
        if a != 1.0:
            self.theta = a * self.theta + (1.0 - a) * self.theta_pretrained
            self.P.mul_(a * a)
        self.P.diagonal().add_(self.sigma_q2)

    def update(self, measurement_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor]) -> dict:
        """measurement_fn(param_dict) -> flat check-satisfaction vector p_i, differentiated
        w.r.t. param_dict's values (e.g. via torch.func.functional_call inside measurement_fn).
        Punctured-touching checks are expected to already be handled by measurement_fn
        (SyndromeLoss.p_vector excludes them in restricted mode / fills them via the
        erasure-peeling fallback), so no separate pruning happens here.

        y_i = 1 (all checks satisfied) is implicit: the innovation is 1 - p_i directly."""
        def h(theta):
            p = measurement_fn(self._split(theta))
            return p, p

        H, p_hat = jacrev(h, has_aux=True, chunk_size=self.jacobian_chunk_size)(self.theta)
        if p_hat.numel() == 0:
            return {'skipped': True}

        dy = torch.ones_like(p_hat) - p_hat
        S = H @ self.P @ H.T
        S.diagonal().add_(self.sigma_r2)
        HP = H @ self.P                       # (M, d)
        Kt = torch.linalg.solve(S, HP)        # (M, d) = S^-1 H P
        K = Kt.T                              # (d, M)

        self.theta = self.theta + K @ dy
        self.P = self.P - K @ S @ Kt
        self.P = 0.5 * (self.P + self.P.T)    # guard against float asymmetry drift
        self._write_back()

        return {'skipped': False, 'num_checks': int(p_hat.numel()),
                'mean_hard_sat': float((p_hat > 0).float().mean()),
                # Raw mean of p_hat itself (range [-1, 1], unlike mean_hard_sat's [0, 1]) -
                # keeps the confidence magnitude mean_hard_sat's >0 threshold throws away, so
                # e.g. "barely satisfied" (p~0.01) and "confidently satisfied" (p~0.99) don't
                # both just read as "satisfied".
                'mean_p': float(p_hat.mean())}
