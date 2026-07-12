"""Soft LDPC syndrome loss (L_synd) for the blind 'tsyn' training mode.

Maps detector-output LLRs (decoder-input stream order, length n per codeword)
into 5G LDPC mother-codeword coordinates (length n_ldpc) and penalises
unsatisfied parity checks softly:

    t_i    = tanh(clamp(L_i, -30, 30) / 2)          (classical-convention LLRs)
    p_j    = prod_{i in N_j} t_i                     (N_j = support of H row j)
    L_synd = -(1/|J|) * sum_j log(clamp((1+p_j)/2, 1e-9, 1.0))

The product is never formed directly: per check we accumulate sum(log|t_i|)
plus a sign product in the log domain (index_add), which also yields the
exact analytic gradient d p_j / d t_i = prod_{others}.

Sign convention: this project (and Sionna's decoder) uses the logit
convention L = log(p(b=1)/p(b=0)), i.e. L>0 => bit 1. The formulas above
assume the classical convention L = log(p0/p1) (L>0 => bit 0, t=+1), so
map_to_mother() negates the incoming LLRs. Filler bits are known zeros and
get +filler_llr in the classical domain (t ~ +1, neutral factor).

Rate matching (rv=0) mirrors Sionna's LDPC5GDecoder.call() expansion exactly:
transmitted LLR i sits at pre-filler position 2Z+i; after the filler block
[k, k+k_filler) is inserted, positions >= k shift up by k_filler. The first
2Z systematic positions and the unused circular-buffer tail are punctured
(never transmitted, no LLR).

Punctured positions would contribute t=0 and silently zero out every check
they touch, so by default the loss is restricted to checks whose support
contains no punctured position (this covers both the first-2Z puncturing and
the tail truncation). Optionally (punctured_fallback=True) the punctured
bits' soft values are estimated with a single variable-node update from
their neighbouring checks — detached from autograd — and the loss then runs
over all checks.

TODO: a min-sum variant of the check update would be the fixed-point-friendly
kernel; not implemented for now.
"""
import numpy as np
import torch


def _import_encoder():
    # The cluster env has Sionna 0.x (sionna.fec...); local installs may have
    # Sionna >= 1.0 where FEC moved under sionna.phy.
    try:
        from sionna.fec.ldpc.encoding import LDPC5GEncoder
    except ImportError:
        from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
    return LDPC5GEncoder


class SyndromeLoss:
    LLR_CLAMP = 30.0

    def __init__(self, k: int, n: int, device='cpu', punctured_fallback: bool = False,
                 filler_llr: float = 30.0):
        """
        k : information bits of the outer code (ldpc_k + crc_length, exactly as
            LDPC5GCodec is constructed in evaluate.py / mimo_channel_dataset).
        n : transmitted codeword bits (ldpc_n).
        """
        LDPC5GEncoder = _import_encoder()
        enc = LDPC5GEncoder(k, n)
        if enc.num_bits_per_symbol is not None:
            raise ValueError("SyndromeLoss assumes no rate-matching output interleaver "
                             "(LDPC5GEncoder built without num_bits_per_symbol).")
        self.n = int(enc.n)
        self.k = int(enc.k)
        self.z = int(enc.z)
        self.k_ldpc = int(enc.k_ldpc)
        self.n_ldpc = int(enc.n_ldpc)
        self.k_filler = self.k_ldpc - self.k
        self.filler_llr = float(filler_llr)
        self.punctured_fallback = bool(punctured_fallback)

        # Inverse rate matching (rv=0), mirroring LDPC5GDecoder.call():
        # transmitted position i -> pre-filler 2Z+i -> +k_filler shift once past k.
        pre = np.arange(self.n) + 2 * self.z
        if pre[-1] >= self.n_ldpc - self.k_filler:
            raise ValueError(f"n={self.n} exceeds the rv=0 circular buffer "
                             f"(n_ldpc={self.n_ldpc}, k_filler={self.k_filler}).")
        tx_to_mother = np.where(pre < self.k, pre, pre + self.k_filler)
        filler_idx = np.arange(self.k_filler) + self.k

        filled = np.zeros(self.n_ldpc, dtype=bool)
        filled[tx_to_mother] = True
        filled[filler_idx] = True
        self._punctured_np = np.flatnonzero(~filled)

        # Parity-check matrix -> edge list (check_idx, bit_idx)
        pcm = enc.pcm
        if hasattr(pcm, 'tocoo'):
            coo = pcm.tocoo()
            rows, cols = np.asarray(coo.row), np.asarray(coo.col)
        else:
            rows, cols = np.nonzero(np.asarray(pcm))
        self.num_checks = int(pcm.shape[0])

        # Usable checks: support entirely inside filled (transmitted or filler)
        # positions. This excludes checks touching the punctured first 2Z
        # systematic columns AND checks touching the untransmitted tail.
        check_touches_punctured = np.zeros(self.num_checks, dtype=bool)
        np.logical_or.at(check_touches_punctured, rows, ~filled[cols])
        usable = ~check_touches_punctured
        self.num_usable = int(usable.sum())
        frac = self.num_usable / max(self.num_checks, 1)
        print(f"[tsyn] syndrome loss: {self.num_usable}/{self.num_checks} checks usable "
              f"({100.0 * frac:.1f}%)  [n={self.n}, n_ldpc={self.n_ldpc}, Z={self.z}, "
              f"k={self.k}, fillers={self.k_filler}, punctured={self._punctured_np.size}]",
              flush=True)
        if self.num_usable == 0 and not self.punctured_fallback:
            # Restricted mode is vacuous (empty mean -> NaN). Short codes like the
            # project default (n=112) leave no check untouched by puncturing.
            print("[tsyn] no usable checks in restricted mode -> auto-enabling the "
                  "punctured-bit fallback (single detached VN update, all checks).",
                  flush=True)
            self.punctured_fallback = True
        elif frac < 0.20 and not self.punctured_fallback:
            print("[tsyn] WARNING: fewer than 20% of checks are usable; consider "
                  "tsyn_punctured_fallback: True", flush=True)

        keep = usable[rows]
        rows_u, cols_u = rows[keep], cols[keep]
        remap = np.full(self.num_checks, -1, dtype=np.int64)
        remap[np.flatnonzero(usable)] = np.arange(self.num_usable)

        dev = torch.device(device)
        self.device = dev
        self.tx_to_mother = torch.as_tensor(tx_to_mother, dtype=torch.long, device=dev)
        self.filler_idx = torch.as_tensor(filler_idx, dtype=torch.long, device=dev)
        self.edge_check_u = torch.as_tensor(remap[rows_u], dtype=torch.long, device=dev)
        self.edge_bit_u = torch.as_tensor(cols_u, dtype=torch.long, device=dev)
        # Full edge set (all checks) for the punctured fallback path
        self.edge_check_all = torch.as_tensor(rows, dtype=torch.long, device=dev)
        self.edge_bit_all = torch.as_tensor(cols, dtype=torch.long, device=dev)
        self.punctured_idx = torch.as_tensor(self._punctured_np, dtype=torch.long, device=dev)

    def _to(self, device):
        if self.device != device:
            for name in ('tx_to_mother', 'filler_idx', 'edge_check_u', 'edge_bit_u',
                         'edge_check_all', 'edge_bit_all', 'punctured_idx'):
                setattr(self, name, getattr(self, name).to(device))
            self.device = device

    def map_to_mother(self, llr_tx: torch.Tensor) -> torch.Tensor:
        """(B, n) project-convention LLRs -> (B, n_ldpc) classical-convention.

        Negation converts logit convention (L>0 => bit 1) to the classical
        convention the loss formulas assume. Fillers (known zeros) get
        +filler_llr; punctured positions stay 0.
        """
        self._to(llr_tx.device)
        batch = llr_tx.shape[0]
        mother = torch.zeros(batch, self.n_ldpc, dtype=llr_tx.dtype, device=llr_tx.device)
        mother[:, self.tx_to_mother] = -llr_tx
        mother[:, self.filler_idx] = self.filler_llr
        return mother

    @staticmethod
    def _check_products(t: torch.Tensor, edge_check: torch.Tensor, edge_bit: torch.Tensor,
                        num_checks: int) -> torch.Tensor:
        """p_j = prod_{i in N_j} t_i via log-domain accumulation (exact gradients)."""
        batch = t.shape[0]
        te = t[:, edge_bit]                                        # (B, E)
        log_abs = torch.log(te.abs().clamp(min=1e-30))
        sum_log = torch.zeros(batch, num_checks, dtype=t.dtype, device=t.device)
        sum_log.index_add_(1, edge_check, log_abs)
        neg = torch.zeros(batch, num_checks, dtype=t.dtype, device=t.device)
        neg.index_add_(1, edge_check, (te < 0).to(t.dtype))
        sign = 1.0 - 2.0 * torch.remainder(neg, 2.0)
        return sign * torch.exp(sum_log)

    @torch.no_grad()
    def _estimate_punctured_t(self, t: torch.Tensor) -> torch.Tensor:
        """Single detached variable-node update for the punctured bits.

        For punctured bit v: L_v = sum_{j in M(v)} 2*artanh(prod_{i in N_j\\v} t_i).
        A check with more than one zero factor contributes a 0 message.
        Returns t with punctured positions replaced by tanh(L_v/2).
        """
        batch = t.shape[0]
        te = t[:, self.edge_bit_all]                               # (B, E)
        is_zero = (te.abs() < 1e-12)
        zero_count = torch.zeros(batch, self.num_checks, device=t.device, dtype=t.dtype)
        zero_count.index_add_(1, self.edge_check_all, is_zero.to(t.dtype))
        te_nz = torch.where(is_zero, torch.ones_like(te), te)
        log_abs = torch.log(te_nz.abs().clamp(min=1e-30))
        sum_log = torch.zeros(batch, self.num_checks, device=t.device, dtype=t.dtype)
        sum_log.index_add_(1, self.edge_check_all, log_abs)
        neg = torch.zeros(batch, self.num_checks, device=t.device, dtype=t.dtype)
        neg.index_add_(1, self.edge_check_all, (te_nz < 0).to(t.dtype))
        nz_prod = (1.0 - 2.0 * torch.remainder(neg, 2.0)) * torch.exp(sum_log)

        # Edges whose bit is punctured: message = nonzero-product of the check
        # if that bit is the check's ONLY zero factor, else 0.
        punc_mask = torch.zeros(self.n_ldpc, dtype=torch.bool, device=t.device)
        punc_mask[self.punctured_idx] = True
        edge_is_punc = punc_mask[self.edge_bit_all]
        e_check = self.edge_check_all[edge_is_punc]
        e_bit = self.edge_bit_all[edge_is_punc]
        msg = nz_prod[:, e_check] * (zero_count[:, e_check] == 1.0).to(t.dtype)
        msg = torch.atanh(msg.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)) * 2.0

        llr_est = torch.zeros(batch, self.n_ldpc, device=t.device, dtype=t.dtype)
        llr_est.index_add_(1, e_bit, msg)
        t_out = t.clone()
        t_out[:, self.punctured_idx] = torch.tanh(
            llr_est[:, self.punctured_idx].clamp(-self.LLR_CLAMP, self.LLR_CLAMP) / 2.0)
        return t_out

    def loss(self, llr_tx: torch.Tensor) -> torch.Tensor:
        """L_synd for a batch of transmitted-LLR codewords, shape (B, n)."""
        mother = self.map_to_mother(llr_tx)
        t = torch.tanh(mother.clamp(-self.LLR_CLAMP, self.LLR_CLAMP) / 2.0)
        if self.punctured_fallback:
            t_punc = self._estimate_punctured_t(t.detach())
            t = t.clone()
            t[:, self.punctured_idx] = t_punc[:, self.punctured_idx]
            p = self._check_products(t, self.edge_check_all, self.edge_bit_all, self.num_checks)
        else:
            p = self._check_products(t, self.edge_check_u, self.edge_bit_u, self.num_usable)
        return -torch.log(((1.0 + p) / 2.0).clamp(min=1e-9, max=1.0)).mean()

    @torch.no_grad()
    def hard_satisfaction(self, llr_tx: torch.Tensor) -> float:
        """Fraction of checks satisfied by the hard decisions (blind health metric).

        Restricted mode: usable checks only. Fallback mode: all checks, with
        punctured bits filled by the single VN-update estimate (undetermined
        punctured bits count as 0, so the metric is noisier there)."""
        mother = self.map_to_mother(llr_tx)
        t = torch.tanh(mother.clamp(-self.LLR_CLAMP, self.LLR_CLAMP) / 2.0)
        if self.punctured_fallback:
            t = self._estimate_punctured_t(t)
            bits = (t < 0).to(torch.long)    # classical: negative => bit 1
            edge_check, edge_bit, num_checks = self.edge_check_all, self.edge_bit_all, self.num_checks
        else:
            bits = (mother < 0).to(torch.long)
            edge_check, edge_bit, num_checks = self.edge_check_u, self.edge_bit_u, self.num_usable
        batch = bits.shape[0]
        par = torch.zeros(batch, num_checks, dtype=torch.long, device=bits.device)
        par.index_add_(1, edge_check, bits[:, edge_bit])
        return (par % 2 == 0).float().mean().item()
