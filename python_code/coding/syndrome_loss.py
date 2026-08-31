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
the tail truncation). Optionally (fallback_iters > 0) the punctured bits'
soft values are estimated with an iterated, detached erasure-peeling
procedure over their neighbouring checks, and the loss then runs over all
checks. fallback_iters=1 is a single variable-node update (a check
contributes a message to a punctured neighbour only if that neighbour is
its one remaining unresolved factor); fallback_iters>1 repeats the update,
so a bit resolved in one round can unlock checks for its neighbours in the
next ("peeling"). The loop stops early at a fixpoint (a round that resolves
nothing further) even if fallback_iters hasn't been reached.

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

    def __init__(self, k: int, n: int, device='cpu', fallback_iters: int = 0,
                 filler_llr: float = 30.0, tag: str = 'synd'):
        """
        k : information bits of the outer code (ldpc_k + crc_length, exactly as
            LDPC5GCodec is constructed in evaluate.py / mimo_channel_dataset).
        n : transmitted codeword bits (ldpc_n).
        fallback_iters : 0 = restricted mode (no fallback); 1 = single detached
            variable-node update (previous punctured_fallback=True behaviour);
            >1 = iterated erasure-peeling over that many rounds (early-stops at
            a fixpoint).
        tag : log-line prefix identifying which caller this instance serves
            (e.g. 'tsyn' for the training loss, 'ekf' for EkfParamTracker) - purely
            cosmetic, does not affect the computation.
        """
        self.tag = tag
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
        self.fallback_iters = int(fallback_iters)
        self._fallback_rounds_logged = False

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
        print(f"[{self.tag}] syndrome: {self.num_usable}/{self.num_checks} checks usable "
              f"({100.0 * frac:.1f}%)  [n={self.n}, n_ldpc={self.n_ldpc}, Z={self.z}, "
              f"k={self.k}, fillers={self.k_filler}, punctured={self._punctured_np.size}]",
              flush=True)
        if self.num_usable == 0 and self.fallback_iters <= 0:
            # Restricted mode is vacuous (empty mean -> NaN). Short codes like the
            # project default (n=112) leave no check untouched by puncturing.
            print(f"[{self.tag}] no usable checks in restricted mode -> auto-enabling the "
                  "punctured-bit fallback (1 round; raise tsyn_fallback_iters for "
                  "more erasure-peeling rounds).", flush=True)
            self.fallback_iters = 1
        elif frac < 0.20 and self.fallback_iters <= 0:
            print(f"[{self.tag}] WARNING: fewer than 20% of checks are usable; consider "
                  "tsyn_fallback_iters: 1 (or higher)", flush=True)

        keep = usable[rows]
        rows_u, cols_u = rows[keep], cols[keep]
        remap = np.full(self.num_checks, -1, dtype=np.int64)
        remap[np.flatnonzero(usable)] = np.arange(self.num_usable)

        # Non-circular check set for fallback mode: a check that itself supplied a
        # peeling message to one of its own punctured neighbours (self.fallback_iters
        # rounds) would then be "verified" against a value it helped fabricate - for a
        # check with exactly one punctured neighbour resolved solely by itself this is
        # provably vacuous (p_j = (prod of its other members)^2 >= 0 always, rewarding
        # LLR magnitude regardless of sign/correctness). This is purely structural
        # (which positions are punctured, and the peeling round budget) - it doesn't
        # depend on the actual LLR values, since punctured positions always start at
        # exactly t=0 - so it's precomputed once here, mirroring the per-round
        # zero_count==1 logic in _estimate_punctured_t exactly (synchronous rounds:
        # each round's contributions are decided from the state at the round's start,
        # applied afterwards, stopping early at a fixpoint).
        check_to_cols = {}
        for r_, c_ in zip(rows.tolist(), cols.tolist()):
            check_to_cols.setdefault(r_, []).append(c_)
        resolved = filled.copy()
        contributing_checks = np.zeros(self.num_checks, dtype=bool)
        for _ in range(max(int(self.fallback_iters), 0)):
            newly = []
            for j, members in check_to_cols.items():
                unresolved = [m for m in members if not resolved[m]]
                if len(unresolved) == 1:
                    newly.append(unresolved[0])
                    contributing_checks[j] = True
            if not newly:
                break
            resolved[newly] = True
        all_resolved = np.array([all(resolved[m] for m in check_to_cols.get(j, []))
                                  for j in range(self.num_checks)], dtype=bool)
        clean = all_resolved & ~contributing_checks   # includes usable checks trivially
        self.num_clean = int(clean.sum())
        if self.fallback_iters > 0:
            n_dead = self.num_checks - int(all_resolved.sum())
            n_self = int(contributing_checks.sum())
            print(f"[{self.tag}] fallback check quality: {self.num_clean}/{self.num_checks} clean "
                  f"(non-circular), {n_self} self-referential (excluded - would be tested against "
                  f"a value they helped fabricate), {n_dead} dead (an unresolved punctured "
                  f"neighbour, excluded)", flush=True)

        keep_c = clean[rows]
        rows_c, cols_c = rows[keep_c], cols[keep_c]
        remap_c = np.full(self.num_checks, -1, dtype=np.int64)
        remap_c[np.flatnonzero(clean)] = np.arange(self.num_clean)

        dev = torch.device(device)
        self.device = dev
        self.tx_to_mother = torch.as_tensor(tx_to_mother, dtype=torch.long, device=dev)
        self.filler_idx = torch.as_tensor(filler_idx, dtype=torch.long, device=dev)
        self.edge_check_u = torch.as_tensor(remap[rows_u], dtype=torch.long, device=dev)
        self.edge_bit_u = torch.as_tensor(cols_u, dtype=torch.long, device=dev)
        # Full edge set (all checks) - used only inside _estimate_punctured_t, which
        # legitimately needs every check touching a punctured bit to estimate it.
        self.edge_check_all = torch.as_tensor(rows, dtype=torch.long, device=dev)
        self.edge_bit_all = torch.as_tensor(cols, dtype=torch.long, device=dev)
        # Clean edge set (all checks) - used for the actual loss/satisfaction output in
        # fallback mode, so a check is never scored against a punctured neighbour it (or
        # a round of peeling seeded by it) helped estimate.
        self.edge_check_clean = torch.as_tensor(remap_c[rows_c], dtype=torch.long, device=dev)
        self.edge_bit_clean = torch.as_tensor(cols_c, dtype=torch.long, device=dev)
        self.punctured_idx = torch.as_tensor(self._punctured_np, dtype=torch.long, device=dev)

    def _to(self, device):
        if self.device != device:
            for name in ('tx_to_mother', 'filler_idx', 'edge_check_u', 'edge_bit_u',
                         'edge_check_all', 'edge_bit_all', 'edge_check_clean',
                         'edge_bit_clean', 'punctured_idx'):
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
    def _estimate_punctured_t(self, t: torch.Tensor, fallback_iters: int) -> torch.Tensor:
        """Iterated, detached erasure-peeling for the punctured bits.

        For punctured bit v: L_v = sum_{j in M(v)} 2*artanh(prod_{i in N_j\\v} t_i).
        A check with more than one still-unresolved (zero) neighbour contributes
        a 0 message. Each round recomputes zero_count/products from the current
        t, so a bit resolved in round r keeps its fixed value and acts as a
        normal nonzero factor for every other check it touches in round r+1 --
        resolving one bit can unlock a check for its neighbours on the next
        round. Already-resolved positions are never recomputed or overwritten
        (only positions still exactly 0 at the start of a round are candidates),
        so more rounds can only fill in more positions, never change or unfill
        one. Stops early once a round resolves nothing further.

        Returns t with punctured positions replaced by tanh(L_v/2) (unresolved
        positions, if any remain, stay at their input value).
        """
        punc_mask = torch.zeros(self.n_ldpc, dtype=torch.bool, device=t.device)
        punc_mask[self.punctured_idx] = True
        edge_punc_struct = punc_mask[self.edge_bit_all]            # (E,) fixed

        batch = t.shape[0]
        t_out = t.clone()
        rounds_run = 0
        n_rounds = max(int(fallback_iters), 0)
        for _ in range(n_rounds):
            rounds_run += 1
            still_zero_before = t_out[:, self.punctured_idx].abs() < 1e-12   # (B, P)

            te = t_out[:, self.edge_bit_all]                       # (B, E)
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

            # Candidate edges: structurally punctured AND still zero for this
            # batch element right now. This is a per-batch, per-round gate
            # (not the fixed structural mask alone), since which bits remain
            # unresolved diverges across the batch once round > 1.
            msg = nz_prod[:, self.edge_check_all] * (zero_count[:, self.edge_check_all] == 1.0).to(t.dtype)
            msg = torch.atanh(msg.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)) * 2.0
            gate = (is_zero & edge_punc_struct.unsqueeze(0)).to(t.dtype)
            msg = msg * gate

            llr_est = torch.zeros(batch, self.n_ldpc, device=t.device, dtype=t.dtype)
            llr_est.index_add_(1, self.edge_bit_all, msg)
            candidate = torch.tanh(
                llr_est[:, self.punctured_idx].clamp(-self.LLR_CLAMP, self.LLR_CLAMP) / 2.0)

            t_out = t_out.clone()
            t_out[:, self.punctured_idx] = torch.where(still_zero_before, candidate,
                                                        t_out[:, self.punctured_idx])

            still_zero_after = t_out[:, self.punctured_idx].abs() < 1e-12
            if torch.equal(still_zero_after, still_zero_before):
                break

        if not self._fallback_rounds_logged:
            status = 'fixpoint' if rounds_run < n_rounds else 'round budget exhausted'
            print(f"[{self.tag}] fallback: erasure-peeling ran {rounds_run}/{n_rounds} "
                  f"round(s) ({status})", flush=True)
            self._fallback_rounds_logged = True
        return t_out

    def p_vector(self, llr_tx: torch.Tensor) -> torch.Tensor:
        """Soft check-satisfaction p_j = prod_{i in N_j} t_i for a batch of transmitted-LLR
        codewords, shape (B, n) -> (B, num_usable) [or (B, num_clean) in fallback mode, with
        punctured positions filled by the erasure-peeling estimate]. Checks touching a
        punctured bit are already excluded in restricted mode (num_usable), so callers never
        see an identically-zero row from puncturing. Fallback mode further excludes any check
        that itself contributed to resolving one of its own punctured neighbours (self-
        referential - see the "clean" set built in __init__) and any check with a neighbour
        that never resolved (dead - always p_j=0) - only "clean" checks are scored.

        Shared by loss() (training-time L_synd) and the EKF measurement model in
        ekf_tracker.py, so both use exactly one implementation of what the syndrome
        measurement is."""
        mother = self.map_to_mother(llr_tx)
        t = torch.tanh(mother.clamp(-self.LLR_CLAMP, self.LLR_CLAMP) / 2.0)
        if self.fallback_iters > 0:
            t_punc = self._estimate_punctured_t(t.detach(), self.fallback_iters)
            t = t.clone()
            t[:, self.punctured_idx] = t_punc[:, self.punctured_idx]
            return self._check_products(t, self.edge_check_clean, self.edge_bit_clean, self.num_clean)
        return self._check_products(t, self.edge_check_u, self.edge_bit_u, self.num_usable)

    def loss(self, llr_tx: torch.Tensor) -> torch.Tensor:
        """L_synd for a batch of transmitted-LLR codewords, shape (B, n)."""
        p = self.p_vector(llr_tx)
        return -torch.log(((1.0 + p) / 2.0).clamp(min=1e-9, max=1.0)).mean()

    @torch.no_grad()
    def hard_satisfaction(self, llr_tx: torch.Tensor) -> float:
        """Fraction of checks satisfied by the hard decisions (blind health metric).

        Restricted mode: usable checks only. Fallback mode: the "clean" checks only
        (punctured bits filled by the erasure-peeling estimate, same as p_vector) -
        self-referential and dead checks are excluded here too, so this diagnostic
        isn't inflated by checks that are trivially self-consistent or permanently
        constant."""
        mother = self.map_to_mother(llr_tx)
        t = torch.tanh(mother.clamp(-self.LLR_CLAMP, self.LLR_CLAMP) / 2.0)
        if self.fallback_iters > 0:
            t = self._estimate_punctured_t(t, self.fallback_iters)
            bits = (t < 0).to(torch.long)    # classical: negative => bit 1
            edge_check, edge_bit, num_checks = self.edge_check_clean, self.edge_bit_clean, self.num_clean
        else:
            bits = (mother < 0).to(torch.long)
            edge_check, edge_bit, num_checks = self.edge_check_u, self.edge_bit_u, self.num_usable
        batch = bits.shape[0]
        par = torch.zeros(batch, num_checks, dtype=torch.long, device=bits.device)
        par.index_add_(1, edge_check, bits[:, edge_bit])
        return (par % 2 == 0).float().mean().item()
