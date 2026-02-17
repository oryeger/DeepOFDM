# sphere_decoder_fast_opt.py
# Optimized soft-output SphereDecoder for MU-MIMO (works for 4 users, 16QAM/64QAM).
#
# Main speed wins vs original + previous "fast":
# 1) Per-node pruning: only test constellation points within the level bound
# 2) Avoid full argsort: argpartition to shortlist K nearest (optional cap)
# 3) Leaf update for dmin is vectorized over users/bits (much less Python looping)
#
# Notes:
# - 'radius' is treated as a SQUARED distance threshold, consistent with your original.
# - For 64QAM, brute force is disabled by default for 4 users (64^4 too big).
#
# If you still want much faster: Numba JIT of the DFS is the next step.

import time
import numpy as np
from itertools import product
import commpy.modulation as mod

from python_code import conf
from python_code.coding.mcs_table import get_mcs


def SphereDecoder(
    H,
    y,
    noise_var=1.0,
    radius=np.inf,
    *,
    keep_bits=None,                  # None => all bits
    llr_definition="logP0_over_P1",  # matches your original sign: (d1-d0)/noise_var
    llr_clip=10000.0,

    # ---- search controls ----
    use_babai_init=True,
    init_expand=None,                # auto if None
    expand_factor=2.0,
    max_attempts=None,               # auto if None
    max_leaves=None,                 # auto if None

    # ---- pruning controls ----
    # Cap number of candidates examined per level after filtering; helps 64QAM a lot.
    max_cands_per_level=None,        # auto if None (depends on modulation)
    # If filtering yields > max_cands_per_level, shortlist nearest via argpartition.
    # If filtering yields 0, we still try nearest point.

    # ---- guarantee controls ----
    guarantee_mode=None,             # auto if None
    bruteforce_max_users=4,
    forced_radius_expand=10.0,
    max_nodes_forced=160000,

    debug=False,
):
    t_start_total = time.perf_counter()

    H = np.asarray(H)
    y = np.asarray(y)

    # Handle radius input
    if isinstance(radius, str):
        if radius.strip().lower() in {"inf", "+inf", "infinity"}:
            radius = float("inf")
        else:
            radius = float(radius)

    n_symbols, _ = y.shape
    n_users = H.shape[1]

    # Determine bits per symbol (bps) using your original logic
    if conf.mod_pilot <= 0:
        bits_per_symbol, _ = get_mcs(conf.mcs)
        bits_per_symbol = int(bits_per_symbol)
    else:
        bits_per_symbol = int(np.log2(conf.mod_pilot))

    bps = int(bits_per_symbol)
    M = int(2 ** bps)

    # ---- Modulation objects ----
    qam = mod.QAMModem(M)
    const = qam.constellation.astype(np.complex128)   # (M,)
    # Bit labels for each constellation index
    bits_table = (
        qam.demodulate(const, demod_type="hard")
        .reshape(M, bps)
        .astype(np.int8)
    )  # (M, bps)

    # ---- defaults tuned for 4 users ----
    if init_expand is None or max_attempts is None or max_leaves is None or guarantee_mode is None or max_cands_per_level is None:
        if bps <= 2:  # QPSK
            _init_expand = 3.5
            _max_attempts = 4
            _max_leaves = 4000
            _guarantee_mode = "hybrid"
            _max_cands = 8
        elif bps == 4:  # 16QAM
            _init_expand = 4.0
            _max_attempts = 4
            _max_leaves = 8000
            _guarantee_mode = "hybrid"   # 16^4 is feasible if needed
            _max_cands = 12
        else:  # 64QAM
            _init_expand = 6.0
            _max_attempts = 5
            _max_leaves = 25000
            _guarantee_mode = "forced_bit"  # avoid 64^4 brute force
            _max_cands = 16

        if init_expand is None: init_expand = _init_expand
        if max_attempts is None: max_attempts = _max_attempts
        if max_leaves is None: max_leaves = _max_leaves
        if guarantee_mode is None: guarantee_mode = _guarantee_mode
        if max_cands_per_level is None: max_cands_per_level = _max_cands

    # keep_bits handling
    if keep_bits is None:
        keep_bits = tuple(range(bps))
    elif isinstance(keep_bits, (int, np.integer)):
        keep_bits = (int(keep_bits),)
    else:
        keep_bits = tuple(int(k) for k in keep_bits)

    if any((k < 0 or k >= bps) for k in keep_bits):
        raise ValueError(f"keep_bits must be within [0, {bps-1}]. Got keep_bits={keep_bits}")

    keep_bits = np.asarray(keep_bits, dtype=np.int32)
    kout = int(len(keep_bits))

    # LLR convention
    if llr_definition == "logP0_over_P1":
        # LLR = (d1 - d0)/noise  ==  - (d0 - d1)/noise
        llr_sign = -1.0
    elif llr_definition == "logP1_over_P0":
        llr_sign = +1.0
    else:
        raise ValueError("llr_definition must be 'logP0_over_P1' or 'logP1_over_P0'")

    # QR decomposition
    Q, R = np.linalg.qr(H, mode="reduced")

    # Precompute |Rdiag| for bounds (avoid abs inside hot loop)
    Rdiag = np.abs(np.diag(R)) + 1e-15

    # Recursion buffers
    x = np.zeros(n_users, dtype=np.complex128)
    idx_vec = np.zeros(n_users, dtype=np.int32)

    # Helper: nearest constellation index
    def nearest_const_index(z):
        # For 64QAM this is still OK; if you want even faster, replace with grid quantization.
        return int(np.argmin(np.abs(const - z)))

    def babai_init(y_tilde):
        z = np.zeros(n_users, dtype=np.complex128)
        for k in range(n_users - 1, -1, -1):
            rhs = y_tilde[k] - np.dot(R[k, k + 1 :], z[k + 1 :])
            z[k] = rhs / R[k, k]
        idxs = np.array([nearest_const_index(z[k]) for k in range(n_users)], dtype=np.int32)
        x0 = const[idxs]
        d0 = np.linalg.norm(y_tilde - R @ x0) ** 2
        return idxs, d0

    def coverage_ok(dmin):
        return np.all(np.isfinite(dmin[:, :, 0]) & np.isfinite(dmin[:, :, 1]))

    # ---- core: filtered enumeration of candidates around est given remaining radius ----
    def enumerate_indices(est, level, partial_dist, radius_sq):
        # Bound on |est - s| from: new_dist = partial_dist + |rhs - rdiag*s|^2 <= radius_sq
        # where rhs = rdiag*est. So |rhs - rdiag*s| = |rdiag|*|est-s|
        rem = radius_sq - partial_dist
        if rem <= 0:
            # no room; still return nearest for robustness
            return np.array([nearest_const_index(est)], dtype=np.int32)

        bound = np.sqrt(rem) / Rdiag[level]  # radius in constellation space
        # Filter points in bound
        d = np.abs(const - est)
        mask = (d <= bound)

        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            return np.array([int(np.argmin(d))], dtype=np.int32)

        # If too many candidates, shortlist K nearest using argpartition
        if idxs.size > max_cands_per_level:
            # distances of filtered
            df = d[idxs]
            k = max_cands_per_level
            part = np.argpartition(df, k-1)[:k]
            idxs = idxs[part]
            df = df[part]
            # sort shortlisted
            idxs = idxs[np.argsort(df)]
            return idxs.astype(np.int32)

        # Otherwise sort all filtered
        idxs = idxs[np.argsort(d[idxs])]
        return idxs.astype(np.int32)

    # ---- truncated DFS that collects best and dmin ----
    def truncated_collect(y_tilde, radius_sq, leaf_cap):
        best_dist = np.inf
        best_idx = None
        dmin = np.full((n_users, kout, 2), np.inf, dtype=np.float64)
        leaf_count = 0

        def dfs(level, partial_dist):
            nonlocal best_dist, best_idx, leaf_count

            if partial_dist > radius_sq:
                return
            if leaf_count >= leaf_cap:
                return

            if level < 0:
                leaf_count += 1
                if partial_dist < best_dist:
                    best_dist = partial_dist
                    best_idx = idx_vec.copy()

                # Vectorized leaf update:
                # idx_vec -> (n_users,)
                # bits_table[idx_vec] -> (n_users, bps)
                # select kept -> (n_users, kout)
                bb = bits_table[idx_vec][:, keep_bits]  # int8
                # Update dmin[u, j, bb[u,j]] = min(current, partial_dist)
                # Do it with small loops over kout (kout <= 6), vectorized over users
                for j in range(kout):
                    bcol = bb[:, j].astype(np.int32)  # (n_users,)
                    for u in range(n_users):
                        b = bcol[u]
                        if partial_dist < dmin[u, j, b]:
                            dmin[u, j, b] = partial_dist
                return

            # compute rhs and est
            rhs = y_tilde[level] - np.dot(R[level, level + 1 :], x[level + 1 :])
            rdiag = R[level, level]
            est = rhs / rdiag

            # enumerate only plausible constellation points
            order = enumerate_indices(est, level, partial_dist, radius_sq)

            for ii in order:
                s = const[ii]
                inc = abs(rhs - rdiag * s) ** 2
                new_dist = partial_dist + inc
                if new_dist > radius_sq:
                    continue
                x[level] = s
                idx_vec[level] = ii
                dfs(level - 1, new_dist)

        dfs(n_users - 1, 0.0)
        return best_dist, best_idx, dmin

    # ---- guaranteed methods ----
    def bruteforce_all(y_tilde):
        best_dist = np.inf
        best_idx = None
        dmin = np.full((n_users, kout, 2), np.inf, dtype=np.float64)

        for tup in product(range(M), repeat=n_users):
            idxs = np.array(tup, dtype=np.int32)
            xvec = const[idxs]
            dist = np.linalg.norm(y_tilde - R @ xvec) ** 2

            if dist < best_dist:
                best_dist = dist
                best_idx = idxs.copy()

            bb = bits_table[idxs][:, keep_bits]
            for j in range(kout):
                bcol = bb[:, j].astype(np.int32)
                for u in range(n_users):
                    b = bcol[u]
                    if dist < dmin[u, j, b]:
                        dmin[u, j, b] = dist

        return best_dist, best_idx, dmin

    def forced_bit_min(y_tilde, radius_sq, u_force, bitpos_force, forced_val):
        nodes = 0
        best_dist = np.inf
        allowed_u = np.where(bits_table[:, bitpos_force] == forced_val)[0].astype(np.int32)

        def dfs(level, partial_dist):
            nonlocal nodes, best_dist
            nodes += 1
            if nodes >= max_nodes_forced:
                return
            if partial_dist > radius_sq:
                return

            if level < 0:
                if partial_dist < best_dist:
                    best_dist = partial_dist
                return

            rhs = y_tilde[level] - np.dot(R[level, level + 1 :], x[level + 1 :])
            rdiag = R[level, level]
            est = rhs / rdiag

            if level == u_force:
                cand = allowed_u
                # order allowed_u by closeness to est
                d = np.abs(const[cand] - est)
                cand = cand[np.argsort(d)]
            else:
                cand = enumerate_indices(est, level, partial_dist, radius_sq)

            for ii in cand:
                s = const[ii]
                inc = abs(rhs - rdiag * s) ** 2
                new_dist = partial_dist + inc
                if new_dist > radius_sq:
                    continue
                x[level] = s
                dfs(level - 1, new_dist)

        dfs(n_users - 1, 0.0)
        return best_dist

    # outputs
    LLRs_3d = np.zeros((n_symbols, n_users, kout), dtype=np.float64)
    hard_3d = np.zeros((n_symbols, n_users, kout), dtype=np.int8)

    # timing stats
    t_search_total = 0.0
    t_llr_total = 0.0
    n_forced = 0
    n_bf = 0

    for n in range(n_symbols):
        y_tilde = Q.conj().T @ y[n, :]

        # initial radius
        if np.isfinite(radius):
            radius_sq = float(radius)
            seed_dist = None
            idx_babai = None
        else:
            if use_babai_init:
                idx_babai, d_babai = babai_init(y_tilde)
                seed_dist = d_babai
            else:
                seed_dist = np.linalg.norm(y_tilde) ** 2 + 1e-12
                idx_babai = None
            radius_sq = seed_dist * init_expand + 1e-12

        best_dist = np.inf
        best_idx = None
        dmin = np.full((n_users, kout, 2), np.inf, dtype=np.float64)

        leaf_cap = max_leaves

        t0 = time.perf_counter()
        for attempt in range(max_attempts):
            bd, bi, dm = truncated_collect(y_tilde, radius_sq, leaf_cap)

            if bi is not None and bd < best_dist:
                best_dist, best_idx = bd, bi

            dmin = np.minimum(dmin, dm)

            if debug and n == 0:
                frac = np.mean(np.isfinite(dmin[:, :, 0]) & np.isfinite(dmin[:, :, 1]))
                print(f"[DEBUG] attempt={attempt} radius_sq={radius_sq:.3e} leaf_cap={leaf_cap} "
                      f"coverage={frac:.3f} best_dist={best_dist:.3e}")

            if coverage_ok(dmin) and best_idx is not None:
                break

            radius_sq *= expand_factor
            leaf_cap *= 2
        t_search_total += time.perf_counter() - t0

        # ensure best solution exists
        if best_idx is None:
            if use_babai_init and idx_babai is not None:
                best_idx = idx_babai
                best_dist = seed_dist
            else:
                best_idx = np.zeros(n_users, dtype=np.int32)
                best_dist = np.linalg.norm(y_tilde - R @ const[best_idx]) ** 2

        # guarantee
        if not coverage_ok(dmin):
            mode = guarantee_mode
            if mode == "hybrid":
                # brute force only if feasible (16QAM, <=4 users).
                mode = "bruteforce" if (M <= 16 and n_users <= bruteforce_max_users) else "forced_bit"

            if mode == "bruteforce":
                n_bf += 1
                best_dist, best_idx, dmin = bruteforce_all(y_tilde)
            elif mode == "forced_bit":
                n_forced += 1
                forced_radius_sq = best_dist * forced_radius_expand + 1e-9
                for u in range(n_users):
                    for j, bitpos in enumerate(keep_bits):
                        for b in (0, 1):
                            if not np.isfinite(dmin[u, j, b]):
                                d_forced = forced_bit_min(y_tilde, forced_radius_sq, u, bitpos, b)
                                dmin[u, j, b] = d_forced if np.isfinite(d_forced) else forced_radius_sq
            else:
                raise ValueError("guarantee_mode must be 'forced_bit', 'bruteforce', or 'hybrid'")

        # hard bits
        bb_best = bits_table[best_idx][:, keep_bits]
        hard_3d[n, :, :] = bb_best

        # llrs
        t0 = time.perf_counter()
        # LLR = llr_sign*(d0-d1)/noise_var
        for u in range(n_users):
            for j in range(kout):
                d0 = dmin[u, j, 0]
                d1 = dmin[u, j, 1]
                llr = llr_sign * (d0 - d1) / noise_var
                if llr_clip is not None:
                    llr = float(np.clip(llr, -llr_clip, llr_clip))
                LLRs_3d[n, u, j] = llr
        t_llr_total += time.perf_counter() - t0

    # reshape to match your pipeline convention: (n_symbols*kout, n_users)
    LLRs_all = LLRs_3d.transpose(0, 2, 1).reshape(n_symbols * kout, n_users)
    hard_bits_all = hard_3d.transpose(0, 2, 1).reshape(n_symbols * kout, n_users)

    t_total = time.perf_counter() - t_start_total
    # print(f"SphereDecoder: {t_total:.3f}s total | {t_total/max(n_symbols,1)*1e3:.2f} ms/sym")

    if debug:
        print(f"[DEBUG] SphereDecoderOpt total={t_total:.3f}s "
              f"| search={t_search_total:.3f}s llr={t_llr_total:.3f}s "
              f"| forced={n_forced} bruteforce={n_bf} "
              f"| {t_total/max(n_symbols,1)*1e3:.2f} ms/sym")

    return LLRs_all, hard_bits_all
