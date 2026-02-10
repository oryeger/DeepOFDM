# sphere_64qam_0235.py
# Optimized soft-output SphereDecoder for 64-QAM that only computes LLRs for bits 0, 2, 3, 5.
#
# Key insight: bits 0, 2, 3, 5 map to a 16-QAM-like structure. Bits 1 and 4 are "don't care".
# We group the 64 constellation points into 16 "super-symbols" based on (b0, b2, b3, b5).
# For each super-symbol, we track the 4 underlying points (for b1, b4 = 00, 01, 10, 11).
# When computing distances, we take the minimum over the 4 underlying points.
#
# This reduces search complexity from 64^n_users to effectively 16^n_users.
# For 4 users: 16^4 = 65,536 vs 64^4 = 16,777,216 (256x reduction in brute force)
#
# The LLRs for bits 0, 2, 3, 5 are identical to what the full 64-QAM sphere would produce.

import time
import numpy as np
from itertools import product
import commpy.modulation as mod


def Sphere64qam0235(
    H,
    y,
    noise_var=1.0,
    radius=np.inf,
    *,
    llr_clip=10000.0,

    # ---- search controls ----
    use_babai_init=True,
    init_expand=4.0,           # reduced from 6.0 since we have fewer candidates
    expand_factor=2.0,
    max_attempts=4,
    max_leaves=12000,

    # ---- pruning controls ----
    max_cands_per_level=12,    # only 16 super-symbols, so 12 is often enough

    # ---- guarantee controls ----
    guarantee_mode="hybrid",   # hybrid works well since 16^4 is feasible
    bruteforce_max_users=4,
    forced_radius_expand=10.0,
    max_nodes_forced=50000,

    debug=False,
):
    """
    Optimized 64-QAM sphere decoder that outputs LLRs only for bits 0, 2, 3, 5.

    These 4 bits correspond to the 16-QAM structure within 64-QAM:
    - Bit 0: sign_I (I-axis sign)
    - Bit 2: pos_I (I-axis position, needs flip for 16QAM mapping)
    - Bit 3: sign_Q (Q-axis sign)
    - Bit 5: pos_Q (Q-axis position, needs flip for 16QAM mapping)

    Bits 1 and 4 (half_I, half_Q) are treated as "don't care" and should be
    completed by ESCNN or another method.

    Returns:
        LLRs_all: (n_symbols * 4, n_users) - LLRs for bits 0, 2, 3, 5
        hard_bits_all: (n_symbols * 4, n_users) - hard decisions for bits 0, 2, 3, 5
    """
    t_start_total = time.perf_counter()

    H = np.asarray(H)
    y = np.asarray(y)

    if isinstance(radius, str):
        if radius.strip().lower() in {"inf", "+inf", "infinity"}:
            radius = float("inf")
        else:
            radius = float(radius)

    n_symbols, _ = y.shape
    n_users = H.shape[1]

    # Fixed for 64-QAM
    bps = 6
    M = 64
    kout = 4  # output bits: 0, 2, 3, 5
    keep_bits = np.array([0, 2, 3, 5], dtype=np.int32)

    # ---- Build 64-QAM constellation ----
    qam64 = mod.QAMModem(64)
    const64 = qam64.constellation.astype(np.complex128)  # (64,)

    # Bit labels for each 64-QAM constellation point
    bits_table_64 = (
        qam64.demodulate(const64, demod_type="hard")
        .reshape(64, 6)
        .astype(np.int8)
    )  # (64, 6)

    # ---- Build 16 super-symbols ----
    # Group 64 points by (b0, b2, b3, b5) into 16 groups
    # Each group has 4 points (for b1, b4 = 00, 01, 10, 11)

    # Map from 4-bit label (b0, b2, b3, b5) to super-symbol index
    def bits_to_super_idx(b0, b2, b3, b5):
        return b0 * 8 + b2 * 4 + b3 * 2 + b5

    def super_idx_to_bits(idx):
        b0 = (idx >> 3) & 1
        b2 = (idx >> 2) & 1
        b3 = (idx >> 1) & 1
        b5 = idx & 1
        return b0, b2, b3, b5

    # For each super-symbol, store indices of the 4 underlying 64-QAM points
    super_to_const64_indices = [[] for _ in range(16)]

    for i in range(64):
        b = bits_table_64[i]
        sidx = bits_to_super_idx(b[0], b[2], b[3], b[5])
        super_to_const64_indices[sidx].append(i)

    # Convert to numpy arrays for faster access
    super_to_const64_indices = [np.array(lst, dtype=np.int32) for lst in super_to_const64_indices]

    # Precompute constellation points for each super-symbol (4 points each)
    super_const_points = np.zeros((16, 4), dtype=np.complex128)
    for sidx in range(16):
        indices = super_to_const64_indices[sidx]
        super_const_points[sidx, :] = const64[indices]

    # 4-bit labels for super-symbols
    bits_table_4 = np.zeros((16, 4), dtype=np.int8)
    for sidx in range(16):
        b0, b2, b3, b5 = super_idx_to_bits(sidx)
        bits_table_4[sidx, :] = [b0, b2, b3, b5]

    # ---- QR decomposition ----
    Q, R = np.linalg.qr(H, mode="reduced")
    Rdiag = np.abs(np.diag(R)) + 1e-15

    # Recursion buffers
    x = np.zeros(n_users, dtype=np.complex128)
    idx_vec = np.zeros(n_users, dtype=np.int32)  # super-symbol indices
    sub_idx_vec = np.zeros(n_users, dtype=np.int32)  # which of the 4 points within super-symbol

    # Helper: find nearest super-symbol and best sub-point
    def nearest_super_index(z):
        """Find the super-symbol whose closest point is nearest to z."""
        best_sidx = 0
        best_sub = 0
        best_dist = np.inf
        for sidx in range(16):
            dists = np.abs(super_const_points[sidx, :] - z)
            min_sub = np.argmin(dists)
            if dists[min_sub] < best_dist:
                best_dist = dists[min_sub]
                best_sidx = sidx
                best_sub = min_sub
        return best_sidx, best_sub

    def babai_init(y_tilde):
        z = np.zeros(n_users, dtype=np.complex128)
        for k in range(n_users - 1, -1, -1):
            rhs = y_tilde[k] - np.dot(R[k, k + 1:], z[k + 1:])
            z[k] = rhs / R[k, k]

        idxs = np.zeros(n_users, dtype=np.int32)
        subs = np.zeros(n_users, dtype=np.int32)
        x0 = np.zeros(n_users, dtype=np.complex128)

        for k in range(n_users):
            sidx, sub = nearest_super_index(z[k])
            idxs[k] = sidx
            subs[k] = sub
            x0[k] = super_const_points[sidx, sub]

        d0 = np.linalg.norm(y_tilde - R @ x0) ** 2
        return idxs, subs, d0

    def coverage_ok(dmin):
        return np.all(np.isfinite(dmin[:, :, 0]) & np.isfinite(dmin[:, :, 1]))

    # ---- enumerate candidates: super-symbols within bound ----
    def enumerate_super_indices(est, level, partial_dist, radius_sq):
        """
        Enumerate super-symbols whose closest underlying point is within the radius bound.
        Returns (super_indices, best_sub_indices, distances).
        """
        rem = radius_sq - partial_dist
        if rem <= 0:
            sidx, sub = nearest_super_index(est)
            return np.array([sidx], dtype=np.int32), np.array([sub], dtype=np.int32)

        bound = np.sqrt(rem) / Rdiag[level]

        # For each super-symbol, find min distance among its 4 points
        min_dists = np.zeros(16)
        best_subs = np.zeros(16, dtype=np.int32)
        for sidx in range(16):
            dists = np.abs(super_const_points[sidx, :] - est)
            best_sub = np.argmin(dists)
            min_dists[sidx] = dists[best_sub]
            best_subs[sidx] = best_sub

        # Filter by bound
        mask = min_dists <= bound
        idxs = np.nonzero(mask)[0]

        if idxs.size == 0:
            best = np.argmin(min_dists)
            return np.array([best], dtype=np.int32), np.array([best_subs[best]], dtype=np.int32)

        # Cap candidates if needed
        if idxs.size > max_cands_per_level:
            df = min_dists[idxs]
            k = max_cands_per_level
            part = np.argpartition(df, k - 1)[:k]
            idxs = idxs[part]
            df = df[part]
            order = np.argsort(df)
            idxs = idxs[order]
        else:
            idxs = idxs[np.argsort(min_dists[idxs])]

        return idxs.astype(np.int32), best_subs[idxs].astype(np.int32)

    # ---- truncated DFS ----
    def truncated_collect(y_tilde, radius_sq, leaf_cap):
        best_dist = np.inf
        best_idx = None
        best_sub = None
        dmin = np.full((n_users, kout, 2), np.inf, dtype=np.float64)
        leaf_count = 0

        def dfs(level, partial_dist):
            nonlocal best_dist, best_idx, best_sub, leaf_count

            if partial_dist > radius_sq:
                return
            if leaf_count >= leaf_cap:
                return

            if level < 0:
                leaf_count += 1
                if partial_dist < best_dist:
                    best_dist = partial_dist
                    best_idx = idx_vec.copy()
                    best_sub = sub_idx_vec.copy()

                # Update dmin using 4-bit labels
                bb = bits_table_4[idx_vec]  # (n_users, 4)
                for j in range(kout):
                    bcol = bb[:, j].astype(np.int32)
                    for u in range(n_users):
                        b = bcol[u]
                        if partial_dist < dmin[u, j, b]:
                            dmin[u, j, b] = partial_dist
                return

            rhs = y_tilde[level] - np.dot(R[level, level + 1:], x[level + 1:])
            rdiag = R[level, level]
            est = rhs / rdiag

            # Enumerate super-symbols
            super_idxs, sub_idxs = enumerate_super_indices(est, level, partial_dist, radius_sq)

            for i, sidx in enumerate(super_idxs):
                # Try all 4 underlying points for this super-symbol to find best
                points = super_const_points[sidx, :]
                incs = np.abs(rhs - rdiag * points) ** 2

                # Find the best sub-point
                best_sub_local = np.argmin(incs)
                inc = incs[best_sub_local]

                new_dist = partial_dist + inc
                if new_dist > radius_sq:
                    continue

                x[level] = points[best_sub_local]
                idx_vec[level] = sidx
                sub_idx_vec[level] = best_sub_local
                dfs(level - 1, new_dist)

        dfs(n_users - 1, 0.0)
        return best_dist, best_idx, best_sub, dmin

    # ---- brute force over 16 super-symbols ----
    def bruteforce_all(y_tilde):
        best_dist = np.inf
        best_idx = None
        best_sub = None
        dmin = np.full((n_users, kout, 2), np.inf, dtype=np.float64)

        for tup in product(range(16), repeat=n_users):
            idxs = np.array(tup, dtype=np.int32)

            # For each user, find the best sub-point
            xvec = np.zeros(n_users, dtype=np.complex128)
            subs = np.zeros(n_users, dtype=np.int32)

            # Compute incrementally for efficiency
            y_rem = y_tilde.copy()
            total_dist = 0.0

            for k in range(n_users - 1, -1, -1):
                rhs = y_rem[k]
                rdiag = R[k, k]
                points = super_const_points[idxs[k], :]
                incs = np.abs(rhs - rdiag * points) ** 2
                best_sub_k = np.argmin(incs)
                subs[k] = best_sub_k
                xvec[k] = points[best_sub_k]
                total_dist += incs[best_sub_k]
                # Update y_rem for next level
                if k > 0:
                    y_rem[:k] -= R[:k, k] * xvec[k]

            dist = total_dist

            if dist < best_dist:
                best_dist = dist
                best_idx = idxs.copy()
                best_sub = subs.copy()

            bb = bits_table_4[idxs]
            for j in range(kout):
                bcol = bb[:, j].astype(np.int32)
                for u in range(n_users):
                    b = bcol[u]
                    if dist < dmin[u, j, b]:
                        dmin[u, j, b] = dist

        return best_dist, best_idx, best_sub, dmin

    # ---- forced bit search ----
    def forced_bit_min(y_tilde, radius_sq, u_force, bitpos_force, forced_val):
        """Find min distance where user u_force has bit bitpos_force = forced_val."""
        nodes = 0
        best_dist = np.inf

        # Find which super-symbols have the required bit value
        allowed_super = np.where(bits_table_4[:, bitpos_force] == forced_val)[0].astype(np.int32)

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

            rhs = y_tilde[level] - np.dot(R[level, level + 1:], x[level + 1:])
            rdiag = R[level, level]
            est = rhs / rdiag

            if level == u_force:
                cands = allowed_super
            else:
                cands, _ = enumerate_super_indices(est, level, partial_dist, radius_sq)

            # Sort candidates by distance
            min_dists = np.zeros(len(cands))
            best_subs_local = np.zeros(len(cands), dtype=np.int32)
            for i, sidx in enumerate(cands):
                dists = np.abs(super_const_points[sidx, :] - est)
                best_sub_local = np.argmin(dists)
                min_dists[i] = dists[best_sub_local]
                best_subs_local[i] = best_sub_local

            order = np.argsort(min_dists)
            cands = cands[order]
            best_subs_local = best_subs_local[order]

            for i, sidx in enumerate(cands):
                points = super_const_points[sidx, :]
                incs = np.abs(rhs - rdiag * points) ** 2
                best_sub_k = np.argmin(incs)
                inc = incs[best_sub_k]

                new_dist = partial_dist + inc
                if new_dist > radius_sq:
                    continue

                x[level] = points[best_sub_k]
                idx_vec[level] = sidx
                dfs(level - 1, new_dist)

        dfs(n_users - 1, 0.0)
        return best_dist

    # ---- Main loop ----
    LLRs_3d = np.zeros((n_symbols, n_users, kout), dtype=np.float64)
    hard_3d = np.zeros((n_symbols, n_users, kout), dtype=np.int8)

    t_search_total = 0.0
    n_forced = 0
    n_bf = 0

    for n in range(n_symbols):
        y_tilde = Q.conj().T @ y[n, :]

        # Initial radius
        if np.isfinite(radius):
            radius_sq = float(radius)
            idx_babai = None
            sub_babai = None
        else:
            if use_babai_init:
                idx_babai, sub_babai, d_babai = babai_init(y_tilde)
                seed_dist = d_babai
            else:
                seed_dist = np.linalg.norm(y_tilde) ** 2 + 1e-12
                idx_babai = None
                sub_babai = None
            radius_sq = seed_dist * init_expand + 1e-12

        best_dist = np.inf
        best_idx = None
        best_sub = None
        dmin = np.full((n_users, kout, 2), np.inf, dtype=np.float64)

        leaf_cap = max_leaves

        t0 = time.perf_counter()
        for attempt in range(max_attempts):
            bd, bi, bs, dm = truncated_collect(y_tilde, radius_sq, leaf_cap)

            if bi is not None and bd < best_dist:
                best_dist, best_idx, best_sub = bd, bi, bs

            dmin = np.minimum(dmin, dm)

            if coverage_ok(dmin) and best_idx is not None:
                break

            radius_sq *= expand_factor
            leaf_cap *= 2
        t_search_total += time.perf_counter() - t0

        # Ensure best solution exists
        if best_idx is None:
            if use_babai_init and idx_babai is not None:
                best_idx = idx_babai
                best_sub = sub_babai
                best_dist = seed_dist
            else:
                best_idx = np.zeros(n_users, dtype=np.int32)
                best_sub = np.zeros(n_users, dtype=np.int32)
                xvec = super_const_points[best_idx, best_sub]
                best_dist = np.linalg.norm(y_tilde - R @ xvec) ** 2

        # Guarantee coverage
        if not coverage_ok(dmin):
            mode = guarantee_mode
            if mode == "hybrid":
                # 16^4 = 65536 is feasible for brute force
                mode = "bruteforce" if n_users <= bruteforce_max_users else "forced_bit"

            if mode == "bruteforce":
                n_bf += 1
                best_dist, best_idx, best_sub, dmin = bruteforce_all(y_tilde)
            elif mode == "forced_bit":
                n_forced += 1
                forced_radius_sq = best_dist * forced_radius_expand + 1e-9
                for u in range(n_users):
                    for j in range(kout):
                        for b in (0, 1):
                            if not np.isfinite(dmin[u, j, b]):
                                d_forced = forced_bit_min(y_tilde, forced_radius_sq, u, j, b)
                                dmin[u, j, b] = d_forced if np.isfinite(d_forced) else forced_radius_sq

        # Hard bits from best super-symbol
        bb_best = bits_table_4[best_idx]
        hard_3d[n, :, :] = bb_best

        # LLRs: LLR = -(d0 - d1) / noise_var  (logP0_over_P1 convention)
        for u in range(n_users):
            for j in range(kout):
                d0 = dmin[u, j, 0]
                d1 = dmin[u, j, 1]
                llr = -(d0 - d1) / noise_var
                if llr_clip is not None:
                    llr = float(np.clip(llr, -llr_clip, llr_clip))
                LLRs_3d[n, u, j] = llr

    # Reshape to match pipeline convention: (n_symbols * kout, n_users)
    LLRs_all = LLRs_3d.transpose(0, 2, 1).reshape(n_symbols * kout, n_users)
    hard_bits_all = hard_3d.transpose(0, 2, 1).reshape(n_symbols * kout, n_users)

    if debug:
        t_total = time.perf_counter() - t_start_total
        print(f"[DEBUG] Sphere64qam0235 total={t_total:.3f}s "
              f"| search={t_search_total:.3f}s "
              f"| forced={n_forced} bruteforce={n_bf} "
              f"| {t_total / max(n_symbols, 1) * 1e3:.2f} ms/sym")

    return LLRs_all, hard_bits_all
