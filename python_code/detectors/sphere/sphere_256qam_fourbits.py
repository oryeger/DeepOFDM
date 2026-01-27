# sphere_256qam_fourbits.py
# Soft-output SphereDecoder for 256-QAM that only computes LLRs for bits 0, 3, 4, 7.
#
# Key insight: bits [0,3,4,7] correspond to a 16-QAM-like "coarse" structure:
#   - bit 0: I sign
#   - bit 3: I inner magnitude (|I| = 1/3 vs outer)
#   - bit 4: Q sign
#   - bit 7: Q inner magnitude (|Q| = 1/3 vs outer)
# The remaining bits [1,2,5,6] are "don't care" and are minimized over (max-log).
#
# We group 256 constellation points into 16 super-symbols by these 4 bits.
# Each super-symbol has 16 underlying points (for the 4 don't-care bits).
#
# When computing distances, we always take the minimum over the 16 underlying points.
# This reduces search complexity from 256^U to ~16^U.

import time
import numpy as np
from itertools import product
import commpy.modulation as mod


def Sphere256qamFourbits(
    H,
    y,
    noise_var=1.0,
    radius=np.inf,
    *,
    llr_clip=10000.0,

    # ---- search controls ----
    use_babai_init=True,
    init_expand=4.0,
    expand_factor=2.0,
    max_attempts=4,
    max_leaves=20000,

    # ---- pruning controls ----
    max_cands_per_level=12,

    # ---- guarantee controls ----
    guarantee_mode="hybrid",
    bruteforce_max_users=4,
    forced_radius_expand=10.0,
    max_nodes_forced=120000,

    debug=False,
):
    """
    Optimized 256-QAM sphere decoder that outputs LLRs only for bits 0, 3, 4, 7
    (I-sign, I-inner-mag, Q-sign, Q-inner-mag), matching commpy's bit labeling.

    Returns:
        LLRs_all: (n_symbols * 4, n_users) - LLRs for bits 0,3,4,7
        hard_bits_all: (n_symbols * 4, n_users) - hard decisions for bits 0,3,4,7
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

    # 256-QAM
    bps = 8
    M = 256
    kout = 4
    keep_bits = np.array([0, 3, 4, 7], dtype=np.int32)  # IMPORTANT: matches your mapping

    # ---- Build 256-QAM constellation ----
    qam256 = mod.QAMModem(256)
    const256 = qam256.constellation.astype(np.complex128)  # (256,)

    # Bit labels for each constellation point (commpy hard demod)
    bits_table_256 = (
        qam256.demodulate(const256, demod_type="hard")
        .reshape(M, bps)
        .astype(np.int8)
    )  # (256, 8)

    # ---- Build 16 super-symbols ----
    # group by bits (0,3,4,7) -> 4-bit label -> super index [0..15]
    def bits_to_super_idx(b0, b3, b4, b7):
        return (b0 << 3) | (b3 << 2) | (b4 << 1) | b7

    def super_idx_to_bits(idx):
        b0 = (idx >> 3) & 1
        b3 = (idx >> 2) & 1
        b4 = (idx >> 1) & 1
        b7 = idx & 1
        return b0, b3, b4, b7

    super_to_const256_indices = [[] for _ in range(16)]

    for i in range(M):
        b = bits_table_256[i]
        sidx = bits_to_super_idx(b[0], b[3], b[4], b[7])
        super_to_const256_indices[sidx].append(i)

    # each should have 16 points
    super_to_const256_indices = [np.array(lst, dtype=np.int32) for lst in super_to_const256_indices]

    # Precompute constellation points for each super-symbol (16 points each)
    KSUB = 16
    super_const_points = np.zeros((16, KSUB), dtype=np.complex128)
    for sidx in range(16):
        idxs = super_to_const256_indices[sidx]
        if idxs.size != KSUB:
            raise RuntimeError(f"Super-symbol {sidx} has {idxs.size} points (expected 16). "
                               f"Check keep_bits={keep_bits.tolist()} vs mapping.")
        super_const_points[sidx, :] = const256[idxs]

    # 4-bit labels for super-symbols
    bits_table_4 = np.zeros((16, 4), dtype=np.int8)
    for sidx in range(16):
        b0, b3, b4, b7 = super_idx_to_bits(sidx)
        bits_table_4[sidx, :] = [b0, b3, b4, b7]

    # ---- QR decomposition ----
    Q, R = np.linalg.qr(H, mode="reduced")
    Rdiag = np.abs(np.diag(R)) + 1e-15

    # recursion buffers
    x = np.zeros(n_users, dtype=np.complex128)
    idx_vec = np.zeros(n_users, dtype=np.int32)     # super-symbol indices
    sub_idx_vec = np.zeros(n_users, dtype=np.int32) # which point within super-symbol (0..15)

    # helper: nearest super-symbol and best sub-point
    def nearest_super_index(z):
        best_sidx = 0
        best_sub = 0
        best_dist = np.inf
        for sidx in range(16):
            dists = np.abs(super_const_points[sidx, :] - z)
            sub = int(np.argmin(dists))
            if dists[sub] < best_dist:
                best_dist = float(dists[sub])
                best_sidx = sidx
                best_sub = sub
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

    def enumerate_super_indices(est, level, partial_dist, radius_sq):
        rem = radius_sq - partial_dist
        if rem <= 0:
            sidx, sub = nearest_super_index(est)
            return np.array([sidx], dtype=np.int32), np.array([sub], dtype=np.int32)

        bound = np.sqrt(rem) / Rdiag[level]

        # min distance to each super-symbol (over its 16 points)
        min_dists = np.zeros(16, dtype=np.float64)
        best_subs = np.zeros(16, dtype=np.int32)
        for sidx in range(16):
            d = np.abs(super_const_points[sidx, :] - est)
            sub = int(np.argmin(d))
            min_dists[sidx] = float(d[sub])
            best_subs[sidx] = sub

        mask = min_dists <= bound
        idxs = np.nonzero(mask)[0]

        if idxs.size == 0:
            best = int(np.argmin(min_dists))
            return np.array([best], dtype=np.int32), np.array([best_subs[best]], dtype=np.int32)

        # cap candidates if needed
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

            if partial_dist > radius_sq or leaf_count >= leaf_cap:
                return

            if level < 0:
                leaf_count += 1
                if partial_dist < best_dist:
                    best_dist = partial_dist
                    best_idx = idx_vec.copy()
                    best_sub = sub_idx_vec.copy()

                bb = bits_table_4[idx_vec]  # (n_users, 4)
                for j in range(kout):
                    bcol = bb[:, j].astype(np.int32)
                    for u in range(n_users):
                        b = int(bcol[u])
                        if partial_dist < dmin[u, j, b]:
                            dmin[u, j, b] = partial_dist
                return

            rhs = y_tilde[level] - np.dot(R[level, level + 1:], x[level + 1:])
            rdiag = R[level, level]
            est = rhs / rdiag

            super_idxs, _ = enumerate_super_indices(est, level, partial_dist, radius_sq)

            for sidx in super_idxs:
                points = super_const_points[sidx, :]  # (16,)
                incs = np.abs(rhs - rdiag * points) ** 2
                best_sub_local = int(np.argmin(incs))
                inc = float(incs[best_sub_local])

                new_dist = partial_dist + inc
                if new_dist > radius_sq:
                    continue

                x[level] = points[best_sub_local]
                idx_vec[level] = int(sidx)
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

            xvec = np.zeros(n_users, dtype=np.complex128)
            subs = np.zeros(n_users, dtype=np.int32)

            y_rem = y_tilde.copy()
            total_dist = 0.0

            for k in range(n_users - 1, -1, -1):
                rhs = y_rem[k]
                rdiag = R[k, k]
                points = super_const_points[idxs[k], :]
                incs = np.abs(rhs - rdiag * points) ** 2
                best_sub_k = int(np.argmin(incs))
                subs[k] = best_sub_k
                xvec[k] = points[best_sub_k]
                total_dist += float(incs[best_sub_k])
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
                    b = int(bcol[u])
                    if dist < dmin[u, j, b]:
                        dmin[u, j, b] = dist

        return best_dist, best_idx, best_sub, dmin

    # ---- forced bit search (optional guarantee) ----
    def forced_bit_min(y_tilde, radius_sq, u_force, bitpos_force, forced_val):
        nodes = 0
        best_dist = np.inf

        allowed_super = np.where(bits_table_4[:, bitpos_force] == forced_val)[0].astype(np.int32)

        def dfs(level, partial_dist):
            nonlocal nodes, best_dist
            nodes += 1
            if nodes >= max_nodes_forced or partial_dist > radius_sq:
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

            # order candidates by closeness
            min_d = np.zeros(len(cands), dtype=np.float64)
            for i, sidx in enumerate(cands):
                d = np.abs(super_const_points[sidx, :] - est)
                min_d[i] = float(np.min(d))
            cands = cands[np.argsort(min_d)]

            for sidx in cands:
                points = super_const_points[sidx, :]
                incs = np.abs(rhs - rdiag * points) ** 2
                best_sub_k = int(np.argmin(incs))
                inc = float(incs[best_sub_k])

                new_dist = partial_dist + inc
                if new_dist > radius_sq:
                    continue

                x[level] = points[best_sub_k]
                idx_vec[level] = int(sidx)
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
                seed_dist = float(np.linalg.norm(y_tilde) ** 2 + 1e-12)
                idx_babai = None
                sub_babai = None
            radius_sq = seed_dist * init_expand + 1e-12

        best_idx = None
        best_sub = None
        best_dist = np.inf
        dmin = np.full((n_users, kout, 2), np.inf, dtype=np.float64)

        leaf_cap = max_leaves

        t0 = time.perf_counter()
        for _ in range(max_attempts):
            bd, bi, bs, dm = truncated_collect(y_tilde, radius_sq, leaf_cap)

            if bi is not None and bd < best_dist:
                best_dist, best_idx, best_sub = bd, bi, bs
            dmin = np.minimum(dmin, dm)

            if coverage_ok(dmin) and best_idx is not None:
                break

            radius_sq *= expand_factor
            leaf_cap *= 2
        t_search_total += time.perf_counter() - t0

        # Ensure best exists
        if best_idx is None:
            if use_babai_init and idx_babai is not None:
                best_idx = idx_babai
                best_sub = sub_babai
                best_dist = seed_dist
            else:
                best_idx = np.zeros(n_users, dtype=np.int32)
                best_sub = np.zeros(n_users, dtype=np.int32)
                xvec = super_const_points[best_idx, best_sub]
                best_dist = float(np.linalg.norm(y_tilde - R @ xvec) ** 2)

        # Guarantee coverage
        if not coverage_ok(dmin):
            mode = guarantee_mode
            if mode == "hybrid":
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
        hard_3d[n, :, :] = bits_table_4[best_idx]

        # LLRs (max-log): LLR = -(d0 - d1) / noise_var
        for u in range(n_users):
            for j in range(kout):
                d0 = dmin[u, j, 0]
                d1 = dmin[u, j, 1]
                llr = -(d0 - d1) / noise_var
                if llr_clip is not None:
                    llr = float(np.clip(llr, -llr_clip, llr_clip))
                LLRs_3d[n, u, j] = llr

    LLRs_all = LLRs_3d.transpose(0, 2, 1).reshape(n_symbols * kout, n_users)
    hard_bits_all = hard_3d.transpose(0, 2, 1).reshape(n_symbols * kout, n_users)

    if debug:
        t_total = time.perf_counter() - t_start_total
        print(f"[DEBUG] Sphere256qamFourbits total={t_total:.3f}s "
              f"| search={t_search_total:.3f}s "
              f"| forced={n_forced} bruteforce={n_bf} "
              f"| {t_total / max(n_symbols, 1) * 1e3:.2f} ms/sym")

    return LLRs_all, hard_bits_all
