# -*- coding: utf-8 -*-
"""
sphere_16qam_evenbits.py

Fast-ish MU sphere-style search for TRUE 16QAM (CommPy mapping), but computes LLRs only
for selected bit positions (default keep_bits=(0,2)).

Key features:
- Robust argument handling (keep_bits can be int or iterable)
- Keyword-only keep_bits (prevents accidental passing of pilot_size=200 etc.)
- Adaptive truncated search (radius + leaf budget expansion)
- "No missing hypothesis" guarantee via targeted forced-bit fallback (or brute-force for small n_users)
- Outputs flattened as (Nsym*len(keep_bits), Nusers), matching your evaluate pipeline slice usage

IMPORTANT:
- This file assumes your evaluate.py calls the function using keywords:
    Sphere16qamEvenbits(H, y, noise_var=noise_var, keep_bits=(0,2), ...)
- If you try to pass positional args after noise_var, Python will error (by design).
"""

import time
import numpy as np
import commpy.modulation as mod
from itertools import product


def Sphere16qamEvenbits(
    H,
    y,
    noise_var=1.0,
    *,
    keep_bits=(0, 2),
    llr_definition="logP0_over_P1",
    llr_clip=10000.0,
    # --- fast search controls ---
    use_babai_init=True,
    init_expand=4.0,
    expand_factor=2.0,
    max_attempts=4,
    max_leaves=3000,
    # --- guarantee fallback controls ---
    guarantee_mode="forced_bit",     # "forced_bit" or "bruteforce" or "hybrid"
    bruteforce_max_users=4,
    forced_radius_expand=8.0,  # radius for forced searches relative to best_dist (squared)
    max_nodes_forced=80000,
    # --- debug ---
    debug=False,
):
    """
    TRUE 16QAM (CommPy mapping) joint detection over users, but outputs only keep_bits (default (0,2)).

    Returns:
      LLRs_all:      (Nsym*len(keep_bits), Nusers)
      hard_bits_all: (Nsym*len(keep_bits), Nusers)

    LLR convention:
      - "logP1_over_P0": LLR = log(P(b=1)/P(b=0)) ≈ (d0 - d1)/noise_var
      - "logP0_over_P1": LLR = log(P(b=0)/P(b=1)) ≈ (d1 - d0)/noise_var

    Guarantee behavior:
      - Run adaptive truncated search to populate dmin[u,j,0/1]
      - If any missing (inf), fill missing entries using:
          * bruteforce (guaranteed) if n_users <= bruteforce_max_users
          * forced_bit (targeted) otherwise
          * hybrid picks bruteforce when feasible else forced_bit
    """

    t0 = time.perf_counter()

    H = np.asarray(H)
    y = np.asarray(y)

    n_sym, _ = y.shape
    n_users = H.shape[1]

    # ---- CommPy 16QAM constellation + labels ----
    qam = mod.QAMModem(16)
    const = qam.constellation.astype(np.complex128)  # (16,)
    bps = int(qam.num_bits_symbol)                   # 4
    bits_table = (
        qam.demodulate(const, demod_type="hard")
        .reshape(len(const), bps)
        .astype(np.int8)
    )  # (16,4)

    # ---- normalize keep_bits input (accept int or iterable) ----
    if isinstance(keep_bits, (int, np.integer)):
        keep_bits = (int(keep_bits),)
    else:
        keep_bits = tuple(int(k) for k in keep_bits)

    # ---- validate keep_bits indices ----
    if any((k < 0 or k >= bps) for k in keep_bits):
        raise ValueError(
            f"keep_bits must be within [0, {bps-1}] for 16QAM. Got keep_bits={keep_bits}"
        )

    kout = len(keep_bits)

    # ---- LLR sign convention ----
    if llr_definition == "logP1_over_P0":
        llr_sign = +1.0  # (d0 - d1)/noise_var
    elif llr_definition == "logP0_over_P1":
        llr_sign = -1.0
    else:
        raise ValueError("llr_definition must be 'logP1_over_P0' or 'logP0_over_P1'")

    # ---- QR ----
    Q, R = np.linalg.qr(H, mode="reduced")

    # buffers for recursion
    x = np.zeros(n_users, dtype=np.complex128)
    idx_vec = np.zeros(n_users, dtype=np.int32)

    # ---------- helpers ----------
    def coverage_ok(dmin):
        return np.all(np.isfinite(dmin[:, :, 0]) & np.isfinite(dmin[:, :, 1]))

    def nearest_const_index(z):
        return int(np.argmin(np.abs(const - z)))

    def babai_init(y_tilde):
        # ZF on QR system: solve R z = y_tilde by back-substitution, then quantize to 16QAM points
        z = np.zeros(n_users, dtype=np.complex128)
        for k in range(n_users - 1, -1, -1):
            rhs = y_tilde[k] - np.dot(R[k, k + 1 :], z[k + 1 :])
            z[k] = rhs / R[k, k]
        idxs = np.array([nearest_const_index(z[k]) for k in range(n_users)], dtype=np.int32)
        x0 = const[idxs]
        d0 = np.linalg.norm(y_tilde - R @ x0) ** 2
        return idxs, d0

    # --- core truncated search that collects dmin for kept bits ---
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

                # update dmin for kept bits
                for u in range(n_users):
                    ii = int(idx_vec[u])
                    for j, bitpos in enumerate(keep_bits):
                        b = int(bits_table[ii, bitpos])
                        if partial_dist < dmin[u, j, b]:
                            dmin[u, j, b] = partial_dist
                return

            rhs = y_tilde[level] - np.dot(R[level, level + 1 :], x[level + 1 :])
            rdiag = R[level, level]
            est = rhs / rdiag

            order = np.argsort(np.abs(const - est))
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

    # --- brute force (guaranteed) ---
    def bruteforce_all(y_tilde):
        best_dist = np.inf
        best_idx = None
        dmin = np.full((n_users, kout, 2), np.inf, dtype=np.float64)

        for tup in product(range(16), repeat=n_users):
            idxs = np.array(tup, dtype=np.int32)
            xvec = const[idxs]
            dist = np.linalg.norm(y_tilde - R @ xvec) ** 2

            if dist < best_dist:
                best_dist = dist
                best_idx = idxs.copy()

            for u in range(n_users):
                ii = int(idxs[u])
                for j, bitpos in enumerate(keep_bits):
                    b = int(bits_table[ii, bitpos])
                    if dist < dmin[u, j, b]:
                        dmin[u, j, b] = dist

        return best_dist, best_idx, dmin

    # --- forced-bit targeted fallback for one missing (u, bitpos, bitval) ---
    def forced_bit_min(y_tilde, radius_sq, u_force, bitpos_force, forced_val):
        nodes = 0
        best_dist = np.inf

        allowed_u = np.where(bits_table[:, bitpos_force] == forced_val)[0]

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
            else:
                cand = range(16)

            cand = sorted(cand, key=lambda ii: abs(const[ii] - est))
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
    LLRs_3d = np.zeros((n_sym, n_users, kout), dtype=np.float64)
    hard_3d = np.zeros((n_sym, n_users, kout), dtype=np.int8)

    # ---------- main loop ----------
    for n in range(n_sym):
        y_tilde = Q.conj().T @ y[n, :]

        # initial radius based on Babai
        if use_babai_init:
            idx_babai, d_babai = babai_init(y_tilde)
            seed_dist = d_babai
        else:
            # loose fallback seed
            seed_dist = np.linalg.norm(y_tilde) ** 2 + 1e-12

        best_dist = np.inf
        best_idx = None
        dmin = np.full((n_users, kout, 2), np.inf, dtype=np.float64)

        radius_sq = seed_dist * init_expand + 1e-12
        leaf_cap = max_leaves

        # adaptive attempts
        for attempt in range(max_attempts):
            bd, bi, dm = truncated_collect(y_tilde, radius_sq, leaf_cap)

            if bi is not None and bd < best_dist:
                best_dist, best_idx = bd, bi

            dmin = np.minimum(dmin, dm)

            if debug and n == 0:
                frac = np.mean(np.isfinite(dmin[:, :, 0]) & np.isfinite(dmin[:, :, 1]))
                print(f"[DEBUG] attempt {attempt}: radius_sq={radius_sq:.3e}, leaf_cap={leaf_cap}, coverage={frac:.3f}")

            if coverage_ok(dmin) and best_idx is not None:
                break

            radius_sq *= expand_factor
            leaf_cap *= 2

        # ensure we have a best solution
        if best_idx is None:
            if use_babai_init:
                best_idx = idx_babai
                best_dist = seed_dist
            else:
                best_idx = np.zeros(n_users, dtype=np.int32)
                best_dist = np.linalg.norm(y_tilde - R @ const[best_idx]) ** 2

        # guarantee step if missing
        if not coverage_ok(dmin):
            mode = guarantee_mode
            if mode == "hybrid":
                mode = "bruteforce" if n_users <= bruteforce_max_users else "forced_bit"

            if mode == "bruteforce":
                if n_users > bruteforce_max_users:
                    raise ValueError(
                        f"bruteforce requested but n_users={n_users} > bruteforce_max_users={bruteforce_max_users}"
                    )
                best_dist, best_idx, dmin = bruteforce_all(y_tilde)

            elif mode == "forced_bit":
                # generous radius for forced bit searches
                forced_radius_sq = best_dist * forced_radius_expand + 1e-9

                # fill missing entries only
                for u in range(n_users):
                    for j, bitpos in enumerate(keep_bits):
                        for b in (0, 1):
                            if not np.isfinite(dmin[u, j, b]):
                                d_forced = forced_bit_min(y_tilde, forced_radius_sq, u, bitpos, b)
                                if np.isfinite(d_forced):
                                    dmin[u, j, b] = d_forced
                                else:
                                    # conservative finite fallback (will be clipped anyway)
                                    dmin[u, j, b] = forced_radius_sq

            else:
                raise ValueError("guarantee_mode must be 'forced_bit', 'bruteforce', or 'hybrid'")

        # hard bits from best_idx
        for u in range(n_users):
            ii = int(best_idx[u])
            for j, bitpos in enumerate(keep_bits):
                hard_3d[n, u, j] = int(bits_table[ii, bitpos])

        # LLRs from dmin (now finite)
        for u in range(n_users):
            for j in range(kout):
                d0b = dmin[u, j, 0]
                d1b = dmin[u, j, 1]
                llr = llr_sign * (d0b - d1b) / noise_var

                if llr_clip is not None:
                    llr = float(np.clip(llr, -llr_clip, llr_clip))
                LLRs_3d[n, u, j] = llr

    # flatten to (Nsym*kout, Nusers)
    LLRs_all = LLRs_3d.transpose(0, 2, 1).reshape(n_sym * kout, n_users)
    hard_bits_all = hard_3d.transpose(0, 2, 1).reshape(n_sym * kout, n_users)

    elapsed = time.perf_counter() - t0
    print(f"Sphere16qamEvenbits: {elapsed:.3f}s | {elapsed/max(n_sym,1)*1e3:.2f} ms/sym")

    return LLRs_all, hard_bits_all
