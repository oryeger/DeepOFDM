import numpy as np
import commpy.modulation as mod
from collections import defaultdict
import time


import numpy as np
import commpy.modulation as mod
from collections import defaultdict
import time

import numpy as np
import time
import commpy.modulation as mod
from collections import defaultdict


def Sphere16qamEvenbits(H, y, noise_var=1.0, radius=np.inf,
                        keep_bits=(0, 2),
                        use_babai_init=True,
                        llr_definition="logP0_over_P1",
                        llr_clip=1000.0,
                        llr_radius_expand=8.0,
                        max_leaves=2000,
                        early_stop_on_coverage=False):
    """
    Joint sphere decoder over users for TRUE 16QAM (CommPy mapping), but outputs only keep_bits (default bits [0,2]).
    Returns flattened arrays to match your evaluate code:
      - LLRs_all:      (Nsym*len(keep_bits), Nusers)
      - hard_bits_all: (Nsym*len(keep_bits), Nusers)

    LLR convention:
      - "logP1_over_P0": LLR = log(P(b=1)/P(b=0)) ≈ (d0 - d1)/noise_var
      - "logP0_over_P1": LLR = log(P(b=0)/P(b=1)) ≈ (d1 - d0)/noise_var

    Speed/robustness knobs:
      - llr_radius_expand: expands Babai metric radius to allow both hypotheses to be found for LLRs
      - max_leaves: limits how many leaf candidates to visit per received vector
      - early_stop_on_coverage: stop once both hypotheses seen for all (user, kept-bit)
    """

    t0 = time.perf_counter()

    H = np.asarray(H)
    y = np.asarray(y)

    n_sym, _ = y.shape
    n_users = H.shape[1]

    # --- CommPy 16QAM constellation + its exact bit labeling ---
    qam = mod.QAMModem(16)
    const = qam.constellation.astype(np.complex128)  # (16,)
    bps = int(qam.num_bits_symbol)                   # 4
    bits_table = qam.demodulate(const, demod_type='hard') \
                    .reshape(len(const), bps).astype(np.int8)  # (16,4)

    keep_bits = tuple(int(k) for k in keep_bits)
    kout = len(keep_bits)

    # --- Group indices by the kept bits (e.g. (bit0, bit2)) ---
    groups = defaultdict(list)
    for i in range(len(const)):
        key = tuple(int(bits_table[i, k]) for k in keep_bits)
        groups[key].append(i)
    group_keys = list(groups.keys())

    # Representative per group for ordering (mean point)
    rep = {k: np.mean(const[idxs]) for k, idxs in groups.items()}

    # --- QR factorization ---
    Q, R = np.linalg.qr(H, mode="reduced")

    # --- LLR sign convention ---
    # p ~ exp(-dist/noise_var) => log(P1/P0) ≈ -(d1-d0)/noise_var = (d0-d1)/noise_var
    if llr_definition == "logP1_over_P0":
        llr_sign = +1.0
    elif llr_definition == "logP0_over_P1":
        llr_sign = -1.0
    else:
        raise ValueError("llr_definition must be 'logP1_over_P0' or 'logP0_over_P1'")

    # --- radius handling: we prune with squared distances ---
    radius_sq_global = np.inf if np.isinf(radius) else float(radius) ** 2

    # Buffers for DFS
    x = np.zeros(n_users, dtype=np.complex128)
    sym_idx = np.zeros(n_users, dtype=np.int32)

    def nearest_const_index(z):
        return int(np.argmin(np.abs(const - z)))

    def babai_init(y_tilde):
        # Back substitution: solve R z = y_tilde
        z = np.zeros(n_users, dtype=np.complex128)
        for k in range(n_users - 1, -1, -1):
            rhs = y_tilde[k] - np.dot(R[k, k+1:], z[k+1:])
            z[k] = rhs / R[k, k]
        idxs = np.array([nearest_const_index(z[k]) for k in range(n_users)], dtype=np.int32)
        x0 = const[idxs]
        d0 = np.linalg.norm(y_tilde - R @ x0) ** 2
        return idxs, d0

    def coverage_ok(dmin):
        return np.all(np.isfinite(dmin[:, :, 0]) & np.isfinite(dmin[:, :, 1]))

    # Outputs per symbol (3D), flattened at end
    LLRs_3d = np.zeros((n_sym, n_users, kout), dtype=np.float64)
    hard_3d = np.zeros((n_sym, n_users, kout), dtype=np.int8)

    # DFS search
    def search(level, partial_dist, y_tilde, radius_sq):
        nonlocal best_dist, best_idx, dmin, leaf_count, stop_flag

        if stop_flag:
            return
        if partial_dist > radius_sq:
            return

        if level < 0:
            leaf_count += 1

            # update best solution (hard decisions)
            if partial_dist < best_dist:
                best_dist = partial_dist
                best_idx = sym_idx.copy()

            # update minima for LLRs for kept bits only
            for u in range(n_users):
                ii = int(sym_idx[u])
                for j, bitpos in enumerate(keep_bits):
                    b = int(bits_table[ii, bitpos])
                    if partial_dist < dmin[u, j, b]:
                        dmin[u, j, b] = partial_dist

            # stopping criteria
            if early_stop_on_coverage and coverage_ok(dmin):
                stop_flag = True
                return
            if leaf_count >= max_leaves:
                stop_flag = True
                return

            return

        rhs = y_tilde[level] - np.dot(R[level, level+1:], x[level+1:])
        rdiag = R[level, level]
        est = rhs / rdiag

        # group-first ordering
        g_order = sorted(group_keys, key=lambda g: abs(rep[g] - est))
        for g in g_order:
            idxs = groups[g]
            # within group: closest first
            idxs_sorted = sorted(idxs, key=lambda ii: abs(const[ii] - est))
            for ii in idxs_sorted:
                s = const[ii]
                inc = abs(rhs - rdiag * s) ** 2
                new_dist = partial_dist + inc
                if new_dist > radius_sq:
                    continue
                x[level] = s
                sym_idx[level] = ii
                search(level - 1, new_dist, y_tilde, radius_sq)
                if stop_flag:
                    return

    # Main loop
    for n in range(n_sym):
        y_tilde = Q.conj().T @ y[n, :]

        dmin = np.full((n_users, kout, 2), np.inf, dtype=np.float64)

        best_dist = np.inf
        best_idx = None

        # Expanded radius for LLR coverage
        radius_sq = radius_sq_global

        if use_babai_init:
            idxs0, d0 = babai_init(y_tilde)
            best_idx = idxs0.copy()
            best_dist = d0
            # Expand so that we see both hypotheses:
            radius_sq = min(radius_sq, best_dist * llr_radius_expand)

            # seed dmin using the Babai point (helps if max_leaves is small)
            for u in range(n_users):
                ii = int(best_idx[u])
                for j, bitpos in enumerate(keep_bits):
                    b = int(bits_table[ii, bitpos])
                    dmin[u, j, b] = min(dmin[u, j, b], best_dist)

        leaf_count = 0
        stop_flag = False
        search(n_users - 1, 0.0, y_tilde, radius_sq)

        if best_idx is None:
            best_idx, best_dist = babai_init(y_tilde)

        # Hard bits
        for u in range(n_users):
            ii = int(best_idx[u])
            for j, bitpos in enumerate(keep_bits):
                hard_3d[n, u, j] = int(bits_table[ii, bitpos])

        # LLRs (Max-Log) with missing-side saturation
        BIG = best_dist + (llr_clip if llr_clip is not None else 50.0) * noise_var

        for u in range(n_users):
            for j in range(kout):
                d0 = dmin[u, j, 0]
                d1 = dmin[u, j, 1]

                if np.isfinite(d0) and np.isfinite(d1):
                    llr = llr_sign * (d0 - d1) / noise_var
                elif np.isfinite(d0) and not np.isfinite(d1):
                    llr = llr_sign * (d0 - BIG) / noise_var
                elif np.isfinite(d1) and not np.isfinite(d0):
                    llr = llr_sign * (BIG - d1) / noise_var
                else:
                    llr = 0.0

                if llr_clip is not None:
                    llr = float(np.clip(llr, -llr_clip, llr_clip))
                LLRs_3d[n, u, j] = llr

    # Flatten to (Nsym*kout, Nusers) in the order: [bit0, bit2, bit0, bit2, ...] per symbol
    LLRs_all = LLRs_3d.transpose(0, 2, 1).reshape(n_sym * kout, n_users)
    hard_bits_all = hard_3d.transpose(0, 2, 1).reshape(n_sym * kout, n_users)

    elapsed = time.perf_counter() - t0
    print(f"Sphere16qamEvenbits: {elapsed:.3f}s | {elapsed/n_sym*1e3:.2f} ms/sym")
    return LLRs_all, hard_bits_all
