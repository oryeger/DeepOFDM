import numpy as np
import commpy.modulation as mod
from collections import defaultdict
import time


import numpy as np
import commpy.modulation as mod
from collections import defaultdict
import time

def Sphere16qamEvenbits(H, y, noise_var=1.0, radius=np.inf,
                        keep_bits=(0, 2),
                        use_babai_init=True,
                        llr_definition="logP1_over_P0",
                        llr_clip=50.0):
    """
    Full 16QAM sphere decoder (joint over users) using CommPy mapping, but returns ONLY bits in keep_bits (default [0,2]).
    Output shapes are flattened to match your pipeline:
      - LLRs_all: (Nsym*len(keep_bits), Nusers)
      - hard_bits_all: (Nsym*len(keep_bits), Nusers)

    LLR convention:
      - "logP1_over_P0":  LLR = log(P(b=1)/P(b=0)) ≈ (d0 - d1)/noise_var
      - "logP0_over_P1":  LLR = log(P(b=0)/P(b=1)) ≈ (d1 - d0)/noise_var
    """

    t0 = time.time()

    # Ensure numpy arrays
    H = np.asarray(H)
    y = np.asarray(y)

    n_sym, n_rx = y.shape
    n_users = H.shape[1]

    # ---- CommPy 16QAM constellation + exact bit labels ----
    qam = mod.QAMModem(16)
    const = qam.constellation.astype(np.complex128)          # (16,)
    bps = int(qam.num_bits_symbol)                           # 4
    bits_table = qam.demodulate(const, demod_type='hard').reshape(len(const), bps).astype(np.int8)  # (16,4)

    keep_bits = tuple(int(k) for k in keep_bits)
    kout = len(keep_bits)

    # ---- Group constellation indices by kept bits ----
    # key: (b_keep0, b_keep1) e.g. (bit0, bit2)
    groups = defaultdict(list)
    for i in range(len(const)):
        key = tuple(int(bits_table[i, k]) for k in keep_bits)
        groups[key].append(i)

    group_keys = list(groups.keys())
    # representative point for group ordering
    rep = {k: np.mean(const[idxs]) for k, idxs in groups.items()}

    # ---- QR decomposition ----
    Q, R = np.linalg.qr(H, mode="reduced")

    # ---- LLR sign convention ----
    # p ~ exp(-dist/noise_var), so log(P1/P0) ~ -(d1-d0)/noise_var = (d0-d1)/noise_var
    if llr_definition == "logP1_over_P0":
        llr_sign = +1.0   # LLR = (d0 - d1)/noise_var
    elif llr_definition == "logP0_over_P1":
        llr_sign = -1.0   # LLR = (d1 - d0)/noise_var
    else:
        raise ValueError("llr_definition must be 'logP1_over_P0' or 'logP0_over_P1'")

    # ---- Radius handling (we prune using squared distances) ----
    radius_sq_global = np.inf if np.isinf(radius) else float(radius)**2

    # Buffers
    x = np.zeros(n_users, dtype=np.complex128)
    sym_idx = np.zeros(n_users, dtype=np.int32)

    def nearest_const_index(z):
        return int(np.argmin(np.abs(const - z)))

    def babai_init(y_tilde):
        # Solve R z = y_tilde by back substitution (ZF on QR system)
        z = np.zeros(n_users, dtype=np.complex128)
        for k in range(n_users - 1, -1, -1):
            rhs = y_tilde[k] - np.dot(R[k, k+1:], z[k+1:])
            z[k] = rhs / R[k, k]
        idxs = np.array([nearest_const_index(z[k]) for k in range(n_users)], dtype=np.int32)
        x0 = const[idxs]
        d0 = np.linalg.norm(y_tilde - R @ x0) ** 2
        return idxs, d0

    # Per-symbol outputs (3D), then flatten at the end
    LLRs_3d = np.zeros((n_sym, n_users, kout), dtype=np.float64)
    hard_3d = np.zeros((n_sym, n_users, kout), dtype=np.int8)

    # --- DFS sphere search ---
    def search(level, partial_dist, y_tilde, radius_sq):
        nonlocal best_dist, best_idx, dmin

        if partial_dist > radius_sq:
            return

        if level < 0:
            # Leaf: update best solution
            if partial_dist < best_dist:
                best_dist = partial_dist
                best_idx = sym_idx.copy()

            # Update Max-Log minima for kept bits only
            for u in range(n_users):
                ii = int(sym_idx[u])
                for j, bitpos in enumerate(keep_bits):
                    b = int(bits_table[ii, bitpos])
                    if partial_dist < dmin[u, j, b]:
                        dmin[u, j, b] = partial_dist
            return

        rhs = y_tilde[level] - np.dot(R[level, level+1:], x[level+1:])
        rdiag = R[level, level]
        est = rhs / rdiag

        # Group-first ordering (fast pruning)
        g_order = sorted(group_keys, key=lambda g: abs(rep[g] - est))
        for g in g_order:
            idxs = groups[g]
            idxs_sorted = sorted(idxs, key=lambda ii: abs(const[ii] - est))
            for ii in idxs_sorted:
                s = const[ii]
                inc = abs(rhs - rdiag * s)**2
                new_dist = partial_dist + inc
                if new_dist > radius_sq:
                    continue
                x[level] = s
                sym_idx[level] = ii
                search(level - 1, new_dist, y_tilde, radius_sq)

    # ---- main loop over symbols ----
    for n in range(n_sym):
        y_tilde = Q.conj().T @ y[n, :]

        # dmin[u, j, b] stores best metric for user u, kept-bit j, bit value b in {0,1}
        dmin = np.full((n_users, kout, 2), np.inf, dtype=np.float64)

        best_dist = np.inf
        best_idx = None
        radius_sq = radius_sq_global

        # Babai init makes pruning much stronger
        if use_babai_init:
            idxs0, d0 = babai_init(y_tilde)
            best_dist = d0
            best_idx = idxs0.copy()
            radius_sq = min(radius_sq, best_dist)

            # Also initialize dmin using Babai candidate (helps LLR stability if pruning is aggressive)
            for u in range(n_users):
                ii = int(best_idx[u])
                for j, bitpos in enumerate(keep_bits):
                    b = int(bits_table[ii, bitpos])
                    dmin[u, j, b] = min(dmin[u, j, b], best_dist)

        # Run sphere
        search(n_users - 1, 0.0, y_tilde, radius_sq)

        print("coverage fraction:",
              np.mean(np.isfinite(dmin[..., 0]) & np.isfinite(dmin[..., 1])))

        # If best_idx still None (should be rare), take Babai
        if best_idx is None:
            best_idx, _ = babai_init(y_tilde)

        # Hard bits + LLRs for kept bits
        for u in range(n_users):
            ii = int(best_idx[u])
            for j, bitpos in enumerate(keep_bits):
                hard_3d[n, u, j] = int(bits_table[ii, bitpos])

                d0 = dmin[u, j, 0]
                d1 = dmin[u, j, 1]
                if np.isfinite(d0) and np.isfinite(d1):
                    llr = llr_sign * (d0 - d1) / noise_var
                    # clip optional (stabilizes decoder)
                    if llr_clip is not None:
                        llr = float(np.clip(llr, -llr_clip, llr_clip))
                    LLRs_3d[n, u, j] = llr
                else:
                    # If one side not visited due to pruning, you can:
                    # - set 0 (uninformative), or
                    # - set large magnitude based on which side exists.
                    # Here: 0 for safety.
                    LLRs_3d[n, u, j] = 0.0

    # ---- Flatten to (Nsym*kout, Nusers) to match your evaluate.py expectations ----
    # (Nsym, Nusers, kout) -> (Nsym, kout, Nusers) -> (Nsym*kout, Nusers)
    LLRs_all = LLRs_3d.transpose(0, 2, 1).reshape(n_sym * kout, n_users)
    hard_bits_all = hard_3d.transpose(0, 2, 1).reshape(n_sym * kout, n_users)

    elapsed = time.time() - t0
    print(f"Sphere16qamEvenbits: {elapsed:.3f}s | {elapsed / n_sym * 1e3:.2f} ms/sym")
    return LLRs_all, hard_bits_all
