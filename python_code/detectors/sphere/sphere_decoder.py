import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

import numpy as np
from itertools import product
import commpy.modulation as mod


# # 16-QAM constellation points (normalized to unit energy)
# qam16 = np.array([-3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
#                   -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
#                   1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j,
#                   3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j]) / np.sqrt(10)

import numpy as np
from itertools import product

import numpy as np
from itertools import product


def SphereDecoder(H, y, noise_var=1.0, radius=np.inf):
    """
    Sphere decoder with LLR computation (Max-Log-MAP).

    Args:
        H: (N_rx x N_users) channel matrix
        y: (N_symbols x N_rx) received signals
        noise_var: noise variance
        radius: initial search radius (float, or 'inf'/np.inf for no constraint).
                NOTE: this radius is on the *squared* Euclidean distance scale
                (because partial_dist accumulates squared residuals).

    Returns:
        LLRs_all:     (N_symbols * bits_per_symbol, N_users)
        hard_bits_all:(N_symbols * bits_per_symbol, N_users)
    """
    import numpy as np
    from itertools import product

    # --- Coerce radius (handle YAML strings like "inf") ---
    if isinstance(radius, str):
        if radius.strip().lower() in {"inf", "+inf", "infinity"}:
            radius = float("inf")
        else:
            radius = float(radius)

    n_symbols, n_rx = y.shape
    n_users = H.shape[1]
    bits_per_symbol = int(np.log2(conf.mod_pilot))
    qam = mod.QAMModem(conf.mod_pilot)

    # Constellation and bit mapping
    constellation = qam.constellation  # complex symbols (numpy array)
    bit_combinations = np.array(list(product([0, 1], repeat=bits_per_symbol)), dtype=np.int64)
    # Map exact constellation values to their bit labels
    sym2bits = {constellation[i]: bit_combinations[i] for i in range(len(constellation))}

    # Precompute QR
    Q, R = np.linalg.qr(H, mode='reduced')

    # Allocate outputs
    LLRs_all = np.zeros((n_symbols, n_users, bits_per_symbol))
    hard_bits_all = np.zeros((n_symbols, n_users, bits_per_symbol), dtype=int)

    # --- Helpers ---
    def append_candidate(partial_x, partial_dist, best, candidates):
        """Append current leaf as a candidate and update best if needed."""
        # Build bits for this candidate (IMPORTANT: use current partial_x, not best["bits"])
        bits = np.concatenate([sym2bits[s] for s in partial_x])
        candidates.append((bits, partial_dist))
        if partial_dist < best["dist"]:
            best["dist"] = partial_dist
            best["bits"] = bits

    def search(level, partial_x, partial_dist, y_tilde, R, best, candidates, radius_sq):
        """Depth-first tree search with sphere pruning on squared radius."""
        if partial_dist > best["dist"] or partial_dist > radius_sq:
            return  # prune branch

        if level < 0:  # reached a full-length vector
            append_candidate(partial_x, partial_dist, best, candidates)
            return

        # Residual for this level
        r_diag = R[level, level]
        rhs = y_tilde[level] - np.dot(R[level, level + 1:], partial_x[level + 1:])

        # Babai estimate for ordering
        est = rhs / r_diag
        dists = np.abs(constellation - est)
        order = np.argsort(dists)

        for idx in order:
            s = constellation[idx]
            new_x = partial_x.copy()
            new_x[level] = s
            new_dist = partial_dist + np.abs(rhs - r_diag * s) ** 2
            search(level - 1, new_x, new_dist, y_tilde, R, best, candidates, radius_sq)

    def brute_force_candidates(y_tilde, R):
        """Emergency fallback to ensure soft info: enumerate all candidates."""
        candidates = []
        best = {"dist": np.inf, "bits": None}
        for symbol_tuple in product(constellation, repeat=n_users):
            x_vec = np.array(symbol_tuple, dtype=complex)
            dist = np.linalg.norm(y_tilde - R @ x_vec) ** 2
            bits = np.concatenate([sym2bits[s] for s in x_vec])
            candidates.append((bits, dist))
            if dist < best["dist"]:
                best["dist"] = dist
                best["bits"] = bits
        return best, candidates

    # Loop over all received symbols
    for idx in range(n_symbols):
        y_tilde = Q.conj().T @ y[idx, :]

        # Sphere search
        best = {"dist": np.inf, "bits": None}
        candidates = []
        radius_sq = radius  # radius is already squared-distance per docstring
        search(n_users - 1, np.zeros(n_users, dtype=complex), 0.0, y_tilde, R, best, candidates, radius_sq)

        # If the sphere was too tight (no candidates), fallback to brute-force for correctness
        if len(candidates) == 0:
            best, candidates = brute_force_candidates(y_tilde, R)

        # --- Compute LLRs (Max-Log-MAP) ---
        n_bits_total = n_users * bits_per_symbol
        LLRs_flat = np.zeros(n_bits_total)

        # Check coverage; if any hypothesis missing for any bit, recompute with brute-force
        need_full = False
        # quick pass to detect missing hypotheses cheaply
        for k in range(n_bits_total):
            has0 = any(bits[k] == 0 for bits, _ in candidates)
            has1 = any(bits[k] == 1 for bits, _ in candidates)
            if not (has0 and has1):
                need_full = True
                break
        if need_full:
            # recompute comprehensive candidate set for *this symbol* only
            best, candidates = brute_force_candidates(y_tilde, R)

        for k in range(n_bits_total):
            d0 = min(dist for bits, dist in candidates if bits[k] == 0)
            d1 = min(dist for bits, dist in candidates if bits[k] == 1)
            LLRs_flat[k] = (d1 - d0) / noise_var

        # reshape into (n_users, bits_per_symbol)
        LLRs_all[idx] = LLRs_flat.reshape(n_users, bits_per_symbol)
        hard_bits_all[idx] = best["bits"].reshape(n_users, bits_per_symbol)

    # Final shape as in your original return
    LLRs_all = LLRs_all.transpose(0, 2, 1).reshape(n_symbols * bits_per_symbol, n_users)
    hard_bits_all = hard_bits_all.transpose(0, 2, 1).reshape(n_symbols * bits_per_symbol, n_users)

    return LLRs_all, hard_bits_all
