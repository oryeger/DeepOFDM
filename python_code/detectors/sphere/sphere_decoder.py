import numpy as np
from itertools import product
import commpy.modulation as mod
from python_code import conf
from python_code.coding.mcs_table import get_mcs

# import time


# ---------------- Helper functions ---------------- #

def append_candidate(partial_x, partial_dist, sym2bits, best, candidates):
    """
    Append a leaf candidate to the candidate list and update the best metric.
    """
    bits = np.concatenate([sym2bits[s] for s in partial_x])
    candidates.append((bits, partial_dist))
    if partial_dist < best["dist"]:
        best["dist"] = partial_dist
        best["bits"] = bits


def sphere_search(level, partial_x, partial_dist, y_tilde, R, sym2bits, best, candidates, radius_sq, constellation, max_candidates=10000):
    """
    Depth-first sphere search with pruning.
    Enhanced with dynamic radius updates and candidate limiting for 64QAM.
    """
    # Prune based on both current best distance and radius
    if partial_dist > best["dist"] or partial_dist > radius_sq:
        return

    # Limit candidate explosion for high-order modulations
    if len(candidates) >= max_candidates:
        return

    if level < 0:  # leaf node
        append_candidate(partial_x, partial_dist, sym2bits, best, candidates)
        return

    r_diag = R[level, level]
    rhs = y_tilde[level] - np.dot(R[level, level + 1:], partial_x[level + 1:])
    est = rhs / r_diag
    order = np.argsort(np.abs(constellation - est))

    # For high-order modulations (64QAM), limit search at upper levels
    # This provides a good trade-off between complexity and performance
    constellation_size = len(constellation)
    if constellation_size >= 64:
        # At upper levels, only explore most promising candidates
        if level >= len(partial_x) // 2:
            # Top half of tree: explore fewer candidates
            max_explore = min(8, constellation_size)
        else:
            # Bottom half: explore more candidates
            max_explore = min(16, constellation_size)
        order = order[:max_explore]

    for idx in order:
        s = constellation[idx]
        new_x = partial_x.copy()
        new_x[level] = s
        new_dist = partial_dist + np.abs(rhs - r_diag * s) ** 2
        sphere_search(level - 1, new_x, new_dist, y_tilde, R, sym2bits, best, candidates, radius_sq, constellation, max_candidates)


def brute_force_candidates(y_tilde, R, n_users, constellation, sym2bits):
    """
    Emergency fallback: enumerate all candidates to ensure soft info.
    """
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


# ---------------- Main Sphere Decoder ---------------- #

def SphereDecoder(H, y, noise_var=1.0, radius=np.inf):
    """
    Sphere decoder with LLR computation (Max-Log-MAP).
    Soft-output version with optional sphere radius.
    Enhanced for 64QAM with adaptive radius and candidate limiting.
    """
    # Handle radius input
    if isinstance(radius, str):
        if radius.strip().lower() in {"inf", "+inf", "infinity"}:
            radius = float("inf")
        else:
            radius = float(radius)

    n_symbols, n_rx = y.shape
    n_users = H.shape[1]
    bits_per_symbol, _ = get_mcs(conf.mcs)
    bits_per_symbol = int(bits_per_symbol)
    if conf.increase_prime_modulation:
        # Reduce modulation: 16QAM(4)→QPSK(2), 64QAM(6)→16QAM(4)
        if bits_per_symbol == 4:
            bits_per_symbol = 2  # QPSK for 16QAM output
        elif bits_per_symbol == 6:
            bits_per_symbol = 4  # 16QAM for 64QAM output
    qam = mod.QAMModem(int(2 ** bits_per_symbol))

    # Constellation and bit mapping
    constellation = qam.constellation
    constellation_size = len(constellation)
    bit_combinations = np.array(list(product([0, 1], repeat=bits_per_symbol)), dtype=np.int64)
    sym2bits = {constellation[i]: bit_combinations[i] for i in range(len(constellation))}

    # QR decomposition
    Q, R = np.linalg.qr(H, mode='reduced')

    # Output arrays
    LLRs_all = np.zeros((n_symbols, n_users, bits_per_symbol))
    hard_bits_all = np.zeros((n_symbols, n_users, bits_per_symbol), dtype=int)

    # Adaptive parameters for 64QAM
    if constellation_size >= 64:
        # For 64QAM, use finite radius if not specified
        if np.isinf(radius):
            # Use config radius or reasonable default
            radius = getattr(conf, 'sphere_radius', 200.0) * noise_var
        # Limit candidates to prevent memory issues
        max_candidates = 50000
    else:
        # For lower modulations, more permissive
        if np.isinf(radius):
            radius = 1000.0 * noise_var
        max_candidates = 100000

    # Loop over received symbols
    for idx in range(n_symbols):
        y_tilde = Q.conj().T @ y[idx, :]

        best = {"dist": np.inf, "bits": None}
        candidates = []
        radius_sq = radius

        # Sphere search with dynamic pruning
        sphere_search(n_users - 1, np.zeros(n_users, dtype=complex), 0.0,
                      y_tilde, R, sym2bits, best, candidates, radius_sq, constellation, max_candidates)

        # Fallback if sphere too tight or no candidates found
        if len(candidates) == 0:
            # For 64QAM, try expanding radius before brute force
            if constellation_size >= 64 and not np.isinf(radius_sq):
                # Try with 2x radius
                expanded_radius = radius_sq * 4.0
                sphere_search(n_users - 1, np.zeros(n_users, dtype=complex), 0.0,
                             y_tilde, R, sym2bits, best, candidates, expanded_radius, constellation, max_candidates)

            # If still no candidates, use brute force (only for small cases)
            if len(candidates) == 0:
                if constellation_size ** n_users <= 100000:  # Prevent catastrophic brute force
                    best, candidates = brute_force_candidates(y_tilde, R, n_users, constellation, sym2bits)
                else:
                    # Emergency: use zero-forcing estimate as single candidate
                    zf_x = np.linalg.lstsq(R, y_tilde, rcond=None)[0]
                    zf_symbols = constellation[np.argmin(np.abs(constellation[:, None] - zf_x[None, :]), axis=0)]
                    zf_dist = np.linalg.norm(y_tilde - R @ zf_symbols) ** 2
                    zf_bits = np.concatenate([sym2bits[s] for s in zf_symbols])
                    candidates.append((zf_bits, zf_dist))
                    best["dist"] = zf_dist
                    best["bits"] = zf_bits

        # --- Compute LLRs (Max-Log-MAP) ---
        n_bits_total = n_users * bits_per_symbol
        LLRs_flat = np.zeros(n_bits_total)

        # Check coverage; fallback to brute-force if missing hypotheses
        for k in range(n_bits_total):
            # collect d0 and d1 across candidates
            d0_candidates = [dist for bits, dist in candidates if bits[k] == 0]
            d1_candidates = [dist for bits, dist in candidates if bits[k] == 1]

            if d0_candidates and d1_candidates:
                # both hypotheses exist → normal LLR
                d0 = min(d0_candidates)
                d1 = min(d1_candidates)
                LLRs_flat[k] = (d1 - d0) / noise_var
            else:
                # missing one side → set to zero
                LLRs_flat[k] = 0.0

        # Store outputs
        LLRs_all[idx] = LLRs_flat.reshape(n_users, bits_per_symbol)
        hard_bits_all[idx] = best["bits"].reshape(n_users, bits_per_symbol)
        # end = time.time()
        # print(f"SphereDecoder took {end - start:.4f} seconds")


    # Final reshape
    LLRs_all = LLRs_all.transpose(0, 2, 1).reshape(n_symbols * bits_per_symbol, n_users)
    hard_bits_all = hard_bits_all.transpose(0, 2, 1).reshape(n_symbols * bits_per_symbol, n_users)

    return LLRs_all, hard_bits_all