import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

import numpy as np
from itertools import product
import commpy.modulation as mod


# 16-QAM constellation points (normalized to unit energy)
qam16 = np.array([-3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
                  -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
                  1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j,
                  3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j]) / np.sqrt(10)

import numpy as np
from itertools import product


import numpy as np
from itertools import product

def SphereDecoder(H, y, noise_var=1.0):
    """
    Brute-force ML detector with LLR computation (multi-user MIMO).
    Each user has 1 spatial layer.

    Args:
        H: (N_rx x N_users) channel matrix
        y: (N_symbols x N_rx) received signals
        noise_var: noise variance

    Returns:
        LLRs_all: (N_symbols, N_users, bits_per_symbol)
        hard_bits_all: (N_symbols, N_users, bits_per_symbol)
    """
    n_symbols, n_rx = y.shape
    n_users = H.shape[1]
    bits_per_symbol = int(np.log2(conf.mod_pilot))
    qam = mod.QAMModem(conf.mod_pilot)

    # constellation and bit mapping
    constellation = qam.constellation  # ordered complex symbols
    bit_combinations = np.array(list(product([0, 1], repeat=bits_per_symbol)), dtype=np.int64)

    # Precompute QR
    Q, R = np.linalg.qr(H, mode='reduced')

    # Allocate outputs
    LLRs_all = np.zeros((n_symbols, n_users, bits_per_symbol))
    hard_bits_all = np.zeros((n_symbols, n_users, bits_per_symbol), dtype=int)

    # Loop over all received symbols
    for idx in range(n_symbols):
        y_tilde = Q.conj().T @ y[idx, :]

        candidates_metrics = []
        best_dist = np.inf
        best_bits = None

        # Enumerate all candidate symbol vectors (brute force)
        for symbol_tuple in product(constellation, repeat=n_users):
            x_vec = np.array(symbol_tuple)
            dist = np.linalg.norm(y_tilde - R @ x_vec) ** 2

            # map each symbol to its bit pattern
            bits = np.concatenate([
                bit_combinations[np.where(constellation == s)[0][0]] for s in x_vec
            ])

            candidates_metrics.append((bits, dist))

            # Track ML solution (hard decision)
            if dist < best_dist:
                best_dist = dist
                best_bits = bits

        # --- Compute LLRs (Max-Log-MAP) ---
        n_bits_total = n_users * bits_per_symbol
        LLRs_flat = np.zeros(n_bits_total)
        for k in range(n_bits_total):
            d0 = min(dist for bits, dist in candidates_metrics if bits[k] == 0)
            d1 = min(dist for bits, dist in candidates_metrics if bits[k] == 1)
            LLRs_flat[k] = (d1 - d0) / noise_var

        # reshape into (n_users, bits_per_symbol)
        LLRs_all[idx] = LLRs_flat.reshape(n_users, bits_per_symbol)
        hard_bits_all[idx] = best_bits.reshape(n_users, bits_per_symbol)

    LLRs_all = LLRs_all.transpose(0, 2, 1).reshape(n_symbols * bits_per_symbol, n_users)
    hard_bits_all = hard_bits_all.transpose(0, 2, 1).reshape(n_symbols * bits_per_symbol, n_users)

    return LLRs_all, hard_bits_all
