import torch

from python_code import DEVICE
from python_code.utils.constants import HALF
import numpy as np


def calculate_mimo_states(n_user: int, transmitted_words: torch.Tensor) -> torch.Tensor:
    """
    Calculates mimo states vector for the transmitted words. Number of states is 2/4/8 ** n_user depending on the
    constellation size.
    """
    states_enumerator = (2 ** torch.arange(n_user)).to(DEVICE)
    gt_states = torch.sum(transmitted_words * states_enumerator, dim=1).long()
    return gt_states


def prob_to_BPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,1] -> '+1'
    """
    return torch.sign(p - HALF)

def prob_to_QAM_index(p: torch.Tensor, num_bits: int, n_users: int) -> torch.Tensor:
    output_symbols = torch.empty(p.size(0),n_users)
    for user in range(n_users):
        user_indexes = [int(user * (num_bits)) + i for i in range(0, num_bits)]
        cur_user_p = p[:,user_indexes]
        sum_of_columns = cur_user_p.sum(dim=1, keepdim=True)
        fourth_column = 1 - sum_of_columns
        cur_user_p_out = torch.cat((cur_user_p, fourth_column), dim=1)
        output_symbols[:,user] = torch.argmax(cur_user_p_out, dim=1)
    return output_symbols


def ensure_tensor_iterable(x):
    if isinstance(x, torch.Tensor):
        return x.flatten()
    elif isinstance(x, int):
        return torch.tensor([x])
    else:
        raise TypeError("Input must be an int or a torch.Tensor")


import numpy as np

def relevant_indices(N, pilot_data_ratio, is_256qam=False):
    """
    Returns indices of the bits that are explicitly computed.

    Default (is_256qam=False):
        Original ratio-based behavior (backward compatible).

    256-QAM mode (is_256qam=True):
        Uses fixed pattern per 8-bit symbol:
            [0, 3, 4, 7] + 8*k
        Requires pilot_data_ratio = 2 (8 bits / 4 computed bits).
    """

    if not is_256qam:
        # ---- original behavior ----
        x = np.arange(0, N, pilot_data_ratio)
        idx = np.floor(x + 0.5).astype(int)   # round-half-up
        idx = idx[idx < N]
        return np.unique(idx)

    # ---- 256-QAM special behavior ----
    if abs(float(pilot_data_ratio) - 2.0) > 1e-12:
        raise ValueError(
            f"is_256qam=True expects pilot_data_ratio=2, got {pilot_data_ratio}"
        )

    keep = np.array([0, 3, 4, 7], dtype=int)
    bps = 8

    n_syms = int(np.ceil(N / bps))
    base = (np.arange(n_syms, dtype=int) * bps)[:, None]
    idx = (base + keep[None, :]).ravel()
    idx = idx[idx < N]

    return np.unique(idx)


def skip_indices(N, pilot_data_ratio, is_256qam=False):
    """
    Returns indices of the bits that are NOT explicitly computed.
    """
    rel = relevant_indices(N, pilot_data_ratio, is_256qam)
    all_idx = np.arange(N, dtype=int)
    return np.setdiff1d(all_idx, rel, assume_unique=False)


def get_64QAM_16QAM_indices_and_probs(total_bits, bits_per_symbol=6):
    """
    For 64QAM with make_64QAM_16QAM_percentage=50:
    Divides the bits into three equal parts (each being a multiple of bits_per_symbol):
    - First third: skip_indices with ratio 3 (constant 1 symbols)
    - Second third: skip_indices with ratio 1.5 (16QAM symbols with prob 0.5)
    - Last third: unchanged (full 64QAM)

    Returns:
        tuple: (first_third_indices, second_third_indices, first_third_size, second_third_size, third_size)
    """
    # Ensure total_bits is divisible by bits_per_symbol
    total_symbols = total_bits // bits_per_symbol

    # Divide symbols into three equal parts (or as equal as possible)
    symbols_per_third = total_symbols // 3
    remaining_symbols = total_symbols % 3

    # Distribute remaining symbols: give extras to the last third
    first_third_symbols = symbols_per_third
    second_third_symbols = symbols_per_third
    third_symbols = symbols_per_third + remaining_symbols

    # Convert to bit counts
    first_third_size = first_third_symbols * bits_per_symbol
    second_third_size = second_third_symbols * bits_per_symbol
    third_size = third_symbols * bits_per_symbol

    # Get skip indices for first third (ratio 3)
    first_third_indices = skip_indices(first_third_size, 3)

    # Get skip indices for second third (ratio 1.5), offset by first_third_size
    second_third_indices_relative = skip_indices(second_third_size, 1.5)
    second_third_indices = second_third_indices_relative + first_third_size

    return first_third_indices, second_third_indices, first_third_size, second_third_size, third_size


