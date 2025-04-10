import torch

from python_code import DEVICE
from python_code.utils.constants import HALF


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

