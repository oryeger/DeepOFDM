from typing import Any

import numpy as np
import torch
from numpy import ndarray, dtype, complexfloating
from numpy._typing import _64Bit
from numpy.core.multiarray import concatenate

from python_code.utils.constants import HALF


class BPSKModulator:
    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        BPSK modulation 0->1, 1->-1
        :param c: the binary codeword
        :return: binary modulated signal
        """
        x_real = (1 - 2 * c)
        x = x_real.astype(np.complex128)
        return x

    @staticmethod
    def demodulate(s: torch.Tensor) -> torch.Tensor:
        """
        symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
        Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'
        :param s: symbols vector
        :return: probabilities vector
        """
        return HALF * (s + 1)

class QPSKModulator:
    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        QPSK modulation 0->1, 1->-1
        :param c: the binary codeword
        :return: binary modulated signal
        """
        real_vector = (1 - 2 * c[0])
        imaginary_vector = (1 - 2 * c[1])
        complex_array = 2 ** 0.5 *(real_vector + 1j * imaginary_vector)
        x = complex_array.astype(np.complex128)
        return x

    @staticmethod
    def demodulate(s: torch.Tensor) -> torch.Tensor:
        """
        symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
        Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'
        :param s: symbols vector
        :return: probabilities vector
        """

        real_part = np.real(s)
        imag_part = np.imag(s)

        # Decision rule for In-phase (c1) and Quadrature-phase (c2)
        c1 = 0 if real_part >= 0 else 1
        c2 = 0 if imag_part >= 0 else 1
        x = np.concatenate([c1,c2])

        return x
