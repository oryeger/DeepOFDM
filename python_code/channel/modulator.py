from typing import Any

import numpy as np
import torch
from typing import Tuple

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
    def demodulate(s: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
        Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'
        :param s: symbols vector
        :return: probabilities vector
        """

        real_part = np.real(s)
        imag_part = np.imag(s)

        # Decision rule for In-phase (c1) and Quadrature-phase (c2)
        x_real = HALF * (np.sign(real_part) + 1)
        x_imag = HALF * (np.sign(imag_part) + 1)

        x = np.ravel(np.column_stack((x_real, x_imag)))
        LLRs = np.ravel(np.column_stack((real_part, imag_part)))


        return x,LLRs


class QAM16Modulator:
    @staticmethod
    def demodulate(s: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
        Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'
        :param s: symbols vector
        :return: probabilities vector
        """

        real_part_0 = np.real(s)
        real_part_1 = 2-np.abs(np.real(s))
        imag_part_0 = np.imag(s)
        imag_part_1 = 2-np.abs(np.imag(s))

        # Decision rule for In-phase (c1) and Quadrature-phase (c2)
        x_real_0 = HALF * (np.sign(real_part_0) + 1)
        x_real_1 = HALF * (np.sign(real_part_1) + 1)
        x_imag_0 = HALF * (np.sign(imag_part_0) + 1)
        x_imag_1 = HALF * (np.sign(imag_part_1) + 1)

        x = np.ravel(np.column_stack((x_real_0, x_real_1,x_imag_0, x_imag_1)))
        LLRs = np.ravel(np.column_stack((real_part_0, real_part_1,imag_part_0, imag_part_1)))

        return x,LLRs
