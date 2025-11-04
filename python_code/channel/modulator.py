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


class QAM64Modulator:
    @staticmethod
    def _D1_64qam(y: np.ndarray) -> np.ndarray:
        """First (MSB) soft metric per component (I or Q)."""
        D = np.empty_like(y, dtype=float)

        m = (np.abs(y) <= 2)
        D[m] = y[m]

        m = (y > 2) & (y <= 4)
        D[m] = 2.0 * (y[m] - 1.0)
        m = (y > 4) & (y <= 6)
        D[m] = 3.0 * (y[m] - 2.0)
        m = (y > 6)
        D[m] = 4.0 * (y[m] - 3.0)

        m = (y >= -4) & (y < -2)
        D[m] = 2.0 * (y[m] + 1.0)
        m = (y >= -6) & (y < -4)
        D[m] = 3.0 * (y[m] + 2.0)
        m = (y < -6)
        D[m] = 4.0 * (y[m] + 3.0)

        return D

    @staticmethod
    def _D2_64qam(y: np.ndarray) -> np.ndarray:
        """Second soft metric per component (I or Q)."""
        a = np.abs(y)
        D = np.empty_like(y, dtype=float)

        m = (a <= 2)
        D[m] = 2.0 * (-a[m] + 3.0)
        m = (a > 2) & (a <= 6)
        D[m] = 4.0 - a[m]
        m = (a > 6)
        D[m] = 2.0 * (-a[m] + 5.0)

        return D

    @staticmethod
    def _D3_64qam(y: np.ndarray) -> np.ndarray:
        """Third (LSB) soft metric per component (I or Q)."""
        a = np.abs(y)
        D = np.empty_like(y, dtype=float)

        m = (a <= 4)
        D[m] = a[m] - 2.0
        m = (a > 4)
        D[m] = -a[m] + 6.0

        return D

    @staticmethod
    def demodulate(s: np.ndarray, denorm: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        64-QAM soft demodulation.
        :param s: complex symbol vector
        :param denorm: scaling factor (e.g., sqrt(42) for normalized constellation)
        :return: (hard bits, LLRs)
        """
        yI = np.real(s) * denorm
        yQ = np.imag(s) * denorm

        DI1, DI2, DI3 = (
            QAM64Modulator._D1_64qam(yI),
            QAM64Modulator._D2_64qam(yI),
            QAM64Modulator._D3_64qam(yI),
        )
        DQ1, DQ2, DQ3 = (
            QAM64Modulator._D1_64qam(yQ),
            QAM64Modulator._D2_64qam(yQ),
            QAM64Modulator._D3_64qam(yQ),
        )

        # Hard decisions
        xI1 = HALF * (np.sign(DI1) + 1)
        xI2 = HALF * (np.sign(DI2) + 1)
        xI3 = HALF * (np.sign(DI3) + 1)
        xQ1 = HALF * (np.sign(DQ1) + 1)
        xQ2 = HALF * (np.sign(DQ2) + 1)
        xQ3 = HALF * (np.sign(DQ3) + 1)

        x = np.ravel(np.column_stack((xI1, xI2, xI3, xQ1, xQ2, xQ3)))
        LLRs = np.ravel(np.column_stack((DI1, DI2, DI3, DQ1, DQ2, DQ3)))

        return x, LLRs