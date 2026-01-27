from __future__ import annotations

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

    """
    qam256_demapper.py

    256-QAM (square) soft demapper aligned to the exact mapping you printed from:
        qam.modulate(bits)

    Observed mapping (per axis) is 16-PAM Gray-coded with levels:
        ±1, ±3, ±5, ±7, ±9, ±11, ±13, ±15
    and nibble -> amplitude order:
        0000,0001,0011,0010,0110,0111,0101,0100,1100,1101,1111,1110,1010,1011,1001,1000

    Bit ordering (per symbol):
        bits[0:4] -> I nibble (bI1,bI2,bI3,bI4)
        bits[4:8] -> Q nibble (bQ1,bQ2,bQ3,bQ4)

    Hard bits are extracted as:
        bit = 0.5*(sign(Dk) + 1)

    denorm:
        - If your symbols are normalized (unit average power), use denorm = sqrt(170).
        - If your symbols already live on the integer grid (±1..±15), use denorm = 1.0.
    """

class QAM256Modulator:
    # ---------------- Per-axis soft metrics (I or Q) ---------------- #

    @staticmethod
    def _D1_256qam(y: np.ndarray) -> np.ndarray:
        """
        Bit-1 (MSB of the 4-bit nibble): sign bit.
        Mapping: negative -> 0, positive -> 1.
        Boundary at 0. Soft metric is simply y.
        """
        return np.asarray(y, dtype=float)

    @staticmethod
    def _D2_256qam(y: np.ndarray) -> np.ndarray:
        """
        Bit-2: inner vs outer magnitude.

        For the printed mapping:
            bit2 = 1 for |y| < 8   (inner: 1,3,5,7)
            bit2 = 0 for |y| > 8   (outer: 9,11,13,15)

        Boundary at |y| = 8.
        Soft metric positive inside, negative outside:
            D2 = 8 - |y|
        """
        y = np.asarray(y, dtype=float)
        return 8.0 - np.abs(y)

    @staticmethod
    def _D3_256qam(y: np.ndarray) -> np.ndarray:
        """
        Bit-3: mid ring vs extremes/center.

        For the printed mapping:
            bit3 = 1 for 4 < |y| < 12   (5,7,9,11)
            bit3 = 0 for |y| < 4 or |y| > 12  (1,3,13,15)

        Boundaries at |y| = 4 and |y| = 12.
        A simple triangular metric with correct sign:
            D3 = 4 - ||y| - 8|
        """
        y = np.asarray(y, dtype=float)
        a = np.abs(y)
        return 4.0 - np.abs(a - 8.0)

    @staticmethod
    def _D4_256qam(y: np.ndarray) -> np.ndarray:
        """
        Bit-4 (LSB of the nibble) for the printed Gray mapping.

        Over positive amplitudes |y| ∈ {1,3,5,7,9,11,13,15} the bit4 pattern is:
            |y|:  1  3  5  7  9  11  13  15
            b4 :  0  1  1  0  0   1   1   0

        So b4=1 in the magnitude bands:
            2 <= |y| < 6   (covers 3,5)
            10 <= |y| < 14 (covers 11,13)
        else b4=0.

        Soft metric: signed distance to the nearest boundary in {2, 6, 10, 14}
        with positive sign inside the b4=1 bands, negative otherwise.
        """
        y = np.asarray(y, dtype=float)
        a = np.abs(y)

        # distance to nearest flip boundary
        d = np.minimum.reduce([
            np.abs(a - 2.0),
            np.abs(a - 6.0),
            np.abs(a - 10.0),
            np.abs(a - 14.0),
        ])

        in_band = ((a >= 2.0) & (a < 6.0)) | ((a >= 10.0) & (a < 14.0))
        sign = np.where(in_band, 1.0, -1.0)
        return sign * d

    # ---------------- Full 256-QAM soft demodulation ---------------- #

    @staticmethod
    def demodulate(s: np.ndarray, denorm: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        256-QAM soft demodulation aligned to the commpy mapping you printed.

        Parameters
        ----------
        s : np.ndarray
            Complex symbol vector/array.
        denorm : float
            Scaling factor to bring s onto the integer grid (±1..±15 per axis).
            Use sqrt(170) if s is normalized, else 1.0.

        Returns
        -------
        x : np.ndarray
            Hard bits (0/1) as a flat vector in the order:
            [bI1,bI2,bI3,bI4,bQ1,bQ2,bQ3,bQ4] per symbol (raveled).
        LLRs : np.ndarray
            Soft metrics in the same ordering (raveled).
        """
        s = np.asarray(s)
        yI = np.real(s) * denorm
        yQ = np.imag(s) * denorm

        DI1 = QAM256Modulator._D1_256qam(yI)
        DI2 = QAM256Modulator._D2_256qam(yI)
        DI3 = QAM256Modulator._D3_256qam(yI)
        DI4 = QAM256Modulator._D4_256qam(yI)

        DQ1 = QAM256Modulator._D1_256qam(yQ)
        DQ2 = QAM256Modulator._D2_256qam(yQ)
        DQ3 = QAM256Modulator._D3_256qam(yQ)
        DQ4 = QAM256Modulator._D4_256qam(yQ)

        # Hard decisions (0/1) from the sign of each metric
        xI1 = HALF * (np.sign(DI1) + 1)
        xI2 = HALF * (np.sign(DI2) + 1)
        xI3 = HALF * (np.sign(DI3) + 1)
        xI4 = HALF * (np.sign(DI4) + 1)

        xQ1 = HALF * (np.sign(DQ1) + 1)
        xQ2 = HALF * (np.sign(DQ2) + 1)
        xQ3 = HALF * (np.sign(DQ3) + 1)
        xQ4 = HALF * (np.sign(DQ4) + 1)

        # Pack per symbol: I bits then Q bits
        x = np.ravel(np.column_stack((xI1, xI2, xI3, xI4, xQ1, xQ2, xQ3, xQ4)))
        LLRs = np.ravel(np.column_stack((DI1, DI2, DI3, DI4, DQ1, DQ2, DQ3, DQ4)))

        return x, LLRs

