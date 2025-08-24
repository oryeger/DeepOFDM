import numpy as np

def apply_iq_mismatch(signal, gain_mismatch_dB=0.0, phi=0.0):
    """
    Apply IQ imbalance impairment using alpha-beta model:
        y = alpha * x + beta * conj(x)

    Args:
        signal : complex ndarray
            Input baseband signal
        gain_mismatch_dB : float
            Gain mismatch between I and Q branches in dB (e.g. 0.5 dB)
        phi : float
            Phase mismatch in radians (e.g. 5° = np.deg2rad(5))

    Returns:
        distorted : complex ndarray
            Output signal with IQ imbalance
    """
    # Convert dB mismatch to linear epsilon
    epsilon = 10 ** (gain_mismatch_dB / 20.0) - 1

    alpha = np.cos(phi / 2) + 1j * epsilon * np.sin(phi / 2)
    beta = epsilon * np.cos(phi / 2) - 1j * np.sin(phi / 2)

    return alpha * signal + beta * np.conj(signal)