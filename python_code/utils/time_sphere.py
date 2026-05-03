"""
Timing test for the full SphereDecoder on 16-QAM.

Generates synthetic channel inputs sized by config.yaml and times the decoder
across `num_res` resource elements. Run from project root:

    python -m python_code.utils.time_sphere
"""
import time
import numpy as np
import commpy.modulation as mod

from python_code import conf
from python_code.detectors.sphere.sphere_decoder import SphereDecoder


N_SYMBOLS = 10
NUM_RES = 24

# Force 16-QAM. mcs=5 -> Qm=4 per mcs_table.py
ORIG_MCS = conf.mcs
conf.mcs = 5

n_ants = conf.n_ants
n_users = conf.n_users
radius = conf.sphere_radius
snr_db = conf.snr

qam = mod.QAMModem(16)
constellation = qam.constellation.astype(np.complex128)
es = float(np.mean(np.abs(constellation) ** 2))
noise_var = es / (10.0 ** (snr_db / 10.0))


def make_inputs(re_seed):
    """Random Rayleigh channel and 16-QAM transmit + AWGN observations."""
    r = np.random.default_rng(re_seed)
    H = (r.standard_normal((n_ants, n_users)) + 1j * r.standard_normal((n_ants, n_users))) / np.sqrt(2.0)
    tx_idx = r.integers(0, 16, size=(N_SYMBOLS, n_users))
    x = constellation[tx_idx]
    noise = (r.standard_normal((N_SYMBOLS, n_ants)) + 1j * r.standard_normal((N_SYMBOLS, n_ants))) * np.sqrt(noise_var / 2.0)
    y = x @ H.T + noise
    return H, y


inputs = [make_inputs(re) for re in range(NUM_RES)]

print(f"[CONFIG] n_ants={n_ants} n_users={n_users} mcs={conf.mcs} (orig={ORIG_MCS}) "
      f"snr={snr_db} dB radius={radius}")
print(f"[INPUTS] n_symbols={N_SYMBOLS} num_res={NUM_RES} -> {N_SYMBOLS * NUM_RES} symbol-decodes")

t0 = time.perf_counter()
for H, y in inputs:
    SphereDecoder(H, y, noise_var, radius)
t_full = time.perf_counter() - t0

n_decodes = NUM_RES * N_SYMBOLS
print(f"[TIME] SphereDecoder total: {t_full:.4f} s  ({t_full / NUM_RES * 1000:.2f} ms/RE, {t_full / n_decodes * 1000:.3f} ms/symbol)")
