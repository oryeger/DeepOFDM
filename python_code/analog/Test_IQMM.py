import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Parameters
# ------------------------------
N = 24         # number of subcarriers
cp_len = 9     # cyclic prefix length
gain_mismatch_dB = 1   # gain mismatch in dB
phi = np.deg2rad(0)   # phase mismatch in radians

# ------------------------------
# Generate random QPSK symbols
# ------------------------------
data_orig = (2*np.random.randint(0, 2, N) - 1) + 1j*(2*np.random.randint(0, 2, N) - 1)

index = 3
data = np.zeros_like(data_orig)
data[index] = data_orig[index]

# data = data_orig


# ------------------------------
# OFDM modulation (IFFT + CP)
# ------------------------------
ofdm_time = np.fft.ifft(data, 128)
ofdm_time_cp = np.hstack([ofdm_time[-cp_len:], ofdm_time])

# ------------------------------
# Apply IQ imbalance (time domain)
# ------------------------------
epsilon = 10 ** (gain_mismatch_dB / 20.0) - 1
alpha = np.cos(phi / 2) + 1j * epsilon * np.sin(phi / 2)
beta = epsilon * np.cos(phi / 2) - 1j * np.sin(phi / 2)

rx_time = alpha * ofdm_time_cp + beta * np.conj(ofdm_time_cp)

# ------------------------------
# OFDM demodulation (remove CP + FFT)
# ------------------------------
rx_no_cp = rx_time[cp_len:]
rx_freq = np.fft.fft(rx_no_cp, 128)

# ------------------------------
# Plot results
# ------------------------------
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# axs[0].stem(np.arange(128), np.abs(data))
# axs[0].set_title("Original OFDM symbols (magnitude)")
# axs[0].set_xlabel("Subcarrier index")
# axs[0].set_ylabel("Magnitude")

axs[1].stem(np.arange(128), np.abs(rx_freq))
axs[1].set_title("After IQ imbalance (magnitude)")
axs[1].set_xlabel("Subcarrier index")
axs[1].set_ylabel("Magnitude")

plt.tight_layout()
plt.show()
