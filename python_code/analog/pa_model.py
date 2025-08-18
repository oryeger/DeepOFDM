import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

def memory_polynomial_model(x, coeffs, orders, M):
    """
    Simulate memory polynomial model output.
    """
    N = len(x)
    y = np.zeros(N, dtype=complex)

    for i, p in enumerate(orders):
        for m in range(M):
            x_del = np.concatenate([np.zeros(m, dtype=complex), x[:N-m]])
            y += coeffs[i, m] * x_del * (np.abs(x_del) ** (p - 1))

    return y


def fit_memory_polynomial(x, y, orders=[1,3,5,7], M=3, lam=1e-6):
    """
    Fit memory polynomial model using LS.
    """
    N = len(x)
    Phi = []

    for p in orders:
        for m in range(M):
            x_del = np.concatenate([np.zeros(m, dtype=complex), x[:N-m]])
            Phi.append(x_del * (np.abs(x_del) ** (p - 1)))

    Phi = np.stack(Phi, axis=1)  # (N, P*M)

    # Ridge regularization
    A = Phi.conj().T @ Phi + lam * np.eye(Phi.shape[1])
    b = Phi.conj().T @ y
    c = np.linalg.solve(A, b)

    coeffs = c.reshape(len(orders), M)
    return coeffs


def plot_amam_ampm(x, y, n_bins=80, title="PA Characteristics"):
    """
    Plot AM/AM and AM/PM curves.
    Args:
        x, y : input and output signals (complex arrays)
        n_bins : number of bins for averaging
    """
    r_in = np.abs(x)
    r_out = np.abs(y)
    phi_in = np.angle(x)
    phi_out = np.angle(y)
    dphi = np.unwrap(phi_out) - np.unwrap(phi_in)

    # Bin average for smoothness
    bins = np.linspace(0, r_in.max(), n_bins)
    idx = np.digitize(r_in, bins)
    amam = [r_out[idx == i].mean() if np.any(idx == i) else np.nan for i in range(1, len(bins))]
    ampm = [dphi[idx == i].mean() if np.any(idx == i) else np.nan for i in range(1, len(bins))]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(bins[1:], amam, "o-")
    ax[0].set_title("AM/AM")
    ax[0].set_xlabel("Input amplitude")
    ax[0].set_ylabel("Output amplitude")

    ax[1].plot(bins[1:], np.rad2deg(ampm), "o-")
    ax[1].set_title("AM/PM")
    ax[1].set_xlabel("Input amplitude")
    ax[1].set_ylabel("Phase shift (deg)")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------- Example ----------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)

    N = 40000
    x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)

    # Define a "true" PA
    orders_true = [1,3,5]
    M_true = 2
    coeffs_true = np.array([
        [1.0+0j, 0.2-0.1j],          # linear
        [0.05+0.02j, -0.02+0.03j],   # cubic
        [-0.01+0j, 0.005+0.005j]     # 5th
    ])

    y = memory_polynomial_model(x, coeffs_true, orders_true, M_true)
    y_noisy = y + 0.01 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))

    coeffs_est = fit_memory_polynomial(x, y_noisy, orders_true, M_true)

    # Simulate with estimated model
    y_est = memory_polynomial_model(x, coeffs_est, orders_true, M_true)

    # Plot
    plot_amam_ampm(x, y, title="True PA Characteristics")
    plot_amam_ampm(x, y_est, title="Estimated PA Characteristics")
