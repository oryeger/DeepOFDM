#!/usr/bin/env python3
import numpy as np
import pandas as pd

# ---- IMPORTANT ----
# For square 256-QAM with levels ±1,±3,...,±15 (unnormalized),
# the average symbol energy is Es = 170.
# If commpy uses unit-average-power normalization, multiplying by sqrt(170)
# de-normalizes back to the integer grid.
# DENORM = np.sqrt(170.0)   # set to 1.0 if you want the raw commpy output as-is
DENORM = 1.0   # set to 1.0 if you want the raw commpy output as-is

NBITS = 8
M = 256

def bits_from_int(k: int, nbits: int = NBITS) -> np.ndarray:
    """MSB-first bit order: 0 -> '00000000', 255 -> '11111111'."""
    return np.array([(k >> (nbits - 1 - i)) & 1 for i in range(nbits)], dtype=int)

def cfmt(z: complex) -> str:
    """Pretty complex printing without scientific noise."""
    a = float(np.real(z))
    b = float(np.imag(z))
    if abs(a) < 1e-12: a = 0.0
    if abs(b) < 1e-12: b = 0.0
    sign = "+" if b >= 0 else "-"
    return f"{a:g}{sign}{abs(b):g}j"

def main():
    # --- create commpy modem ---
    # pip install commpy  (package name is often "scikit-commpy")
    from commpy.modulation import QAMModem
    qam = QAMModem(M)

    rows = []
    for idx in range(M):
        bits = bits_from_int(idx, NBITS)

        # commpy expects array-like bits
        sym = qam.modulate(bits)
        sym = np.asarray(sym).reshape(-1)[0]  # scalar

        # De-normalize display (remove constellation normalization)
        sym_dn = sym * DENORM

        I = float(np.real(sym_dn))
        Q = float(np.imag(sym_dn))

        # If it is supposed to be integer grid, round tiny numerical errors
        I_disp = int(np.round(I)) if abs(I - np.round(I)) < 1e-9 else I
        Q_disp = int(np.round(Q)) if abs(Q - np.round(Q)) < 1e-9 else Q

        rows.append({
            "index": idx,
            "bits_int": idx,
            "bits": "".join(map(str, bits.tolist())),
            "I": I_disp,
            "Q": Q_disp,
            "symbol": cfmt(sym_dn),
        })

    df = pd.DataFrame(rows, columns=["index", "bits_int", "bits", "I", "Q", "symbol"])

    # Print full table (all 256 rows)
    print(df.to_string(index=False))

    # Save to CSV for easy inspection
    out_csv = "qam256_mapping_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Helpful diagnostics
    maxI = np.max(np.abs(df["I"].astype(float).values))
    maxQ = np.max(np.abs(df["Q"].astype(float).values))
    print(f"Max |I|: {maxI}, Max |Q|: {maxQ}")
    print(f"DENORM used: {DENORM}")

if __name__ == "__main__":
    main()
