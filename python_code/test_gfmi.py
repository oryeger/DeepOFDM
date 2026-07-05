"""Tests for calc_gfmi. Run as a plain script: python python_code/test_gfmi.py"""
import sys
import numpy as np
from scipy.special import expit as sigmoid

# ---------------------------------------------------------------------------
# Inline the estimator so this test has no dependency on the evaluate module
# (which requires a GPU environment and the full conf singleton).
# ---------------------------------------------------------------------------
def calc_gfmi(llrs) -> float:
    a = np.abs(np.asarray(llrs).flatten().astype(np.float64))
    finite_mask = np.isfinite(a)
    a = a[finite_mask]
    if a.size == 0:
        return 0.0
    term1 = np.log1p(np.exp(-a)) / np.log(2)
    term2 = (a / np.log(2)) * sigmoid(-a)
    return float(np.clip(1.0 - np.mean(term1 + term2), 0.0, 1.0))


def genie_mi(bits, llrs) -> float:
    """Reference: genie MI using transmitted bits."""
    b = np.asarray(bits).flatten()
    L = np.asarray(llrs).flatten().astype(np.float64)
    signs = 1 - 2 * b  # b=0 → +1, b=1 → -1
    # log2(1 + exp(-(1-2b)*L)), clipped for safety
    vals = np.log1p(np.exp(-signs * L)) / np.log(2)
    return float(max(1.0 - np.mean(vals), 0.0))


PASS = 0
FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        print(f'  PASS  {name}')
        PASS += 1
    else:
        print(f'  FAIL  {name}' + (f'  ({detail})' if detail else ''))
        FAIL += 1


# ---------------------------------------------------------------------------
# Test 1: GFMI(zeros) == 0.0 exactly
# ---------------------------------------------------------------------------
print('Test 1: zero LLRs -> MI = 0')
result = calc_gfmi(np.zeros(10000))
check('GFMI(zeros) == 0.0', result == 0.0, f'got {result}')

# ---------------------------------------------------------------------------
# Test 2: large |L| → MI ≈ 1
# ---------------------------------------------------------------------------
print('Test 2: |L|=100 -> MI ~= 1')
result = calc_gfmi(np.full(10000, 100.0))
check('GFMI(100) within 1e-6 of 1.0', abs(result - 1.0) < 1e-6, f'got {result}')

# ---------------------------------------------------------------------------
# Test 3: consistency with genie MI
# ---------------------------------------------------------------------------
print('Test 3: consistency with genie MI for calibrated Gaussian LLRs'  )
rng = np.random.default_rng(42)
N = 1_000_000
for mu in (0.5, 2.0, 8.0):
    bits = rng.integers(0, 2, size=N)
    # L ~ N((1-2b)*mu, 2*mu)  →  consistent Gaussian mixture
    means = (1 - 2 * bits) * mu
    L = rng.normal(means, np.sqrt(2 * mu))
    gfmi_val = calc_gfmi(L)
    genie_val = genie_mi(bits, L)
    diff = abs(gfmi_val - genie_val)
    check(f'mu={mu}: |GFMI - genie_MI| < 0.01  (GFMI={gfmi_val:.4f}, genie={genie_val:.4f})',
          diff < 0.01, f'diff={diff:.4f}')

# ---------------------------------------------------------------------------
# Test 4: scale sensitivity — GFMI(3L) > GFMI(L) > GFMI(0.3L)
# ---------------------------------------------------------------------------
print('Test 4: scale sensitivity')
rng2 = np.random.default_rng(7)
mu = 2.0
bits4 = rng2.integers(0, 2, size=100_000)
means4 = (1 - 2 * bits4) * mu
L_base = rng2.normal(means4, np.sqrt(2 * mu))
g_low  = calc_gfmi(0.3 * L_base)
g_mid  = calc_gfmi(L_base)
g_high = calc_gfmi(3.0 * L_base)
check('GFMI(3L) > GFMI(L)', g_high > g_mid, f'{g_high:.4f} vs {g_mid:.4f}')
check('GFMI(L) > GFMI(0.3L)', g_mid > g_low, f'{g_mid:.4f} vs {g_low:.4f}')

# ---------------------------------------------------------------------------
print()
print(f'Results: {PASS} passed, {FAIL} failed')
sys.exit(0 if FAIL == 0 else 1)
