"""theory.py - Information-theoretic benchmarks (capacity, dispersion, normal approximation).

All quantities are in bits.  Nothing here is hard-coded: limits such as the
BI-AWGN Shannon limit or the BSC threshold are computed numerically.
"""
import math

import numpy as np

LN2 = math.log(2.0)
_GH_T, _GH_W = np.polynomial.hermite.hermgauss(200)   # nodes/weights for int e^{-t^2} f(t) dt


# ---------------------------------------------------------------- basic functions
def h2(p):
    """Binary entropy h2(p) = -p log2 p - (1-p) log2(1-p), with h2(0) = h2(1) = 0."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    out = np.where((p <= 0) | (p >= 1), 0.0, out)
    return float(out) if out.ndim == 0 else out


def Q(x):
    """Gaussian tail Q(x) = P(N(0,1) > x) = erfc(x / sqrt 2) / 2."""
    if np.ndim(x) == 0:
        return 0.5 * math.erfc(float(x) / math.sqrt(2.0))
    return np.array([0.5 * math.erfc(v / math.sqrt(2.0)) for v in np.ravel(x)]).reshape(np.shape(x))


def Qinv(eps):
    """Inverse Gaussian tail: the x with Q(x) = eps, found by bisection (Q is strictly decreasing)."""
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie in (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Q(mid) > eps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- BSC
def bsc_capacity(p):
    """Capacity of the binary symmetric channel: C = 1 - h2(p) bits per channel use."""
    return 1.0 - h2(p)


def bsc_dispersion(p):
    """Channel dispersion of BSC(p) (Polyanskiy-Poor-Verdu 2010): V = p(1-p) log2^2((1-p)/p)."""
    return p * (1 - p) * math.log2((1 - p) / p) ** 2


def bsc_threshold(R):
    """Largest crossover probability p in (0, 1/2) with 1 - h2(p) >= R (bisection)."""
    lo, hi = 1e-15, 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bsc_capacity(mid) > R:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normal_approx_rate(n, eps, C, V):
    """PPV normal approximation R*(n, eps) ~ C - sqrt(V/n) Qinv(eps) + log2(n)/(2n)."""
    return C - math.sqrt(V / n) * Qinv(eps) + math.log2(n) / (2 * n)


def normal_approx_fer(n, R, C, V):
    """Inverse normal approximation: FER of an ideal (n, R) code, eps ~ Q((C - R + log2(n)/(2n)) / sqrt(V/n)).

    This is an approximation (not a bound) and is least accurate at very short n.
    """
    if V <= 0:
        return 0.0 if R < C else 1.0
    return Q((C - R + math.log2(n) / (2 * n)) / math.sqrt(V / n))


def bsc_normal_approx_fer(p, n, R):
    """Normal-approximation FER of the best (n, R) code over BSC(p)."""
    return normal_approx_fer(n, R, bsc_capacity(p), bsc_dispersion(p))


# ---------------------------------------------------------------- BI-AWGN (BPSK)
def ebn0_to_sigma2(ebn0_db, R):
    """Noise variance for unit-energy BPSK: sigma^2 = 1 / (2 R Eb/N0), Eb/N0 in dB, R = info bits per channel use."""
    return 1.0 / (2.0 * R * 10.0 ** (ebn0_db / 10.0))


def _biawgn_info_density(sigma2):
    """Information density samples i(x;y) = 1 - log2(1 + exp(-2y/sigma^2)) at Gauss-Hermite nodes for x = +1.

    With y = 1 + sqrt(2) sigma t, the expectation E[f(Y)] = (1/sqrt(pi)) sum_k w_k f(y_k).
    By symmetry of the BI-AWGN channel the x = -1 case gives the same distribution.
    """
    y = 1.0 + math.sqrt(2.0 * sigma2) * _GH_T
    i = 1.0 - np.logaddexp(0.0, -2.0 * y / sigma2) / LN2
    return i, _GH_W / math.sqrt(math.pi)


def biawgn_capacity(sigma2):
    """Capacity of the binary-input AWGN channel with BPSK (+-1) and noise variance sigma^2.

    C = 1 - E[log2(1 + exp(-2Y/sigma^2))], Y ~ N(1, sigma^2), evaluated with 200-point Gauss-Hermite quadrature.
    """
    i, w = _biawgn_info_density(sigma2)
    return float(np.sum(w * i))


def biawgn_capacity_grid(sigma2, points=200001):
    """Same capacity by brute-force trapezoidal integration on a fine grid (used to cross-check the quadrature)."""
    s = math.sqrt(sigma2)
    y = np.linspace(1 - 12 * s, 1 + 12 * s, points)
    pdf = np.exp(-(y - 1) ** 2 / (2 * sigma2)) / math.sqrt(2 * math.pi * sigma2)
    f = pdf * (1.0 - np.logaddexp(0.0, -2.0 * y / sigma2) / LN2)
    return float(np.sum((f[1:] + f[:-1]) * 0.5 * np.diff(y)))


def biawgn_dispersion(sigma2):
    """Dispersion of the BI-AWGN channel with uniform BPSK input: V = Var[i(X;Y)] (bits^2)."""
    i, w = _biawgn_info_density(sigma2)
    c = np.sum(w * i)
    return float(np.sum(w * i * i) - c * c)


def biawgn_normal_approx_fer(ebn0_db, n, R):
    """Normal-approximation FER of the best (n, R) code over BI-AWGN at the given Eb/N0 (dB)."""
    s2 = ebn0_to_sigma2(ebn0_db, R)
    return normal_approx_fer(n, R, biawgn_capacity(s2), biawgn_dispersion(s2))


def biawgn_min_ebn0_db(R):
    """Shannon limit of BPSK: the smallest Eb/N0 (dB) with C_BIAWGN(sigma^2(Eb/N0, R)) >= R, by bisection."""
    lo, hi = -2.0, 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if biawgn_capacity(ebn0_to_sigma2(mid, R)) >= R:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def uncoded_bpsk_ber(ebn0_db):
    """Bit error rate of uncoded BPSK over AWGN: Pb = Q(sqrt(2 Eb/N0))."""
    g = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    return Q(np.sqrt(2.0 * g))


def uncoded_ebn0_for_ber(ber):
    """Eb/N0 (dB) at which uncoded BPSK reaches the given BER (bisection on Q(sqrt(2x)))."""
    lo, hi = -10.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if uncoded_bpsk_ber(mid) > ber:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
