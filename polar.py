"""polar.py - Polar codes: encoder, frozen-set design, SC and CRC-aided SC-List decoding (LLR domain).

Conventions (used consistently by encoder, decoders and the reliability computation)
------------------------------------------------------------------------------------
* x = u G_N with G_N = F^{(x)n}, F = [[1, 0], [1, 1]] over GF(2), NO bit-reversal.
  Recursively x = [ (u_L xor u_R) G_{N/2}, u_R G_{N/2} ].
* The SC decoder splits the channel LLR vector into halves (a | b):
      left  child LLRs = f(a, b)                 (the 'minus' / check-node channel W-)
      right child LLRs = g(a, b, beta_L) = b + (1 - 2 beta_L) a   (the 'plus' channel W+)
  so the most significant bit of index i selects the FIRST transform applied to the raw
  channel.  Index 0 = W^{--...-} is the worst synthetic channel, index N-1 = W^{++...+}
  the best.  The reliability recursions below build the index in exactly this order.
* LLR convention: L = ln P(y|0)/P(y|1), so L > 0 favours bit 0.

Theory: channel polarization (Arikan 2009) - as N grows, the synthetic channels
W_N^(i) seen by successive cancellation become either almost perfect or almost
useless; the fraction of good ones tends to the capacity.  Information bits go on the
K most reliable channels; the rest are frozen to 0.
"""
import math
import time

import numpy as np

from theory import ebn0_to_sigma2

F_KERNEL = np.array([[1, 0], [1, 1]], dtype=np.uint8)


# ============================================================================ encoder
def polar_transform(u):
    """x = u F^{(x)n} along the last axis via the O(N log N) in-place butterfly (any leading batch shape).

    Stage with half-size h:  x[j] ^= x[j + h] inside every block of 2h.  The stages commute
    (Kronecker structure), and the transform is an involution over GF(2): G_N G_N = I.
    """
    x = np.array(u, dtype=np.uint8, copy=True)
    N = x.shape[-1]
    if N & (N - 1):
        raise ValueError("N must be a power of two")
    lead = x.shape[:-1]
    h = 1
    while h < N:
        v = x.reshape(lead + (N // (2 * h), 2, h))
        v[..., 0, :] ^= v[..., 1, :]
        h *= 2
    return x


def generator_matrix(N):
    """Explicit Kronecker-power generator G_N = F^{(x)n} (for self-tests; O(N^2) memory)."""
    G = np.ones((1, 1), dtype=np.uint8)
    while G.shape[0] < N:
        G = np.kron(F_KERNEL, G)
    return G


def encoder_matches_matrix(N, trials=20, seed=0):
    """Self-test: the fast butterfly equals u G_N (mod 2) computed by explicit matrix multiplication."""
    rng = np.random.default_rng(seed)
    G = generator_matrix(N).astype(np.int64)
    u = rng.integers(0, 2, (trials, N), dtype=np.uint8)
    return bool(np.array_equal(polar_transform(u), (u.astype(np.int64) @ G) % 2))


# ============================================================================ reliability design
def log1mexp(x):
    """ln(1 - e^x) for x < 0, accurate at both ends (Maechler's switch at -ln 2)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(x > -math.log(2.0), np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


def bhattacharyya_bsc(N, p):
    """(ln Z, ln(1 - Z)) of the N synthetic channels of BSC(p) by the Bhattacharyya recursion.

    Z(W) = 2 sqrt(p(1-p));  Z(W-) = 2Z - Z^2 ,  Z(W+) = Z^2.
    Exact for the BEC; for other channels Z(W-) <= 2Z - Z^2 is only an upper bound, so the
    ranking is a heuristic there.  Index ordering: MSB = first transform (see module docstring).
    """
    return _bhattacharyya(N, 2.0 * math.sqrt(p * (1.0 - p)))


def bhattacharyya_bec(N, eps):
    """(ln Z, ln(1 - Z)) for BEC(eps), where Z = erasure probability and the recursion is exact."""
    return _bhattacharyya(N, eps)


def _bhattacharyya(N, z0):
    """Bhattacharyya recursion carried in (ln Z, ln(1-Z)) so neither Z -> 0 nor Z -> 1 loses precision.

    minus: 1 - Z- = (1 - Z)^2  ->  ln(1-Z-) = 2 ln(1-Z),  ln Z- = ln(1 - e^{ln(1-Z-)})
    plus : Z+ = Z^2            ->  ln Z+ = 2 ln Z,        ln(1-Z+) = ln(1 - e^{ln Z+})
    """
    lz, l1z = np.array([math.log(z0)]), np.array([math.log1p(-z0)])
    while lz.size < N:
        m1 = 2.0 * l1z
        p0 = 2.0 * lz
        lz = np.stack([log1mexp(m1), p0], axis=1).reshape(-1)
        l1z = np.stack([m1, log1mexp(p0)], axis=1).reshape(-1)
    return lz, l1z


def bhattacharyya_log_bsc(N, p):
    """ln Z of the synthetic channels of BSC(p) (see bhattacharyya_bsc)."""
    return bhattacharyya_bsc(N, p)[0]


def bhattacharyya_log_bec(N, eps):
    """ln Z of the synthetic channels of BEC(eps) (exact erasure probabilities)."""
    return bhattacharyya_bec(N, eps)[0]


def bhattacharyya_reliability(lz, l1z):
    """Reliability score ln((1-Z)/Z) = -logit(Z): monotone decreasing in Z and precise at both ends."""
    return l1z - lz


def phi_chung(x):
    """Chung-Richardson-Urbanke (2001) approximation of phi(x) = 1 - E[tanh(U/2)], U ~ N(x, 2x), returned as ln phi.

    phi(x) ~ exp(-0.4527 x^0.86 + 0.0218)              for 0 < x < 10
    phi(x) ~ sqrt(pi/x) exp(-x/4) (1 - 10/(7x))         for x >= 10
    (clipped to phi <= 1; phi(0) = 1).
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    lo = (x > 0) & (x < 10)
    hi = x >= 10
    out[lo] = -0.4527 * x[lo] ** 0.86 + 0.0218
    xh = x[hi]
    out[hi] = 0.5 * np.log(np.pi / xh) - xh / 4.0 + np.log(1.0 - 10.0 / (7.0 * xh))
    return np.minimum(out, 0.0)


def phi_numeric(x, nodes=200):
    """phi(x) = E[2 / (1 + e^U)], U ~ N(x, 2x), by Gauss-Hermite quadrature (reference for phi_chung)."""
    t, w = np.polynomial.hermite.hermgauss(nodes)
    u = x + math.sqrt(4.0 * x) * t
    return float(np.sum(w * 2.0 / (1.0 + np.exp(u))) / math.sqrt(math.pi))


def _phi_inv_log(target_log):
    """Inverse of ln phi_chung by vectorised bisection in ln x (ln phi is decreasing in x)."""
    target_log = np.asarray(target_log, dtype=float)
    lo = np.full(target_log.shape, math.log(1e-12))
    hi = np.full(target_log.shape, math.log(1e7))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        above = phi_chung(np.exp(mid)) > target_log      # phi still too large -> x must grow
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return np.exp(0.5 * (lo + hi))


GA_SMALL = 0.05      # below this mean LLR use the small-mean asymptote 1 - phi(m) ~ m/2


def ga_log_means(N, sigma2):
    """Gaussian approximation (Trifonov 2012): ln of the mean LLR of each synthetic channel of BI-AWGN(sigma^2).

    Channel LLR ~ N(m, 2m) with m = 2/sigma^2.  Recursion:
        m(W+) = 2m,     m(W-) = phi^{-1}( 1 - (1 - phi(m))^2 ),
    with phi from Chung et al.  For very unreliable channels (m < GA_SMALL) Chung's formula is
    inaccurate, so the small-mean asymptote 1 - phi(m) ~ m/2 is used, giving m(W-) = 2 (1 - phi(m))^2
    (= m^2 / 2).  Means are carried as ln m so that repeated squaring never underflows to ties.
    Larger mean = more reliable.
    """
    lm = np.array([math.log(2.0 / sigma2)])
    while lm.size < N:
        m = np.exp(lm)
        lp = phi_chung(m)
        one_minus_phi = np.where(m < GA_SMALL, 0.5 * m, -np.expm1(lp))
        target = lp + np.log(2.0 - np.exp(lp))                   # ln(1 - (1 - phi)^2)
        inv = _phi_inv_log(target)
        with np.errstate(divide="ignore"):
            small = math.log(2.0) + 2.0 * np.where(m < GA_SMALL, lm - math.log(2.0), np.log(one_minus_phi))
        minus = np.where((m < GA_SMALL) | (inv < GA_SMALL), small, np.log(inv))
        lm = np.stack([minus, lm + math.log(2.0)], axis=1).reshape(-1)
    return lm


def ga_means(N, sigma2):
    """Mean LLRs of the synthetic channels (exp of ga_log_means; tiny values may underflow to 0)."""
    return np.exp(ga_log_means(N, sigma2))


def select_info_set(reliability, K):
    """Indices of the K largest reliability scores (ties broken towards the larger index), sorted ascending."""
    N = reliability.size
    order = np.lexsort((np.arange(N), reliability))     # ascending: reliability, then index
    return np.sort(order[N - K:])


# ============================================================================ LLR kernels
def f_minsum(a, b):
    """Min-sum check-node update f(a,b) = sign(a) sign(b) min(|a|, |b|)."""
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def f_exact(a, b):
    """Exact box-plus 2 atanh(tanh(a/2) tanh(b/2)), written stably with the Jacobian logarithm."""
    return (np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
            + np.log1p(np.exp(-np.abs(a + b))) - np.log1p(np.exp(-np.abs(a - b))))


def g_func(a, b, beta):
    """Variable-node update g(a, b, u) = b + (1 - 2u) a."""
    return b + np.where(beta.astype(bool), -a, a)


def path_penalty(llr, bit):
    """Path-metric increment ln(1 + exp(-(1 - 2u) L)) >= 0: the negative log-likelihood of deciding u given L."""
    return np.logaddexp(0.0, np.where(np.asarray(bit).astype(bool), llr, -llr))


# ============================================================================ code
class PolarCode:
    """(N, K) polar code; optionally K = (K - r) payload bits + r CRC bits.

    Effective code rate R_eff = (K - r) / N  (payload bits per channel use).
    Payload || CRC are placed on the information positions in increasing index order.
    """

    def __init__(self, N=256, K=128, crc=None, design="bsc", design_param=0.05, f_mode="minsum",
                 info_set=None):
        """Design the code.  design='bsc' (Bhattacharyya at p=design_param), 'awgn' (GA at Eb/N0=design_param dB,
        using R_eff in sigma^2), or 'bec' (exact Bhattacharyya at eps=design_param)."""
        if N not in (2 ** k for k in range(1, 15)):
            raise ValueError("N must be a power of two")
        if not 0 < K <= N:
            raise ValueError("need 0 < K <= N")
        self.N, self.K, self.crc = N, K, crc
        self.r = crc.r if crc is not None else 0
        if K <= self.r:
            raise ValueError("K must exceed the CRC length")
        self.k_payload = K - self.r
        self.R_eff = self.k_payload / N
        self.design, self.design_param = design, design_param
        # reliability: larger = better. BSC/BEC: ln((1-Z)/Z); AWGN: ln(GA mean LLR)
        if design == "bsc":
            self.reliability = bhattacharyya_reliability(*bhattacharyya_bsc(N, design_param))
        elif design == "bec":
            self.reliability = bhattacharyya_reliability(*bhattacharyya_bec(N, design_param))
        elif design == "awgn":
            self.reliability = ga_log_means(N, ebn0_to_sigma2(design_param, self.R_eff))
        else:
            raise ValueError("design must be 'bsc', 'awgn' or 'bec'")
        self.info = np.sort(np.asarray(info_set)) if info_set is not None else select_info_set(self.reliability, K)
        self.frozen = np.ones(N, dtype=bool)
        self.frozen[self.info] = False
        self._cnt = np.concatenate([[0], np.cumsum(~self.frozen)])
        self.f = f_minsum if f_mode == "minsum" else f_exact

    # ------------------------------------------------------------------ encoding
    def encode(self, payload):
        """Encode payload blocks (..., K - r) -> codewords (..., N): append CRC, place on info set, polar transform."""
        payload = np.asarray(payload, dtype=np.uint8)
        if self.crc is not None:
            payload = np.concatenate([payload, self.crc.compute_batch(payload)], axis=-1)
        u = np.zeros(payload.shape[:-1] + (self.N,), dtype=np.uint8)
        u[..., self.info] = payload
        return polar_transform(u)

    def _n_info(self, start, n):
        """Number of information positions in u[start : start + n]."""
        return self._cnt[start + n] - self._cnt[start]

    # ------------------------------------------------------------------ SC
    def decode_sc(self, llr):
        """Plain successive-cancellation decoding of LLRs (B, N) -> dict(payload, info_bits, crc_ok, x_hat).

        Hard decision at each information leaf: u = 1 iff L < 0 (L = 0 -> 0); frozen leaves are 0.
        All-frozen subtrees are skipped (their decisions are 0 regardless of the LLRs).
        """
        llr = np.atleast_2d(np.asarray(llr, dtype=float))
        x_hat = self._sc(llr, 0)
        return self._finish_single(x_hat)

    def _sc(self, alpha, start):
        """SC recursion on LLRs alpha (B, n) for u[start:start+n]; returns the re-encoded subtree bits (B, n)."""
        n = alpha.shape[-1]
        if self._n_info(start, n) == 0:
            return np.zeros(alpha.shape, dtype=np.uint8)
        if n == 1:
            return (alpha < 0).astype(np.uint8)
        h = n // 2
        a, b = alpha[..., :h], alpha[..., h:]
        bl = self._sc(self.f(a, b), start)
        br = self._sc(g_func(a, b, bl), start + h)
        return np.concatenate([bl ^ br, br], axis=-1)

    def _finish_single(self, x_hat):
        """Recover u from x (G_N is an involution), split payload / CRC and check the CRC."""
        u = polar_transform(x_hat)
        info_bits = u[..., self.info]
        out = {"x_hat": x_hat, "info_bits": info_bits, "payload": info_bits[..., :self.k_payload]}
        out["crc_ok"] = self.crc.check_batch(info_bits) if self.crc is not None else None
        return out

    # ------------------------------------------------------------------ SCL
    def decode_scl(self, llr, L=8):
        """(CRC-aided) successive-cancellation list decoding of LLRs (B, N) with list size L.

        Each path carries a path metric PM = sum of ln(1 + exp(-(1-2u_i) L_i)) over all leaves
        (negative log-likelihood).  At every information bit each path forks into u=0 and u=1;
        the L candidates with the smallest PM survive.  Ties are broken towards the
        hard-decision (ML) bit, which makes L = 1 identical to plain SC.  Lazy copying:
        instead of copying the whole decoder state at every fork, each recursion node
        returns the permutation 'surviving path j descends from path perm[j]' and its
        parent gathers only the arrays it still needs (O(N log N) gathers per frame).
        At the end the most likely path with a valid CRC is chosen; if no path passes,
        the best-metric path is output and crc_ok = False ('CRC FAILED').
        """
        llr = np.atleast_2d(np.asarray(llr, dtype=float))
        B, N = llr.shape
        self._pm = np.full((B, L), np.inf)
        self._pm[:, 0] = 0.0
        self._bidx = np.arange(B)[:, None]
        alpha = np.repeat(llr[:, None, :], L, axis=1)
        beta, _ = self._scl(alpha, 0)
        pm = self._pm
        u = polar_transform(beta)
        info_bits = u[..., self.info]                            # (B, L, K)
        finite = np.isfinite(pm)
        if self.crc is not None:
            ok = self.crc.check_batch(info_bits) & finite
            score = np.where(ok, pm, np.inf)
            best = np.where(ok.any(axis=1), np.argmin(score, axis=1), np.argmin(pm, axis=1))
            crc_ok = ok.any(axis=1)
        else:
            best = np.argmin(pm, axis=1)
            crc_ok = None
        b = np.arange(B)
        info_sel = info_bits[b, best]
        return {"x_hat": beta[b, best], "info_bits": info_sel, "payload": info_sel[:, :self.k_payload],
                "crc_ok": crc_ok, "path_metric": pm[b, best], "list_info_bits": info_bits, "list_pm": pm}

    def _gather(self, arr, perm):
        """arr[b, perm[b, j]] for every block b: copy the state of the surviving parents."""
        return arr[self._bidx, perm]

    _skip_rate0 = True

    def _scl(self, alpha, start):
        """SCL recursion on alpha (B, L, n); returns (beta (B, L, n), perm (B, L) or None for identity)."""
        n = alpha.shape[-1]
        if self._skip_rate0 and self._n_info(start, n) == 0:
            self._pm = self._pm + path_penalty(self._rate0_leaf_llrs(alpha), 0).sum(axis=-1)
            return np.zeros(alpha.shape, dtype=np.uint8), None
        if n == 1:
            return self._scl_leaf(alpha[..., 0], start)
        h = n // 2
        a, b = alpha[..., :h], alpha[..., h:]
        bl, pl = self._scl(self.f(a, b), start)
        if pl is not None:
            a, b = self._gather(a, pl), self._gather(b, pl)
        br, pr = self._scl(g_func(a, b, bl), start + h)
        if pr is not None:
            bl = self._gather(bl, pr)
            perm = pr if pl is None else np.take_along_axis(pl, pr, axis=1)
        else:
            perm = pl
        return np.concatenate([bl ^ br, br], axis=-1), perm

    def _rate0_leaf_llrs(self, alpha):
        """Leaf LLRs of an all-frozen subtree (all partial sums 0, so g = a + b), level by level.

        Performs exactly the same floating-point operations as the bit-by-bit recursion.
        """
        n = alpha.shape[-1]
        lead = alpha.shape[:-1]
        x = alpha
        blocks = 1
        while blocks < n:
            size = n // blocks
            v = x.reshape(lead + (blocks, size))
            a, b = v[..., :size // 2], v[..., size // 2:]
            x = np.concatenate([self.f(a, b), b + a], axis=-1).reshape(lead + (n,))
            blocks *= 2
        return x

    def _scl_leaf(self, lam, i):
        """Information leaf i: fork every path, keep the L best of the 2L candidates."""
        L = lam.shape[1]
        ml = (lam < 0).astype(np.uint8)
        mag = np.abs(lam)
        pm = self._pm
        cand = np.concatenate([pm + np.logaddexp(0.0, -mag), pm + np.logaddexp(0.0, mag)], axis=1)
        order = np.argsort(cand, axis=1, kind="stable")[:, :L]
        path = order % L
        flip = (order >= L).astype(np.uint8)
        self._pm = np.take_along_axis(cand, order, axis=1)
        bits = np.take_along_axis(ml, path, axis=1) ^ flip
        return bits[..., None], path

    # ------------------------------------------------------------------ validation
    def validate_frozen_set(self, decoders=True):
        """Self-test of the design: |A| = K, index N-1 most / index 0 least reliable, N-1 in A (and 0 frozen
        when K < N), and the all-zero codeword decodes to all-zero with SC and SCL."""
        rel = self.reliability
        ok = (self.info.size == self.K and int(np.argmax(rel)) == self.N - 1 and int(np.argmin(rel)) == 0
              and (self.N - 1) in self.info and (self.K == self.N or 0 not in self.info))
        if ok and decoders:
            llr = np.full((1, self.N), 20.0)                     # noiseless all-zero codeword
            ok = (not self.decode_sc(llr)["info_bits"].any()) and (not self.decode_scl(llr, 4)["info_bits"].any())
        return bool(ok)


# ============================================================================ OPTIONAL TIER 2: PAC codes
PAC_GENERATOR = (1, 0, 1, 1, 0, 1, 1)        # c(D) = 1 + D^2 + D^3 + D^5 + D^6, binary 1011011 = octal 133
_PARITY6 = np.array([bin(s).count("1") & 1 for s in range(64)], dtype=np.uint8)


def conv_precode(v, gen=PAC_GENERATOR):
    """Rate-1 convolutional precoder u_i = sum_j c_j v_{i-j} (mod 2) along the last axis.

    With c_0 = 1 the map is an upper-triangular Toeplitz matrix with unit diagonal, hence invertible.
    """
    v = np.asarray(v, dtype=np.uint8)
    u = v.copy() if gen[0] else np.zeros_like(v)
    for j, c in enumerate(gen[1:], 1):
        if c:
            u[..., j:] ^= v[..., :-j]
    return u


def conv_deprecode(u, gen=PAC_GENERATOR):
    """Inverse precoder (requires c_0 = 1): v_i = u_i xor sum_{j>=1} c_j v_{i-j}, solved sequentially."""
    if gen[0] != 1:
        raise ValueError("precoder is invertible only with c_0 = 1")
    u = np.asarray(u, dtype=np.uint8)
    v = np.zeros_like(u)
    for i in range(u.shape[-1]):
        acc = u[..., i].copy()
        for j, c in enumerate(gen[1:], 1):
            if c and i - j >= 0:
                acc ^= v[..., i - j]
        v[..., i] = acc
    return v


def rm_rate_profile(N, K, reliability):
    """RM rate profile: the K indices of largest Hamming weight wt(i), ties broken by reliability (larger first)."""
    wt = np.array([bin(i).count("1") for i in range(N)])
    order = np.lexsort((np.arange(N), reliability, wt))       # ascending by weight, then reliability, then index
    return np.sort(order[N - K:])


class PACCode(PolarCode):
    """OPTIONAL TIER 2 - PAC code (Arikan 2019): x = conv(v) G_N with an RM rate profile on v.

    Data bits go to v on the rate-profile set A (no CRC), v is 0 elsewhere.  Decoding is
    SC-List over the u-domain with the 6-bit convolutional state carried by every path: at
    leaf i, u_i = v_i xor (state taps); frozen v_i = 0 still gives a (state-dependent) u_i,
    so there are no all-zero subtrees to skip.  Note: published PAC results use L = 128+ or
    Fano sequential decoding; at small L the advantage over CA-polar may not appear.
    """

    _skip_rate0 = False

    def __init__(self, N=256, K=117, design="awgn", design_param=2.5, f_mode="minsum", gen=PAC_GENERATOR):
        """Design: reliability from the usual GA/Bhattacharyya construction, info set from the RM profile."""
        super().__init__(N, K, None, design=design, design_param=design_param, f_mode=f_mode)
        self.gen = tuple(gen)
        if len(self.gen) != 7 or self.gen[0] != 1:
            raise ValueError("generator must have 7 taps with c_0 = 1")
        self.info = rm_rate_profile(N, K, self.reliability)
        self.frozen = np.ones(N, dtype=bool)
        self.frozen[self.info] = False
        self._cnt = np.concatenate([[0], np.cumsum(~self.frozen)])
        self._mask = sum(c << (j - 1) for j, c in enumerate(self.gen[1:], 1))

    def encode(self, payload):
        """payload (..., K) -> v on A -> u = conv(v) -> x = u G_N."""
        payload = np.asarray(payload, dtype=np.uint8)
        v = np.zeros(payload.shape[:-1] + (self.N,), dtype=np.uint8)
        v[..., self.info] = payload
        return polar_transform(conv_precode(v, self.gen))

    def decode_sc(self, llr):
        """SC decoding of a PAC code = SCL with L = 1."""
        return self.decode_scl(llr, 1)

    def decode_scl(self, llr, L=8):
        """List decoding with per-path convolutional state; the best-metric path is the output (no CRC)."""
        llr = np.atleast_2d(np.asarray(llr, dtype=float))
        B = llr.shape[0]
        self._pm = np.full((B, L), np.inf)
        self._pm[:, 0] = 0.0
        self._state = np.zeros((B, L), dtype=np.int64)
        self._bidx = np.arange(B)[:, None]
        beta, _ = self._scl(np.repeat(llr[:, None, :], L, axis=1), 0)
        best = np.argmin(self._pm, axis=1)
        b = np.arange(B)
        x_hat = beta[b, best]
        v = conv_deprecode(polar_transform(x_hat), self.gen)
        info_bits = v[:, self.info]
        return {"x_hat": x_hat, "info_bits": info_bits, "payload": info_bits, "crc_ok": None,
                "path_metric": self._pm[b, best]}

    def _scl_leaf(self, lam, i):
        """Leaf i: u_i = v_i xor parity(state & taps); fork on v_i only on the rate-profile set."""
        par = _PARITY6[self._state & self._mask]
        if self.frozen[i]:
            self._pm = self._pm + path_penalty(lam, par)
            self._state = (self._state << 1) & 63
            return par[..., None], None
        L = lam.shape[1]
        ml = (lam < 0).astype(np.uint8)
        mag = np.abs(lam)
        cand = np.concatenate([self._pm + np.logaddexp(0.0, -mag), self._pm + np.logaddexp(0.0, mag)], axis=1)
        order = np.argsort(cand, axis=1, kind="stable")[:, :L]
        path = order % L
        flip = (order >= L).astype(np.uint8)
        self._pm = np.take_along_axis(cand, order, axis=1)
        u = np.take_along_axis(ml, path, axis=1) ^ flip
        v = u ^ np.take_along_axis(par, path, axis=1)
        self._state = ((np.take_along_axis(self._state, path, axis=1) << 1) | v) & 63
        return u[..., None], path

    def validate_frozen_set(self, decoders=True):
        """|A| = K, precoder invertible, and the all-zero codeword decodes to zero."""
        rng = np.random.default_rng(0)
        v = rng.integers(0, 2, (4, self.N), dtype=np.uint8)
        ok = self.info.size == self.K and np.array_equal(conv_deprecode(conv_precode(v, self.gen), self.gen), v)
        if ok and decoders:
            ok = not self.decode_scl(np.full((1, self.N), 20.0), 4)["info_bits"].any()
        return bool(ok)


# ============================================================================ simulation helper
def simulate_block_errors(code, llr_fn, frames, rng, L=1, batch=200, use_list=None):
    """Monte-Carlo helper: random payloads -> encode -> llr_fn(codewords, rng) -> decode; returns error counts.

    use_list=None picks SC when L == 1 and no CRC, SCL otherwise.  Returns dict with frames, frame errors,
    payload bit errors and decoding time.
    """
    done = fe = be = 0
    t_dec = 0.0
    while done < frames:
        nb = min(batch, frames - done)
        pay = rng.integers(0, 2, (nb, code.k_payload), dtype=np.uint8)
        llr = llr_fn(code.encode(pay), rng)
        t0 = time.perf_counter()
        if (use_list is None and L == 1 and code.crc is None) or use_list is False:
            out = code.decode_sc(llr)
        else:
            out = code.decode_scl(llr, L)
        t_dec += time.perf_counter() - t0
        err = out["payload"] != pay
        be += int(err.sum())
        fe += int(err.any(axis=1).sum())
        done += nb
    return {"frames": done, "frame_errors": fe, "bit_errors": be, "time": t_dec}
