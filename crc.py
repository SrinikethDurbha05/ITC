"""crc.py - Cyclic redundancy check (CRC) codes, implemented bitwise from scratch.

Theory
------
A CRC with generator polynomial g(x) of degree r maps a k-bit message m(x)
to the r-bit remainder c(x) = m(x) * x^r mod g(x) over GF(2).  The transmitted
word m(x) * x^r + c(x) is then a multiple of g(x), so its remainder is zero.
Because the map m -> c is GF(2)-linear (zero initial register, no final XOR),
it can also be written as c = m * M (mod 2) with a k x r matrix M, which we use
to check many decoder candidates (all list paths of all blocks) at once.

Error-detection guarantees used in the tests:
  * every single-bit error is detected when g(x) has at least two terms;
  * every burst of length <= r is detected when g(x) has a non-zero constant
    term (x does not divide g(x)).

Polynomials (3GPP TS 38.212, Sec. 5.1; the hex value includes the x^r term):
  CRC-6   : x^6 + x^5 + 1                                  -> 0x61
  CRC-11  : x^11 + x^10 + x^9 + x^5 + 1                     -> 0xE21
  CRC-24C : x^24+x^23+x^21+x^20+x^17+x^15+x^13+x^12+x^8+x^4+x^2+x+1 -> 0x1B2B117
"""
from functools import lru_cache

import numpy as np

CRC_POLYS = {
    "CRC-6": (6, 0x61),
    "CRC-11": (11, 0xE21),
    "CRC-24C": (24, 0x1B2B117),
}


def poly_mod(bits, poly, r):
    """Remainder of the GF(2) polynomial `bits` (MSB = highest power) modulo g(x).

    Textbook long division: wherever the leading coefficient is 1, XOR the
    generator (r+1 coefficients) into the dividend.  Returns r bits (uint8).
    """
    a = np.array(bits, dtype=np.uint8).copy()
    if a.size < r + 1:
        a = np.concatenate([np.zeros(r + 1 - a.size, np.uint8), a])
    g = np.array([(poly >> (r - i)) & 1 for i in range(r + 1)], dtype=np.uint8)
    for i in range(a.size - r):
        if a[i]:
            a[i:i + r + 1] ^= g
    return a[-r:].copy()


class CRC:
    """A CRC code: c(x) = m(x) x^r mod g(x), zero initial value, no reflection/final XOR."""

    def __init__(self, name="CRC-11"):
        """Select one of the 3GPP polynomials by name ('CRC-6', 'CRC-11', 'CRC-24C')."""
        if name not in CRC_POLYS:
            raise ValueError(f"unknown CRC {name!r}; choose from {list(CRC_POLYS)}")
        self.name = name
        self.r, self.poly = CRC_POLYS[name]

    def __repr__(self):
        """Name and generator polynomial."""
        return f"{self.name} (0x{self.poly:X})"

    def compute(self, payload):
        """CRC of one message by bitwise long division of m(x) * x^r by g(x)."""
        payload = np.asarray(payload, dtype=np.uint8)
        return poly_mod(np.concatenate([payload, np.zeros(self.r, np.uint8)]), self.poly, self.r)

    def check(self, word):
        """True iff payload||crc is divisible by g(x) (remainder all zero)."""
        return not poly_mod(word, self.poly, self.r).any()

    def matrix(self, k):
        """k x r GF(2) matrix M with CRC(m) = m M; row i is the CRC of unit vector e_i (linearity)."""
        return _crc_matrix(self.poly, self.r, k)

    def compute_batch(self, payloads):
        """Vectorized CRC of payloads with shape (..., k), using the linear map c = m M mod 2."""
        payloads = np.asarray(payloads, dtype=np.uint8)
        k = payloads.shape[-1]
        M = self.matrix(k).astype(np.int32)
        return ((payloads.astype(np.int32) @ M) & 1).astype(np.uint8)

    def check_batch(self, words):
        """Vectorized check of words with shape (..., k + r): True where the CRC matches."""
        words = np.asarray(words, dtype=np.uint8)
        k = words.shape[-1] - self.r
        return np.all(self.compute_batch(words[..., :k]) == words[..., k:], axis=-1)


@lru_cache(maxsize=64)
def _crc_matrix(poly, r, k):
    """Cached generator of the CRC matrix for (poly, r, k); row i = CRC(e_i) by long division."""
    M = np.zeros((k, r), dtype=np.uint8)
    for i in range(k):
        m = np.zeros(k + r, np.uint8)
        m[i] = 1
        M[i] = poly_mod(m, poly, r)
    M.setflags(write=False)
    return M


def self_test(name="CRC-11", k=117, seed=0):
    """Quick self-test: remainder of payload||CRC is zero and every single-bit error is detected."""
    rng = np.random.default_rng(seed)
    crc = CRC(name)
    m = rng.integers(0, 2, k, dtype=np.uint8)
    w = np.concatenate([m, crc.compute(m)])
    if not crc.check(w):
        return False
    for i in range(w.size):
        e = w.copy()
        e[i] ^= 1
        if crc.check(e):
            return False
    return bool(np.array_equal(crc.compute_batch(m), crc.compute(m)))
