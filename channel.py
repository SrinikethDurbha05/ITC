"""channel.py - Wired-link channel models: lossless, BSC(p) and BPSK over AWGN (BI-AWGN).

* BSC(p): a hard-decision link (a comparator/slicer at the line receiver) flips each
  bit independently with probability p.  LLR = (1 - 2y) ln((1-p)/p).
* BI-AWGN: BPSK 0 -> +1, 1 -> -1 over additive white Gaussian (thermal) noise
  y = s + n, n ~ N(0, sigma^2), sigma^2 = 1 / (2 R Eb/N0), with R the number of
  information bits per channel use (R_eff = (K - r)/N for the coded system, R = 1
  for uncoded BPSK).  LLR = 2y / sigma^2.
* Lossless: y = x, LLR = +-LOSSLESS_LLR (a large finite value).
Channel bit errors are always counted on hard decisions at the channel output.
"""
import math

import numpy as np

import theory

LOSSLESS_LLR = 20.0


class Channel:
    """A memoryless binary-input channel; kind in {'lossless', 'bsc', 'awgn'}."""

    def __init__(self, kind="bsc", p=None, ebn0_db=None, rate=1.0):
        """Create a channel.  BSC needs 0 < p < 0.5; AWGN needs Eb/N0 (dB) and the information rate R."""
        self.kind = kind
        if kind == "bsc":
            if p is None or not 0.0 < p < 0.5:
                raise ValueError("BSC crossover probability must lie in (0, 0.5)")
            self.p = float(p)
        elif kind == "awgn":
            if ebn0_db is None or not -20.0 <= ebn0_db <= 30.0:
                raise ValueError("Eb/N0 must be given in dB within [-20, 30]")
            if not 0.0 < rate <= 1.0:
                raise ValueError("rate must lie in (0, 1]")
            self.ebn0_db, self.rate = float(ebn0_db), float(rate)
            self.sigma2 = theory.ebn0_to_sigma2(ebn0_db, rate)
        elif kind != "lossless":
            raise ValueError("kind must be 'lossless', 'bsc' or 'awgn'")

    def describe(self):
        """Human-readable channel description."""
        if self.kind == "bsc":
            return f"BSC (p = {self.p:g})"
        if self.kind == "awgn":
            return f"BPSK over AWGN (Eb/N0 = {self.ebn0_db:g} dB, sigma^2 = {self.sigma2:.4f})"
        return "Lossless"

    def transmit(self, x, rng):
        """Send bits x (any shape) through the channel; returns dict(y, hard, llr, errors)."""
        x = np.asarray(x, dtype=np.uint8)
        if self.kind == "lossless":
            y = x.copy()
            hard = y
            llr = (1.0 - 2.0 * y) * LOSSLESS_LLR
        elif self.kind == "bsc":
            y = x ^ (rng.random(x.shape) < self.p).astype(np.uint8)
            hard = y
            llr = (1.0 - 2.0 * y) * math.log((1.0 - self.p) / self.p)
        else:
            s = 1.0 - 2.0 * x
            y = s + math.sqrt(self.sigma2) * rng.standard_normal(x.shape)
            hard = (y < 0).astype(np.uint8)
            llr = 2.0 * y / self.sigma2
        return {"y": y, "hard": hard, "llr": llr, "errors": int(np.count_nonzero(hard != x))}

    def raw_bit_error_prob(self):
        """Probability that a hard decision at this channel's output is wrong (at the coded symbol energy)."""
        if self.kind == "bsc":
            return self.p
        if self.kind == "awgn":
            return float(theory.Q(1.0 / math.sqrt(self.sigma2)))
        return 0.0

    def uncoded_bit_error_prob(self):
        """Bit error probability if the source bits were sent uncoded at the same Eb/N0 (R = 1) or same p."""
        if self.kind == "bsc":
            return self.p
        if self.kind == "awgn":
            return float(theory.uncoded_bpsk_ber(self.ebn0_db))
        return 0.0

    def capacity(self):
        """Capacity in bits per channel use (BSC: 1 - h2(p); AWGN: BI-AWGN capacity at this sigma^2)."""
        if self.kind == "bsc":
            return float(theory.bsc_capacity(self.p))
        if self.kind == "awgn":
            return theory.biawgn_capacity(self.sigma2)
        return 1.0
