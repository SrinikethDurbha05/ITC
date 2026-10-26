"""utils.py - Shared helpers: bit/byte conversion, empirical entropies, test corpora, formatting."""
import math
from collections import Counter

import numpy as np


def bytes_to_bits(data):
    """MSB-first bit expansion of a byte string (uint8 array of 0/1)."""
    if len(data) == 0:
        return np.zeros(0, np.uint8)
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))


def bits_to_bytes(bits):
    """Inverse of bytes_to_bits (length must be a multiple of 8)."""
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % 8:
        raise ValueError("bit count is not a multiple of 8")
    return np.packbits(bits).tobytes()


def entropy_h0(symbols):
    """Empirical order-0 entropy H0 = -sum_a f(a) log2 f(a) in bits/symbol (plug-in estimate)."""
    n = len(symbols)
    if n == 0:
        return 0.0
    return -sum(c / n * math.log2(c / n) for c in Counter(symbols).values())


def entropy_h1(symbols):
    """Empirical order-1 conditional entropy H(X_i | X_{i-1}) in bits/symbol over the n-1 consecutive pairs.

    H1 = -sum_{a,b} f(a,b) log2 f(b|a).  As a plug-in estimate on short inputs it is strongly
    biased downwards (it ignores the cost of describing the model), so it is not an achievable
    rate for a universal code - it is shown only as a reference.
    """
    n = len(symbols)
    if n < 2:
        return 0.0
    pairs = Counter(zip(symbols[:-1], symbols[1:]))
    prev = Counter(symbols[:-1])
    m = n - 1
    return -sum(c / m * math.log2(c / prev[a]) for (a, _), c in pairs.items())


# Evaluation text for Fig. 5, original prose written for this project and deliberately
# DIFFERENT from cm_coder.WARM_TEXT, so warm-start results are not inflated by memorisation.
EVAL_TEXT = (
    "Last summer our class built a small radio link between two buildings on the "
    "campus. The idea was simple: one computer would read a short note typed by a "
    "student, turn every letter into bits, and push those bits down a long copper "
    "cable to a second computer on the other side of the yard. On the first day "
    "nothing worked. The cable picked up noise from a motor in the workshop, and "
    "about one bit in twenty arrived flipped. Words came out as strange strings of "
    "symbols, and nobody could read them.\n"
    "Our teacher asked us to think before we added more wires. Instead of sending "
    "each bit once, she said, we should add a few extra bits that depend on the "
    "message in a clever way. Then the receiver could guess the most likely message, "
    "even when some bits were wrong. We tried repeating every bit three times, but "
    "that wasted two thirds of the cable. Later we learnt about codes that protect "
    "long blocks of bits at once, and about decoders that keep a short list of "
    "candidate messages instead of trusting a single guess.\n"
    "Another group worked on the other end of the problem. Plain English is full of "
    "patterns: after the letter q almost always comes u, spaces appear every few "
    "letters, and common words such as the, and, of and to show up again and again. "
    "A program that learns these patterns while it reads can describe the next "
    "letter with fewer bits than eight. When both groups joined their programs, the "
    "note was first squeezed into fewer bits, then wrapped in protection, and only "
    "then sent over the noisy wire. At the far end the steps ran in reverse order.\n"
    "By the end of the month the link carried short notes, a list of exam dates, and "
    "even a recipe for lemon rice without a single mistake, as long as the noise "
    "stayed below a certain level. Above that level every method failed sooner or "
    "later, which matched what the theory had told us to expect.\n"
)


def english_sample(nbytes):
    """First nbytes bytes of EVAL_TEXT (ASCII, so byte count = character count)."""
    data = EVAL_TEXT.encode("ascii")
    if nbytes > len(data):
        raise ValueError(f"EVAL_TEXT has only {len(data)} bytes")
    return data[:nbytes]


def skewed_binary(n, p1, rng):
    """i.i.d. Bernoulli(p1) bit sequence of length n (list of ints)."""
    return (rng.random(n) < p1).astype(int).tolist()


def fmt_sci(x, digits=3):
    """Compact scientific formatting used in tables."""
    return f"{x:.{digits}e}"
