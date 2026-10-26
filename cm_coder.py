"""cm_coder.py - Bitwise context-mixing (CM) compressor with a binary arithmetic coder.

Theory
------
Source coding theorem: a lossless code needs at least H bits/symbol on average;
an arithmetic coder driven by a probability model P spends -log2 P(x) bits
(+ a few bits of termination), so the better the model predicts, the shorter
the output.  Following the PAQ/LPAQ lineage (Mahoney), the model predicts ONE
BIT at a time (MSB first within a byte):

  * several context models (order-0..3 byte contexts in text mode, order-0..8
    bit contexts in binary mode) and a match model each output P(bit = 1);
  * a logistic mixer combines them in the logit ("stretch") domain:
        st_i = stretch(p_i) = ln(p_i / (1 - p_i))
        p    = squash(sum_i w_i st_i)
    and learns online by gradient descent on coding cost:
        w_i += lr * (y - p) * st_i
    with one weight set per small context (bit position in the byte);
  * an APM/SSE stage refines the mixer output given the partial-byte context.

Determinism: probabilities (12/16-bit), stretch/squash tables, mixer weights
(16.16 fixed point) and the coder all use integer arithmetic, so the decoder,
running the same model on the bits it has already decoded, reproduces every
probability exactly.  The receiver uses only the received bits plus the
built-in warm-start text, which both ends have.

Bitstream = 32-bit header [mode bit | warm-start bit | 30-bit symbol count]
followed by the arithmetic-coded payload.  Symbols are bytes (text mode, UTF-8)
or bits (binary mode).
"""
import math

PROB_BITS = 12
PMAX = (1 << PROB_BITS) - 1          # probabilities live in [1, 4095] / 4096
HEADER_BITS = 32
MAX_COUNT = (1 << 30) - 1

# ----------------------------------------------------------------------------- tables
_KNOTS = [int(4096.0 / (1.0 + math.exp(-(i - 16) / 2.0))) for i in range(33)]


def _squash_int(d):
    """Integer logistic squash(d) = 4096 / (1 + e^{-d/256}), by linear interpolation of 33 knots (LPAQ style)."""
    if d > 2047:
        d = 2047
    if d < -2047:
        d = -2047
    w = d & 127
    j = (d >> 7) + 16
    return (_KNOTS[j] * (128 - w) + _KNOTS[j + 1] * w + 64) >> 7


SQUASH = [_squash_int(d) for d in range(-2047, 2048)]       # index d + 2047


def squash(d):
    """Logit (scaled by 256) -> 12-bit probability, via table."""
    if d > 2047:
        d = 2047
    elif d < -2047:
        d = -2047
    return SQUASH[d + 2047]


def _build_stretch():
    """stretch = inverse of squash on the 12-bit grid: stretch(p) ~ 256 ln(p/(1-p)), integer, monotone."""
    t = [0] * 4096
    pi = 0
    for x in range(-2047, 2048):
        v = _squash_int(x)
        for i in range(pi, v + 1):
            t[i] = x
        pi = v + 1
    for i in range(pi, 4096):
        t[i] = 2047
    return t


STRETCH = _build_stretch()
# adaptive learning rate of a slot seen n times: 1/(n + 1.5), in 16-bit fixed point
RATE = [int(65536 / (n + 1.5)) for n in range(1024)]


# ----------------------------------------------------------------------------- arithmetic coder
class ArithmeticEncoder:
    """32-bit carry-less binary arithmetic coder (LPAQ style).

    The interval [x1, x2] is split at xmid = x1 + (x2 - x1) * p / 4096, where
    p = P(bit = 1) in [1, 4095].  Leading bytes that agree are shipped out, so
    the interval is always at least 2^24 wide after normalisation except in
    the rare 'straddle' case, which costs a little efficiency but never
    correctness (x1 <= xmid < x2 always holds).
    """

    def __init__(self):
        """Start with the full 32-bit interval."""
        self.x1, self.x2 = 0, 0xFFFFFFFF
        self.out = []

    def encode(self, bit, p):
        """Encode one bit with P(1) = p / 4096 (p is an integer, clamped to [1, 4095])."""
        if p < 1:
            p = 1
        elif p > PMAX:
            p = PMAX
        x1, x2 = self.x1, self.x2
        r = x2 - x1
        xmid = x1 + (r >> 12) * p + (((r & 0xFFF) * p) >> 12)
        if bit:
            x2 = xmid
        else:
            x1 = xmid + 1
        out = self.out
        while not ((x1 ^ x2) & 0xFF000000):
            out.append(x2 >> 24)
            x1 = (x1 << 8) & 0xFFFFFFFF
            x2 = ((x2 << 8) & 0xFFFFFFFF) | 255
        self.x1, self.x2 = x1, x2

    def finish(self):
        """Return the code as a list of bits: shipped bytes + the shortest tail t with t000... in [x1, x2].

        The decoder pads with zeros, so the shortest prefix of a number in the final
        interval whose remaining bits are zero identifies the interval uniquely.
        """
        bits = []
        for b in self.out:
            bits.extend((b >> (7 - i)) & 1 for i in range(8))
        for nb in range(33):
            v = (self.x2 >> (32 - nb)) << (32 - nb) if nb < 32 else self.x2
            if v >= self.x1:
                bits.extend((v >> (31 - i)) & 1 for i in range(nb))
                break
        return bits


class ArithmeticDecoder:
    """Decoder mirroring ArithmeticEncoder exactly (same integer arithmetic); reads zeros past the end."""

    def __init__(self, bits, start=0):
        """Read the first 32 code bits starting at bit offset `start`."""
        self.bits, self.pos = bits, start
        self.n = len(bits)
        self.x1, self.x2 = 0, 0xFFFFFFFF
        self.x = self._read(32)

    def _read(self, k):
        """Next k bits as an integer (MSB first), zero-padded beyond the end of the stream."""
        v = 0
        bits, pos, n = self.bits, self.pos, self.n
        for i in range(k):
            v = (v << 1) | (bits[pos + i] if pos + i < n else 0)
        self.pos = pos + k
        return v

    def decode(self, p):
        """Decode one bit given the same P(1) = p / 4096 the encoder used."""
        if p < 1:
            p = 1
        elif p > PMAX:
            p = PMAX
        x1, x2 = self.x1, self.x2
        r = x2 - x1
        xmid = x1 + (r >> 12) * p + (((r & 0xFFF) * p) >> 12)
        bit = 1 if self.x <= xmid else 0
        if bit:
            x2 = xmid
        else:
            x1 = xmid + 1
        while not ((x1 ^ x2) & 0xFF000000):
            x1 = (x1 << 8) & 0xFFFFFFFF
            x2 = ((x2 << 8) & 0xFFFFFFFF) | 255
            self.x = ((self.x << 8) & 0xFFFFFFFF) | self._read(8)
        self.x1, self.x2 = x1, x2
        return bit


# ----------------------------------------------------------------------------- model components
def _slot_update(v, y, limit):
    """Adaptive-rate probability slot.  v packs a 16-bit P(1) and a hit count n (v = p16 << 10 | n).

    p += (y - p) / (n + 1.5): a running frequency estimate early on (fast learning)
    that becomes an exponential moving average once n reaches `limit` (tracks drift).
    """
    p = v >> 10
    n = v & 1023
    p += (((y << 16) - p) * RATE[n]) >> 16
    if n < limit:
        n += 1
    return (p << 10) | n


SLOT_INIT = (32768 << 10)                 # p = 1/2, n = 0


class APM:
    """Adaptive probability map / SSE: refines p given a context by interpolating 33 learnt buckets in the stretch domain."""

    def __init__(self, n_ctx, rate=7):
        """One row of 33 16-bit probabilities per context, initialised to the identity map."""
        self.t = [_squash_int((j - 16) * 128) * 16 for _ in range(n_ctx) for j in range(33)]
        self.rate = rate
        self.idx = 0

    def p(self, pr, cx):
        """Map probability pr (12-bit) in context cx to a refined 12-bit probability."""
        s = STRETCH[pr] + 2048
        lo = s >> 7
        w = s & 127
        base = cx * 33 + lo
        t = self.t
        self.idx = base + (w >> 6)
        return (t[base] * (128 - w) + t[base + 1] * w) >> 11

    def update(self, y):
        """Move the nearer bucket towards the coded bit."""
        i = self.idx
        self.t[i] += ((y << 16) - self.t[i]) >> self.rate


class TextModel:
    """Byte-oriented CM predictor: order-0..3 context models + match model, logistic mixer, APM."""

    N_IN = 6                    # o0, o1, o2, o3, match, bias
    LIMIT = 255                 # adaptive-rate cap of the context slots
    LR = 6                      # mixer learning rate: dw = err * st * LR >> 14 (16.16 weights)
    MINLEN = 4                  # match model: minimum match length in bytes

    def __init__(self):
        """Empty model (p = 1/2 everywhere)."""
        self.c0, self.bitpos, self.c4 = 1, 0, 0
        self.t0 = [SLOT_INIT] * 256
        self.t1 = [SLOT_INIT] * 65536
        self.t2, self.t3 = {}, {}
        self.buf = bytearray()
        self.mtab = {}
        self.mptr, self.mlen = 0, 0
        self.mm = [SLOT_INIT] * 64
        self.w = [[int(0.3 * 65536)] * self.N_IN for _ in range(8)]
        self.apm = APM(256)
        self._st = None

    def clone(self):
        """Independent copy of the full model state (used to cache the warm-started model)."""
        m = TextModel.__new__(TextModel)
        m.c0, m.bitpos, m.c4 = self.c0, self.bitpos, self.c4
        m.t0, m.t1 = self.t0[:], self.t1[:]
        m.t2, m.t3 = dict(self.t2), dict(self.t3)
        m.buf = bytearray(self.buf)
        m.mtab = dict(self.mtab)
        m.mptr, m.mlen = self.mptr, self.mlen
        m.mm = self.mm[:]
        m.w = [row[:] for row in self.w]
        m.apm = APM.__new__(APM)
        m.apm.t, m.apm.rate, m.apm.idx = self.apm.t[:], self.apm.rate, 0
        m._st = None
        return m

    def cap_confidence(self, cap):
        """After warm start, keep the learnt probabilities but limit counts so the model re-adapts quickly."""
        def f(v):
            """Cap the hit count stored in the low 10 bits of a slot."""
            return v if (v & 1023) <= cap else (v & ~1023) | cap
        self.t0 = [f(v) for v in self.t0]
        self.t1 = [f(v) for v in self.t1]
        self.t2 = {k: f(v) for k, v in self.t2.items()}
        self.t3 = {k: f(v) for k, v in self.t3.items()}

    def p(self):
        """Predict P(next bit = 1) as a 12-bit integer."""
        c0, c4 = self.c0, self.c4
        i1 = ((c4 & 0xFF) << 8) | c0
        k2 = ((c4 & 0xFFFF) << 8) | c0
        k3 = ((c4 & 0xFFFFFF) << 8) | c0
        v0 = self.t0[c0]
        v1 = self.t1[i1]
        v2 = self.t2.get(k2, SLOT_INIT)
        v3 = self.t3.get(k3, SLOT_INIT)
        S = STRETCH
        s0 = S[v0 >> 14]
        s1 = S[v1 >> 14]
        s2 = S[v2 >> 14]
        s3 = S[v3 >> 14]
        if self.mlen > 0:
            exp_bit = (self.buf[self.mptr] >> (7 - self.bitpos)) & 1
            mi = (min(self.mlen, 31) << 1) | exp_bit
            sm = S[self.mm[mi] >> 14]
        else:
            exp_bit, mi, sm = -1, -1, 0
        w = self.w[self.bitpos]
        dot = (w[0] * s0 + w[1] * s1 + w[2] * s2 + w[3] * s3 + w[4] * sm + w[5] * 256) >> 16
        pm = squash(dot)
        pa = self.apm.p(pm, c0)
        pr = (pm + 3 * pa) >> 2
        if pr < 1:
            pr = 1
        elif pr > PMAX:
            pr = PMAX
        self._st = (i1, k2, k3, s0, s1, s2, s3, sm, exp_bit, mi, pm)
        return pr

    def update(self, y):
        """Learn from the coded bit y (all components), then advance the bit/byte context."""
        i1, k2, k3, s0, s1, s2, s3, sm, exp_bit, mi, pm = self._st
        lim = self.LIMIT
        c0 = self.c0
        self.t0[c0] = _slot_update(self.t0[c0], y, lim)
        self.t1[i1] = _slot_update(self.t1[i1], y, lim)
        self.t2[k2] = _slot_update(self.t2.get(k2, SLOT_INIT), y, lim)
        self.t3[k3] = _slot_update(self.t3.get(k3, SLOT_INIT), y, lim)
        if mi >= 0:
            self.mm[mi] = _slot_update(self.mm[mi], y, 1023)
            if exp_bit != y:
                self.mlen = 0
        err = ((y << 12) - pm) * self.LR
        w = self.w[self.bitpos]
        w[0] += (s0 * err) >> 14
        w[1] += (s1 * err) >> 14
        w[2] += (s2 * err) >> 14
        w[3] += (s3 * err) >> 14
        w[4] += (sm * err) >> 14
        w[5] += (256 * err) >> 14
        self.apm.update(y)
        c0 = (c0 << 1) | y
        self.bitpos += 1
        if self.bitpos == 8:
            self._end_byte(c0 & 0xFF)
            c0, self.bitpos = 1, 0
        self.c0 = c0

    def _end_byte(self, c):
        """Byte boundary: append to history, extend or look up the longest match of the last MINLEN bytes."""
        buf = self.buf
        if self.mlen > 0 and buf[self.mptr] == c:
            self.mlen += 1
            self.mptr += 1
        else:
            self.mlen = 0
        buf.append(c)
        self.c4 = ((self.c4 << 8) | c) & 0xFFFFFFFF
        pos = len(buf)
        if pos >= self.MINLEN:
            key = self.c4
            if self.mlen == 0:
                cand = self.mtab.get(key)
                if cand is not None:
                    n = 0
                    while n < 32 and cand - 1 - n >= 0 and buf[cand - 1 - n] == buf[pos - 1 - n]:
                        n += 1
                    if n >= self.MINLEN:
                        self.mptr, self.mlen = cand, n
            self.mtab[key] = pos


class BinaryModel:
    """Bit-oriented CM predictor for raw binary input: order-0..8 bit-history contexts, logistic mixer, APM."""

    ORDERS = 9                  # orders 0..8
    N_IN = ORDERS + 1           # + bias
    LIMIT = 1023
    LR = 3

    def __init__(self):
        """Empty model; one table of 512 slots holds all orders (index (1 << k) | last k bits)."""
        self.h = 0
        self.t = [SLOT_INIT] * 512
        self.w = [[int(0.15 * 65536)] * self.N_IN for _ in range(4)]
        self.apm = APM(256)
        self._st = None

    def clone(self):
        """Independent copy of the model state."""
        m = BinaryModel.__new__(BinaryModel)
        m.h, m.t = self.h, self.t[:]
        m.w = [row[:] for row in self.w]
        m.apm = APM.__new__(APM)
        m.apm.t, m.apm.rate, m.apm.idx = self.apm.t[:], self.apm.rate, 0
        m._st = None
        return m

    def cap_confidence(self, cap):
        """Limit slot counts after warm start so the model re-adapts quickly to the actual source."""
        self.t = [v if (v & 1023) <= cap else (v & ~1023) | cap for v in self.t]

    def p(self):
        """Predict P(next bit = 1) as a 12-bit integer."""
        h = self.h
        idx = [(1 << k) | (h & ((1 << k) - 1)) for k in range(self.ORDERS)]
        st = [STRETCH[self.t[i] >> 14] for i in idx]
        st.append(256)
        w = self.w[h & 3]
        pm = squash(sum(a * b for a, b in zip(w, st)) >> 16)
        pa = self.apm.p(pm, h & 0xFF)
        pr = (pm + 3 * pa) >> 2
        pr = 1 if pr < 1 else (PMAX if pr > PMAX else pr)
        self._st = (idx, st, pm)
        return pr

    def update(self, y):
        """Learn from bit y and shift it into the bit history."""
        idx, st, pm = self._st
        for i in idx:
            self.t[i] = _slot_update(self.t[i], y, self.LIMIT)
        err = ((y << 12) - pm) * self.LR
        w = self.w[self.h & 3]
        for j in range(self.N_IN):
            w[j] += (st[j] * err) >> 14
        self.apm.update(y)
        self.h = ((self.h << 1) | y) & 0xFFFF


# ----------------------------------------------------------------------------- warm start
# Original text written for this project (not copied from any source).  Both
# encoder and decoder pre-train on it identically, so it is shared prior knowledge.
WARM_TEXT = (
    "Hello, my friend. How are you today? I am fine, thank you. My name is on the "
    "list, and I will be there at the meeting in the morning. Please send me a "
    "message when you are ready, and we can talk about the plan for the week.\n"
    "The weather is good today. The sun is bright and the sky is clear, but the "
    "wind is cold near the river. In the evening it may rain, so take an umbrella "
    "with you if you go out. We are going to the market to buy some bread, milk, "
    "rice, eggs, fruit and vegetables for the family.\n"
    "This is a simple test of a live communication system. A message is typed by "
    "the user, compressed by the source encoder, protected by the channel encoder, "
    "sent through a noisy channel, and then decoded at the receiver. If everything "
    "works, the receiver gets back exactly the same message that was sent. The "
    "system counts the number of bits, the number of errors, and the time taken.\n"
    "Information theory tells us how many bits we need to describe a source and "
    "how many bits per second we can send reliably over a channel. The entropy of "
    "a source is the average amount of information in each symbol. The capacity "
    "of a channel is the highest rate at which we can communicate with an error "
    "probability that is as small as we like. A good code works close to these "
    "limits, and a good decoder finds the most likely message that was sent.\n"
    "Thank you for your help with the project. I have finished the first part of "
    "the work and I will start the second part tomorrow. Can you call me after "
    "class? My phone number is in the email that I sent yesterday. See you soon, "
    "and have a nice day!\n"
    "Where are you from? I am from a small town in the south, but now I live and "
    "study in the city. I like to read books, play cricket and football, listen "
    "to music and watch movies with my friends. What do you like to do in your "
    "free time? Do you have any brothers or sisters? My brother is a doctor and "
    "my sister is a teacher at a school near our house.\n"
    "The quick brown fox jumps over the lazy dog. Every good student should check "
    "the result twice before writing the final answer. The data were collected "
    "over many days, and the numbers show a clear trend: as the noise level goes "
    "up, the error rate also goes up, unless the code is strong enough to correct "
    "the errors. When the rate of the code is below the capacity, long codes can "
    "make the error rate very small.\n"
    "Good morning, good afternoon, good evening and good night. Yes, no, maybe, "
    "please, sorry, thanks, okay. I love you. I miss you. Happy birthday! Welcome "
    "home. Let us meet at the station at nine o'clock. The train leaves at ten "
    "and arrives in the city before noon. It is the first time that I travel "
    "alone, and I am a little nervous, but I think that it will be a great day.\n"
    "Numbers: 0 1 2 3 4 5 6 7 8 9 10 20 50 100 2024 2025 2026. Questions: who, "
    "what, when, where, why, how? People often send short messages such as hi, "
    "hello there, what is up, call me, on my way, see you later, all is well, "
    "the job is done, and the system is working.\n"
)
WARM_CAP = 12
_WARM_CACHE = {}


def _warm_bits():
    """The warm-start text as a list of bits (used to pre-train the binary model)."""
    out = []
    for b in WARM_TEXT.encode("utf-8"):
        out.extend((b >> (7 - i)) & 1 for i in range(8))
    return out


def new_model(mode, warm):
    """Fresh model for 'text' or 'binary' mode; if warm, a copy of the model pre-trained on WARM_TEXT.

    Pre-training runs only predict/update (no coding) so encoder and decoder obtain
    bit-identical states.  The result is cached and cloned (the cache is derived
    solely from the built-in text, which the receiver also has).
    """
    cls = TextModel if mode == "text" else BinaryModel
    if not warm:
        return cls()
    if mode not in _WARM_CACHE:
        m = cls()
        if mode == "text":
            for byte in WARM_TEXT.encode("utf-8"):
                for i in range(8):
                    m.p()
                    m.update((byte >> (7 - i)) & 1)
        else:
            for b in _warm_bits():
                m.p()
                m.update(b)
        m.cap_confidence(WARM_CAP)
        _WARM_CACHE[mode] = m
    return _WARM_CACHE[mode].clone()


# ----------------------------------------------------------------------------- public API
def _header(mode, warm, count):
    """32-bit header: bit 31 = mode (0 text, 1 binary), bit 30 = warm start, bits 29..0 = symbol count."""
    v = ((1 if mode == "binary" else 0) << 31) | ((1 if warm else 0) << 30) | count
    return [(v >> (31 - i)) & 1 for i in range(32)]


def compress(data, mode="text", warm=True):
    """Compress bytes (mode='text') or a 0/1 sequence (mode='binary'); returns a list of bits incl. the 32-bit header."""
    if mode == "text":
        data = bytes(data)
    elif mode == "binary":
        data = [int(b) for b in data]
        if any(b not in (0, 1) for b in data):
            raise ValueError("binary input must contain only 0 and 1")
    else:
        raise ValueError("mode must be 'text' or 'binary'")
    if len(data) > MAX_COUNT:
        raise ValueError("input too long for the 30-bit length field")
    model = new_model(mode, warm)
    enc = ArithmeticEncoder()
    if mode == "text":
        for byte in data:
            for i in range(8):
                y = (byte >> (7 - i)) & 1
                enc.encode(y, model.p())
                model.update(y)
    else:
        for y in data:
            enc.encode(y, model.p())
            model.update(y)
    return _header(mode, warm, len(data)) + enc.finish()


def parse_header(bits):
    """Read (mode, warm, count) from the first 32 bits."""
    if len(bits) < HEADER_BITS:
        raise ValueError("stream shorter than the 32-bit header")
    v = 0
    for b in bits[:HEADER_BITS]:
        v = (v << 1) | int(b)
    return ("binary" if v >> 31 else "text"), bool((v >> 30) & 1), v & MAX_COUNT


def decompress(bits, max_count=1 << 22):
    """Inverse of compress(): returns (mode, warm, data) using only the received bits (and the shared warm text).

    Trailing zero padding is harmless (the arithmetic decoder reads zeros past the end).
    A corrupted header announcing more than max_count symbols raises ValueError.
    """
    bits = [int(b) for b in bits]
    mode, warm, count = parse_header(bits)
    if count > max_count:
        raise ValueError(f"header announces {count} symbols (> {max_count}); stream is corrupted")
    model = new_model(mode, warm)
    dec = ArithmeticDecoder(bits, HEADER_BITS)
    if mode == "text":
        out = bytearray()
        for _ in range(count):
            c = 0
            for _ in range(8):
                y = dec.decode(model.p())
                model.update(y)
                c = (c << 1) | y
            out.append(c)
        return mode, warm, bytes(out)
    out = []
    for _ in range(count):
        y = dec.decode(model.p())
        model.update(y)
        out.append(y)
    return mode, warm, out
