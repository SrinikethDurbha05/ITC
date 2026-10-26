# ITC Live Communication System

**Context-mixing source coding + CRC-aided polar codes (SC-List) over a noisy wired channel**

A from-scratch Python implementation (numpy + matplotlib only) of a complete digital
communication chain for a "Theory of Coding" assignment. Every number in this README was
produced by the code in this folder (commands are given next to each result).
No Huffman, Shannon-Fano, Hamming or other textbook-classical code is used anywhere.
The baselines are uncoded transmission, plain SC decoding, and theoretical limits.
zlib and LZMA appear only as reference bars in Fig. 5.

---

## 1. Pipeline

```
 TRANSMITTER                                                                RECEIVER
 ───────────                                                                ────────
 text (UTF-8 bytes) ─┐                                                   ┌─> recovered text / bits
 or 0/1 sequence  ───┤                                                   │
                     v                                                   │
      ┌───────────────────────────────┐                   ┌───────────────────────────────┐
      │ [A] context-mixing compressor │                   │ [A] context-mixing decoder    │
      │  orders 0-3 + match model     │                   │  same model, same integers    │
      │  logistic mixer + APM         │                   │  reads the 32-bit header      │
      │  32-bit binary arithmetic coder│                  │                               │
      └──────────────┬────────────────┘                   └───────────────^───────────────┘
        32-bit header + payload bits                         payload bits, padding ignored
                     v                                                    │
      ┌───────────────────────────────┐                   ┌───────────────┴───────────────┐
      │ split into blocks of K-r bits │                   │ CA-SCL decoder (list L)       │
      │ zero-pad the last block       │                   │ pick the best path whose CRC  │
      │ append CRC-r  (CRC-11)        │                   │ passes, else flag CRC FAILED  │
      │ place on the K best channels  │                   │ (plain SC is run on the same  │
      │ x = u·F^{⊗n}, N = 256         │                   │  LLRs for comparison)         │
      └──────────────┬────────────────┘                   └───────────────^───────────────┘
                     v                                                    │ LLRs
      ┌──────────────────────────────────────────────────────────────────┴──────┐
      │ [C] channel: lossless | BSC(p) | BPSK over AWGN (Eb/N0, R_eff)           │
      └──────────────────────────────────────────────────────────────────────────┘
```

| File | Content |
|---|---|
| `cm_coder.py` | bitwise context-mixing model (text: orders 0-3 + match model; binary: bit-history orders 0-8), logistic mixer, APM/SSE, 32-bit carry-less binary arithmetic coder, warm start, 32-bit header |
| `crc.py` | bitwise CRC-6 / CRC-11 / CRC-24C (3GPP TS 38.212 polynomials), long division and a vectorised linear-map version |
| `polar.py` | butterfly encoder, Bhattacharyya and GA frozen-set design, SC, CA-SCL with lazy copying, *optional* PAC codes |
| `channel.py` | lossless, BSC(p), BI-AWGN channel models with LLRs |
| `theory.py` | BSC capacity and dispersion, PPV normal approximation, BI-AWGN capacity and dispersion (quadrature), Shannon limit by bisection, uncoded BPSK |
| `utils.py` | entropy estimators H0/H1, bit helpers, the evaluation text for Fig. 5 |
| `main.py` | live CLI (interactive or scripted) |
| `plot_results.py` | Monte-Carlo simulations (multiprocessing), Figs. 1-7, `results.csv`, `source_coding.csv` |
| `tests/test_all.py` | 36 unit and integration tests |

---

## 2. How to run

Requirements: Python 3.10+ with `numpy` and `matplotlib` (developed on Python 3.14.4, numpy 2.4.4, matplotlib 3.11.2, Windows 11).

```bash
python -m unittest -v                     # all tests (~15-50 s)

python main.py                            # interactive live demo (loops with "Run again? (y/n)")
python main.py --text "my name is sunny" --channel bsc --p 0.05 --seed 42      # scripted, no prompts
python main.py --binary 0001000000100000 --channel awgn --ebn0 1.5 --seed 7
python main.py --help                     # --N 64|128|256|512  --K  --crc 6|11|24  --L 1..32  --no-warm  --design-ebn0

python plot_results.py --quick            # fast run: <= 2 000 frames per point (~20 s on 12 cores)
python plot_results.py                    # full run: <= 50 000 frames per point
python plot_results.py --no-show --outdir results --seed 42 --workers 8
```

The interactive program asks, in this order, for: the input type (text or binary), the
message, the channel (1 lossless, 2 BSC, 3 BPSK/AWGN), p or Eb/N0 (and the design Eb/N0
for AWGN), the **random seed (default 42)**, and whether to use the default code
(N=256, K=128, CRC-11, L=8, warm start ON). If you say no, it asks for N, the CRC, K, L
and warm start. All inputs are validated: binary input must contain only 0/1, p must lie
in (0, 0.5), K must satisfy r < K <= N, and so on. The seed drives only the channel noise,
so a run is fully reproducible. `plot_results.py` seeds every simulation point with
`(seed, point id)`, so its results do not depend on how the processes are scheduled.

---

## 3. Theory and design

### 3.1 Source coding: context mixing + arithmetic coding (`cm_coder.py`)

*Source coding theorem.* A uniquely decodable code needs on average at least H bits per
symbol for a source of entropy H. An arithmetic coder fed with model probabilities P
spends about -log2 P(x) bits plus a few bits of termination. Compression is therefore
exactly as good as the model's predictions. (A unit test checks that the coder stays
within 1% + 40 bits of the ideal -log2 P length.)

*Model (PAQ/LPAQ lineage, Mahoney).* The model predicts one bit at a time, MSB-first
within each byte.
* **Text mode** has four context models:
  * order-0: the partial byte;
  * order-1: the previous byte plus the partial byte;
  * order-2 and order-3: hash tables keyed by the previous 2 or 3 bytes plus the partial byte.

  Each slot holds a 16-bit probability and a hit count n, and is updated by
  `p += (y - p)/(n + 1.5)`, an adaptive learning rate capped at n = 255. A **match model**
  looks up the most recent occurrence of the last 4 bytes, verifies the match backwards
  (up to 32 bytes) and predicts the next bit of the byte that followed it. Its confidence
  is learnt per match length.
* **Binary mode** uses the same machinery with order-0...order-8 contexts over the bit history.
* **Logistic mixer:** `st_i = stretch(p_i) = ln(p_i/(1-p_i))`, `p = squash(Σ w_i st_i)`,
  `w_i += lr·(y - p)·st_i`. There is one weight set per bit position (text) or per
  last-2-bits context (binary).
* **APM/SSE:** 33 interpolated buckets per partial-byte context. The final probability is
  `(p_mix + 3·p_APM)/4`, clamped to [1, 4095]/4096.
* **Determinism:** probabilities (12/16-bit), stretch/squash tables (integer interpolation),
  mixer weights (16.16 fixed point) and the coder are all integers. The decoder runs the
  identical model on the bits it has already decoded and reproduces every probability
  exactly.
* **Arithmetic coder:** a 32-bit carry-less binary coder,
  `xmid = x1 + (x2-x1)·p/4096`. Bytes are shifted out when the top bytes of x1 and x2
  agree. The termination is the *shortest* bit string t such that t000... lies in the
  final interval, which works because the decoder reads zeros past the end. That also
  makes the zero padding of the last polar block harmless.
* **Header:** 32 bits = `[mode bit | warm-start bit | 30-bit symbol count]`. It is counted
  in every reported compressed size and is channel-coded together with the payload. The
  receiver therefore learns the mode, the warm-start flag and the length from the
  bitstream alone.
* **Warm start (default ON):** both ends pre-train on a built-in English text of
  2 979 bytes. I wrote it for this project (`cm_coder.WARM_TEXT`). After pre-training,
  counts are capped at 12 so the model re-adapts quickly. The receiver has the same text,
  so no side information is used. The tests clear every cached model before decoding.
  Fig. 5 is measured on a *different* text (`utils.EVAL_TEXT`), so the warm-start gain
  is not inflated by memorisation.

### 3.2 Channel coding: CRC-aided polar codes (`crc.py`, `polar.py`)

*Noisy-channel coding theorem.* Every rate R < C is achievable with error probability
→ 0 as the blocklength grows, and no rate R > C is.

*Channel polarization (Arıkan 2009).* Combining two copies of W with F = [[1,0],[1,1]]
and decoding them successively yields a worse channel W⁻ (check node, f) and a better
channel W⁺ (variable node, g). Applying the transform recursively n = log2 N times makes
the N synthetic channels polarize: as N → ∞ a fraction C becomes noiseless and the rest
useless. Information bits are sent on the K most reliable synthetic channels, and the
remaining (frozen) inputs are fixed to 0.

**Conventions (used everywhere).**
* `x = u·F^{⊗n}` with **no bit-reversal permutation**, computed by an in-place
  O(N log N) butterfly. A test checks it against the explicit Kronecker matrix for
  N = 2...32.
* The decoder splits the LLR vector into halves (a | b). The left child gets f(a, b) and
  the right child gets g = b + (1-2β_L)a. The MSB of index i is therefore the first
  transform applied to the raw channel: index 0 = W⁻⁻…⁻ is the worst and N-1 = W⁺⁺…⁺ the
  best.
* The reliability recursions build the index in the same order. This is *verified*: for
  the BEC, the genie-aided erasure rate of every u_i measured through the decoder's own
  recursion matches the exact Bhattacharyya value (max. difference 0.0024 over 2·10⁵
  frames, test `test_bec_genie_matches_bhattacharyya`).
* LLR = ln P(y|0)/P(y|1). BSC: `(1-2y)·ln((1-p)/p)`. BI-AWGN (0→+1, 1→−1): `2y/σ²`.

**Frozen-set design** (information set = the K most reliable synthetic channels, computed
here, not copied from the 5G table):
* *BSC, Bhattacharyya:* Z₀ = 2√(p(1-p)), Z(W⁻) = 2Z − Z², Z(W⁺) = Z². The recursion is
  **exact only for the BEC**. For the BSC and other channels, 2Z − Z² is an upper bound on
  Z(W⁻), so the resulting ranking is a heuristic. The recursion is carried in
  (ln Z, ln(1−Z)) because in plain floating point the worst channels all round to Z = 1
  and tie. (I hit this bug: index 0 was not ranked last until the fix.)
* *BI-AWGN, Gaussian approximation (Trifonov 2012):* m₀ = 2/σ², m(W⁺) = 2m,
  m(W⁻) = φ⁻¹(1 − (1 − φ(m))²). φ uses the Chung–Richardson–Urbanke approximation, and a
  unit test checks it against numerical quadrature of φ(x) = E[2/(1+e^U)], U ~ N(x, 2x),
  agreeing within 5%. For m < 0.05, where Chung's formula is poor, the small-mean
  asymptote 1 − φ(m) ≈ m/2 gives m(W⁻) = m²/2. Means are carried as ln m.
  σ² = 1/(2·R_eff·10^{EbN0/10}) at the design Eb/N0 (default 2.5 dB).
* *Self-test* (shown live as "Frozen-set validation"): |A| = K, argmax reliability = N−1,
  argmin = 0, and the all-zero codeword decodes to zero with SC and SCL.

**CRC.** Each block carries K − r payload bits followed by an r-bit CRC, placed on the
information positions in increasing index order. CRC-11 = 0xE21
(x¹¹+x¹⁰+x⁹+x⁵+1, 5G NR uplink) is the default. CRC-6 = 0x61 and CRC-24C = 0x1B2B117 are
also supported. Tests check that payload‖CRC has zero remainder, that every single-bit
error is detected, and that every burst of length ≤ r is detected.

**Effective code rate** (used for every Eb/N0 and every capacity comparison):
**R_eff = (K − r)/N** payload bits per channel use. The default is 117/256 = **0.457**.

**Decoders.**
* *SC:* recursive f/g in the LLR domain, min-sum f by default (exact box-plus optional).
  All-frozen subtrees are skipped, since their decisions are 0 regardless of the LLRs.
* *CA-SCL (Tal–Vardy):* every path carries a path metric PM = Σ ln(1 + e^{−(1−2û_i)L_i}),
  a negative log-likelihood penalty. At each information bit every path forks and the L
  best of the 2L candidates survive. **Lazy copying:** a recursion node returns the
  permutation "survivor j descends from path perm[j]", and its parent gathers only the
  arrays it still needs. Nothing is deep-copied at a fork, and all frames of a batch are
  decoded together (arrays of shape B × L × n). At the end, the best-metric path whose
  CRC passes is chosen; if none passes, the best path is output and the block is flagged
  **CRC FAILED**.
* *Invariant:* ties between the two children of a path are broken towards the
  hard-decision bit, so **SCL with L = 1 is bit-identical to SC**. A test checks this on
  noisy BSC and AWGN data for several N, with and without CRC.

### 3.3 Channels (`channel.py`): why these are wired-channel models

* **BSC(p):** a wired link whose receiver makes hard decisions (a comparator or slicer at
  the line receiver, as in most digital interfaces) delivers bits that are wrong with
  some probability p. Memoryless independent flips are the standard first-order model of
  such a link. LLRs then have a single magnitude ln((1−p)/p), so the decoder gets no
  soft information.
* **BPSK over AWGN (BI-AWGN):** the dominant impairment of a short copper or coax link
  with matched filtering is thermal (Johnson–Nyquist) noise at the receiver, which is
  white and Gaussian. Antipodal signalling (0→+1, 1→−1) with
  **σ² = 1/(2·R·10^{Eb/N0/10})** gives the binary-input AWGN channel. Here
  R = R_eff = (K−r)/N for the coded system and R = 1 for uncoded BPSK, so all schemes are
  compared at the same energy per *payload* bit.
* **Lossless:** y = x, with LLR = ±20.

Channel bit errors are always counted on hard decisions at the channel output.

### 3.4 Finite-blocklength benchmarks (`theory.py`)

* BSC capacity C = 1 − h2(p). A test checks C(0.11) ≈ 0.5. The threshold p* for
  R_eff = 0.457 is p* = 0.1248 (bisection).
* BSC dispersion V = p(1−p)·log2²((1−p)/p).
* **Polyanskiy–Poor–Verdú normal approximation:**
  R*(n, ε) ≈ C − √(V/n)·Q⁻¹(ε) + log2(n)/(2n). The inverse form predicts the FER of an
  *ideal* code: ε ≈ Q((C − R + log2(n)/(2n)) / √(V/n)). This is an **approximation, not
  a bound**: it is least reliable at very short n and at very small or very large ε. For
  BI-AWGN the same formula is used with the BI-AWGN capacity and dispersion
  V = Var[i(X;Y)], both computed by 200-point Gauss–Hermite quadrature (cross-checked
  against a 200 001-point trapezoidal grid). Treating the log2(n)/(2n) third-order term
  as accurate for the binary-input AWGN channel is itself an approximation.
* BI-AWGN Shannon limit for rate R: the smallest Eb/N0 with C(σ²) ≥ R, by bisection. It
  gives 0.187 dB for R = 1/2 (the textbook value) and **−0.002 dB for R_eff = 0.457**,
  drawn as a vertical line. Nothing is hard-coded.
* Uncoded BPSK BER = Q(√(2Eb/N0)), verified by a Monte-Carlo test (2·10⁶ bits, within 3%).
  It reaches BER = 10⁻³ at 6.790 dB.

---

## 4. Live demo (`main.py`)

`python main.py --text "my name is sunny" --channel bsc --p 0.05 --seed 42` (full output
in `results/demo_bsc.txt`; the interactive mode prints the same report):

```
[2] CONTEXT-MIXING SOURCE CODING
Warm start             : ON
Original source bits   : 128
Compressed bits        : 80  (includes 32-bit length header)
Compression ratio      : 1.600 : 1
Size reduction         : 37.50%
Average code length    : 5.0000 bits/symbol
Without warm start     : 139 bits
...
[4] CHANNEL TRANSMISSION
Channel type           : BSC (p = 0.05)
Channel bit errors     : 12
Channel BER            : 0.046875
...
[6] FINAL RESULTS
Errors corrected by decoder : 12
Source BER             : 0.0000000000
Capacity check         : R_eff = 0.457 < C = 1 - h2(p) = 0.7136
Finite-blocklength     : predicted FER of ideal code (N=256) = 1.275e-06  (normal approximation)

RESULT: PASS
```

Notes on the report fields:
* Without warm start, this 16-byte message needs 139 bits, *more* than its 128 raw bits.
  The program says so whenever the compressed size is not smaller than the raw size.
* "Blocks rescued by list" runs plain SC on the same received LLRs and counts blocks that
  SC got wrong and CA-SCL got right. This uses the transmitted data, so it is an
  evaluation-only metric. For example, the AWGN run at 1.0 dB with seed 42 shows 1 rescued
  block (`results/demo_awgn.txt`).
* "Errors corrected by decoder" counts codeword positions where the hard decision was
  wrong and the decoded codeword is right.
* On failure the program prints `RESULT: FAIL` and the reason (`CRC FAILED in block(s) ...`,
  a length mismatch, or a corrupted header).

---

## 5. Results (`python plot_results.py`, seed 42)

Full run: 102 simulation points, 388 s on 11 worker processes. Each point stopped after
100 frame errors or 50 000 frames. Raw counts, frames and errors for every point are in
`results_full/results.csv` and `results_full/raw_results.json`, and the console output is
in `results_full/full_log.txt`. Code: N = 256, K = 128 with CRC-11 (117 payload bits),
**R_eff = 117/256 = 0.457** for every coded scheme. AWGN frozen sets use the GA at a
design Eb/N0 of 2.5 dB; BSC frozen sets use Bhattacharyya at the channel p.

### Figures (in `results_full/`; `--quick` writes the same set to `results/`)

| Figure | Content |
|---|---|
| `fig1_ber_awgn.png` | **Fig. 1:** BER vs Eb/N0 (0–5 dB): uncoded BPSK (simulated + Q(√(2Eb/N0))), polar SC (L=1, no CRC, K=117), CA-SCL L=8 (CRC-11), plus SCL L=8 without CRC and CA-SCL L=8 with CRC-6 (both added to explain the CRC trade-off below). Vertical line: BI-AWGN Shannon limit for R=0.457. |
| `fig2_fer_awgn.png` | **Fig. 2:** FER for the same schemes, plus the normal approximation for an ideal (256,117) code and the uncoded 117-bit frame error rate. |
| `fig3_fer_bsc.png` | **Fig. 3:** BSC, FER vs p for SC and CA-SCL L=8, the normal approximation, and the capacity threshold p* = 0.1248. |
| `fig4_list_size.png` | **Fig. 4:** FER vs L ∈ {1,…,32} at 2.0 dB with CRC-11 (error counts annotated), and decoding time per block. |
| `fig5_source_coding.png` | **Fig. 5:** bits/symbol of CM (warm / cold), empirical H0 and H1, raw-deflate and raw-LZMA2 references on 16/64/256/1024 bytes of English and a P(1)=0.1 binary source (10 000 bits), plus compression ratios. |
| `fig6_polarization.png` | **Fig. 6:** polarization of N=256 channels: Bhattacharyya Z by index (BSC, p=0.05, frozen/info coloured), sorted Z, and sorted GA means (2.5 dB), with the frozen/information split marked. |
| `fig7_pac_optional.png` | **Fig. 7 (optional Tier 2):** PAC vs CA-polar at L=8 and L=32. |

All curves have 95% error bars: Wilson intervals for FER. For BER the interval is
computed over frames, because bit errors cluster within frames; it is a Wilson interval
scaled by BER/FER when fewer than 20 frames were in error. A point with zero errors is
drawn as a hollow ▽ at the 95% upper bound 3/frames ("rule of three"). It is a floor
marker, not a measurement.

### Key numbers (all printed by `plot_results.py`)

| Scheme (R_eff = 0.457, N = 256) | Eb/N0 at BER 10⁻³ | Coding gain vs uncoded (6.790 dB, theory) | Eb/N0 at FER 10⁻³ | Gap to normal approx. (1.80 dB) |
|---|---|---|---|---|
| Polar SC (L=1, no CRC) | 3.19 dB | 3.60 dB | 3.75 dB | 1.95 dB |
| SCL L=8, no CRC | 2.42 dB | 4.37 dB | 2.93 dB | 1.13 dB |
| **CA-SCL L=8, CRC-11** | 2.55 dB | 4.24 dB | 2.80 dB | 1.00 dB |
| CA-SCL L=8, CRC-6 | 2.32 dB | 4.47 dB | 2.63 dB | 0.83 dB |
| CA-SCL L=32, CRC-11 (Fig. 7) | – | – | 2.50 dB | 0.69 dB |
| PAC L=8 (optional) | – | – | 2.86 dB | 1.06 dB |
| PAC L=32 (optional) | – | – | 2.58 dB | 0.78 dB |

(The BER-crossing column is 6.790 minus the gain. All crossings are log-linear
interpolations between adjacent simulated points.) The BI-AWGN Shannon limit for
R = 0.457 is −0.002 dB. Uncoded BPSK does not reach BER 10⁻³ in the 0–5 dB window, so its
crossing is the analytical value.

**BSC (Fig. 3).** At p = 0.05, SC has FER 3.45·10⁻² (107/3 100 frames) and CA-SCL L=8 has
3.95·10⁻³ (101/25 600). The normal approximation predicts 1.3·10⁻⁶ for an ideal code. At
p = 0.03, CA-SCL had 1 error in 50 000 frames and SC had 23. CA-SCL is better than SC over
the whole range. Both approach FER ≈ 0.76 at p = 0.12, just below the capacity threshold
p* = 0.1248.

**List size (Fig. 4, 2.0 dB, CRC-11).** FER was 0.247 (L=1), 0.115, 0.055, 0.034, 0.020 and
0.0125 (L=32). The normal approximation predicts 3.0·10⁻⁴. Decoding time per block in the
main process (Python/numpy, batched 200 blocks per call) was 0.067 / 0.111 / 0.201 /
0.383 / 0.748 / 1.592 ms for L = 1…32, so it grows roughly linearly in L. Plain SC took
0.030 ms. Decoding *one block per call*, which is the latency of the live demo, costs
5.2–6.7 ms for every L, because Python per-node overhead dominates.

**Source coding (Fig. 5)**, in bits/symbol including the 32-bit header (deflate and LZMA2
without container headers, which favours them):

| Input | CM warm | CM cold | raw deflate | raw LZMA2 | H0 | H1 (plug-in) |
|---|---|---|---|---|---|---|
| English 16 B | 5.625 | 8.562 | 9.000 | 10.000 | 3.203 | 0.533 |
| English 64 B | 4.078 | 6.031 | 7.125 | 8.625 | 3.909 | 1.731 |
| English 256 B | 3.234 | 4.719 | 5.250 | 6.406 | 4.143 | 2.614 |
| English 1024 B | 2.863 | 3.725 | 4.219 | 4.789 | 4.217 | 3.027 |
| Binary P(1)=0.1, 10⁴ bits | 0.490 | 0.478 | 0.598 | 0.684 | 0.468 | 0.468 |

* Context mixing beats both references on every input.
* Warm start is essential for short English messages: without it, 16 bytes **expand**
  (8.56 bits/byte > 8).
* At 1 024 bytes the cold model already beats H0 (3.73 < 4.22 bits/byte), because it
  exploits context.
* The empirical H1 is *below* the CM rate for short texts. That does not mean the coder is
  bad: the plug-in H1 of a 16-byte string (0.53) ignores the cost of learning the
  conditional distribution, and no universal code can achieve it.
* For the binary source, cold CM gets to within 0.010 bits/bit of H0 = h2(p̂) = 0.468 (0.003
  of it is the header).
* **Warm start *hurts* the binary source (0.490 vs 0.478).** Pre-training on the bits of
  English text is a mismatched prior for an i.i.d. Bernoulli source. This is reported as
  measured, not tuned away.

### Investigation: why is CRC-11 not always better than no CRC?

At low and medium SNR, Fig. 2 shows CA-SCL L=8 with CRC-11 *worse* than plain SCL L=8 at
the same R_eff. At 2.0 dB the FER is 2.83·10⁻² vs 1.96·10⁻², and at 2.5 dB it is 5.2·10⁻³ vs 4.3·10⁻³. CRC-11
only wins from somewhere between 2.5 and 3.0 dB (3.0 dB: 3.4·10⁻⁴ vs 8.0·10⁻⁴). This looked suspicious, so I measured *why*.
The decoder reports, for every frame error, whether the transmitted vector was still in
the final list. That is the last column of `results.csv`:

| 2.0 dB | frame errors | errors with the correct path still in the list |
|---|---|---|
| SCL L=8, no CRC (K=117) | 100 / 5 100 | 74 |
| CA-SCL L=8, CRC-6 (K=123) | 104 / 7 600 | 5 |
| CA-SCL L=8, CRC-11 (K=128) | 102 / 3 600 | **0** |

Without a CRC, most errors are *selection* errors: the right codeword was in the list but
another path had a better metric. The CRC removes almost all of those, so it works as
intended. However, to keep R_eff fixed, the CRC bits must be carried on 11 extra synthetic
channels, so the K=128 code uses less reliable channels than the K=117 code. Its correct
path is then *lost from the list* more often. At N=256 and L=8 this rate loss outweighs
the selection gain until high SNR. A 6-bit CRC balances the two better and is the best
L=8 scheme at every simulated SNR from 1.5 to 3.0 dB. At 3.5 dB and above, the error counts
are too small to rank CRC-6 and CRC-11. With L=32 (Fig. 7), CRC-11 pays off: it gains
0.30 dB at FER 10⁻³ over L=8. This is the known finite-length trade-off: the CRC length
should be matched to the list size and blocklength. It is a property of the code, not a
bug. I kept CRC-11 as the default because the assignment specifies it.

**PAC (optional Tier 2, Fig. 7).** With an RM rate profile, the octal-133 precoder and the
same L, PAC is slightly better than CA-polar with CRC-11 at low SNR. At L=8 it is
comparable at FER 10⁻³ (2.86 vs 2.80 dB), and at L=32 it is slightly worse (2.58 vs
2.50 dB). The large PAC gains reported in the literature use list sizes of 128 or more, or
Fano sequential decoding, and **are not reached at L ≤ 32 here.** PAC also has no CRC, so
it cannot flag failed blocks.

---

## 6. Tests (`python -m unittest -v`, 36 tests)

* **CRC:** zero remainder, the known value of CRC(1), every single-bit error and every
  burst ≤ r detected (all three polynomials), and the batch version equals long division.
* **Theory:**
  * C(0.11) = 0.5 ± 10⁻³ and the BSC threshold;
  * the normal-approximation FER decreases as p decreases, and its inverse is consistent;
  * Q / Q⁻¹;
  * BI-AWGN capacity by quadrature equals the fine-grid value, and the R=1/2 limit is in
    (0.1, 0.3) dB;
  * uncoded BPSK BER equals Monte-Carlo within 3%.
* **Context mixing:**
  * round trips with warm start on and off on: empty input, 1 byte, random bytes, English,
    emoji and non-ASCII text, 5 200 bytes, 200 zero bytes, and binary inputs (empty, 1 bit,
    random, skewed, periodic);
  * the receiver decodes after the model cache is cleared;
  * padding is harmless and a corrupted header is rejected;
  * arithmetic coder with extreme or out-of-range probabilities, including coding the
    "impossible" symbol;
  * coder cost ≤ ideal + 1%;
  * squash/stretch consistency.
* **Polar:**
  * fast encoder = Kronecker matrix (N = 2…32) and G² = I;
  * BEC genie test of the bit-ordering convention;
  * Chung φ vs quadrature;
  * frozen-set validation for N = 64…512 and BSC/AWGN designs;
  * noiseless round trip for N = 64…512 with every CRC;
  * exact box-plus option;
  * **SCL(L=1) ≡ SC bit for bit**;
  * with 14 injected errors per block, CA-SCL rescues blocks SC fails (fixed seed);
  * FER decreases with L.
* **PAC:** generator = octal 133, the precoder is invertible (unit-diagonal Toeplitz), the
  RM profile weights, and noiseless and noisy decoding.
* **Channels and end-to-end:**
  * BSC flip rate, AWGN mean, variance and LLR, and input validation;
  * **determinism:** identical seeds give identical received bits and results, different
    seeds give different noise;
  * **200 random inputs** (text with emoji and non-ASCII characters, and binary) over the
    lossless channel with random N, CRC, L and warm start: all PASS;
  * noisy PASS and FAIL reporting.

---

## 7. Limitations (honest)

* **Short messages.**
  * The 32-bit header alone is 2 bits/byte for a 16-byte message.
  * Without warm start, short inputs *expand*; with warm start, gains depend on how much
    the message resembles the built-in text.
  * The binary mode's warm start is a mismatched prior and slightly hurts i.i.d. sources.
  * A message shorter than one block still costs a full 256-bit codeword (e.g. 80
    compressed bits + 37 padding bits).
* **Python speed.**
  * CM coding runs at roughly 0.2 Mbit/s of source data (1 024 bytes in about 35 ms).
  * Polar decoding is fast only when many blocks are batched (0.38 ms/block at L=8); one
    block per call costs about 5–7 ms because of interpreter overhead.
  * FERs below about 10⁻⁵ were not reachable in the allotted frames, as the ▽ markers show.
* **Normal approximation.** It is an approximation, not a bound. It is least accurate at
  short blocklengths and very low error rates, and for BI-AWGN the log2(n)/(2n) term is
  itself an approximation. The plotted curve is a reference for an *ideal* code, not a
  guarantee.
* **Frozen-set design.**
  * Bhattacharyya is exact only for the BEC; for the BSC it ranks channels by an upper
    bound (a heuristic).
  * GA relies on the Chung φ approximation and a small-mean asymptote.
  * One fixed AWGN design point (2.5 dB) is used for all Eb/N0.
  * No 5G sequence, rate matching or puncturing is used (no rate matching is needed here
    because N is a power of two).
* **Path metric and f-function.** The min-sum f (the default) costs some performance
  compared with exact box-plus, which is available via `f_mode="exact"`.
* **CRC-11 at N=256, L=8** is not the best trade-off (Section 5); CRC-6 performs better here.
* **"Blocks rescued" and "errors corrected"** use knowledge of the transmitted data, so
  they are evaluation metrics, not receiver-side quantities.
* Floating-point LLR decoding is deterministic on a given machine and numpy build. Only
  the source coder is guaranteed bit-exact across platforms, because it uses integer
  arithmetic only.

---

## 8. References

1. E. Arıkan, "Channel Polarization: A Method for Constructing Capacity-Achieving Codes for
   Symmetric Binary-Input Memoryless Channels," *IEEE Trans. Inf. Theory*, vol. 55, no. 7,
   pp. 3051–3073, 2009.
2. I. Tal and A. Vardy, "List Decoding of Polar Codes," *IEEE Trans. Inf. Theory*, vol. 61,
   no. 5, pp. 2213–2226, 2015.
3. Y. Polyanskiy, H. V. Poor and S. Verdú, "Channel Coding Rate in the Finite Blocklength
   Regime," *IEEE Trans. Inf. Theory*, vol. 56, no. 5, pp. 2307–2359, 2010.
4. M. Mahoney, PAQ / LPAQ compressor documentation and source
   (mattmahoney.net/dc), and *Data Compression Explained*.
5. E. Arıkan, "From sequential decoding to channel polarization and back again," Shannon
   Lecture, IEEE ISIT 2019 (PAC codes); also arXiv:1908.09594 (2019) — *verify exact title*.
6. 3GPP TS 38.212, "NR; Multiplexing and channel coding" (CRC polynomials, Sec. 5.1).
7. P. Trifonov, "Efficient Design and Decoding of Polar Codes," *IEEE Trans. Commun.*,
   vol. 60, no. 11, pp. 3221–3227, 2012 (Gaussian approximation design).
8. S.-Y. Chung, T. J. Richardson and R. L. Urbanke, "Analysis of Sum-Product Decoding of
   Low-Density Parity-Check Codes Using a Gaussian Approximation," *IEEE Trans. Inf. Theory*,
   vol. 47, no. 2, pp. 657–670, 2001 (φ approximation).
9. A. Balatsoukas-Stimming, M. Bastani Parizi and A. Burg, "LLR-Based Successive
   Cancellation List Decoding of Polar Codes," *IEEE Trans. Signal Process.*, vol. 63,
   no. 19, pp. 5165–5179, 2015 (LLR-domain path metric) — *verify volume/pages*.
10. K. Niu and K. Chen, "CRC-Aided Decoding of Polar Codes," *IEEE Commun. Lett.*, vol. 16,
    no. 10, pp. 1668–1671, 2012 — *verify volume/pages*.
