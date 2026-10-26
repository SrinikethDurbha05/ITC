"""main.py - ITC live communication system (interactive CLI).

Pipeline:  input -> context-mixing source encoder -> CRC + polar encoder -> channel
           -> CA-SCL polar decoder -> context-mixing source decoder -> output

Run interactively:            python main.py
Run once without prompts:     python main.py --text "my name is sunny" --channel bsc --p 0.05 --seed 42
"""
import argparse
import math
import sys

import numpy as np

import cm_coder
import theory
from channel import Channel
from crc import CRC
from polar import PolarCode, encoder_matches_matrix
from utils import bytes_to_bits, entropy_h0, entropy_h1

LINE = "=" * 66
SUB = "-" * 66
CRC_CHOICES = {"6": "CRC-6", "11": "CRC-11", "24": "CRC-24C", "CRC-6": "CRC-6", "CRC-11": "CRC-11",
               "CRC-24C": "CRC-24C"}
DEFAULTS = dict(N=256, K=128, crc="CRC-11", L=8, warm=True, design_ebn0=2.5, lossless_p_design=0.05)


# =============================================================================== pipeline
def run_pipeline(data, mode="text", channel=None, seed=42, N=256, K=128, crc_name="CRC-11", L=8,
                 warm=True, design_ebn0=2.5, p_design=None, compare_sc=True, cold_size=True):
    """Run the full transmit/receive chain once and return every reported quantity in a dict.

    data: str (text mode) or sequence of 0/1 (binary mode).  The random generator seeded with
    `seed` drives only the channel noise, so a run is reproducible for a given seed.
    The receiver side uses only the channel output, the code parameters and the shared warm text.
    """
    channel = channel or Channel("lossless")
    rng = np.random.default_rng(seed)
    res = {"mode": mode, "seed": seed, "channel": channel}

    # ---------------------------------------------------------------- [1] source
    if mode == "text":
        raw = data.encode("utf-8")
        symbols = list(raw)
        src_bits = bytes_to_bits(raw)
        res.update(message=data, raw=raw, n_symbols=len(raw))
    else:
        raw = [int(b) for b in data]
        symbols = raw
        src_bits = np.array(raw, dtype=np.uint8)
        res.update(message="".join(map(str, raw)), raw=raw, n_symbols=len(raw))
    n_src = int(src_bits.size)
    ones = int(src_bits.sum())
    res.update(src_bits=src_bits, n_src=n_src, p1=ones / n_src if n_src else 0.0,
               p0=(n_src - ones) / n_src if n_src else 0.0, H0=entropy_h0(symbols), H1=entropy_h1(symbols))

    # ---------------------------------------------------------------- [2] source coding
    comp = np.array(cm_coder.compress(raw, mode, warm), dtype=np.uint8)
    res["comp_bits"] = comp
    res["n_comp"] = int(comp.size)
    res["n_comp_other"] = len(cm_coder.compress(raw, mode, not warm)) if cold_size else None
    res["warm"] = warm

    # ---------------------------------------------------------------- [3] channel coding
    crc = CRC(crc_name)
    if channel.kind == "awgn":
        code = PolarCode(N, K, crc, design="awgn", design_param=design_ebn0)
        res["design_text"] = f"Gaussian approximation (BI-AWGN, design Eb/N0 = {design_ebn0:g} dB)"
    else:
        pd = p_design if p_design is not None else (channel.p if channel.kind == "bsc"
                                                    else DEFAULTS["lossless_p_design"])
        code = PolarCode(N, K, crc, design="bsc", design_param=pd)
        res["design_text"] = f"Bhattacharyya (BSC, p_design = {pd:g})"
    kp = code.k_payload
    n_blocks = max(1, math.ceil(comp.size / kp))
    padding = n_blocks * kp - comp.size
    payload = np.concatenate([comp, np.zeros(padding, np.uint8)]).reshape(n_blocks, kp)
    x = code.encode(payload)
    res.update(code=code, n_blocks=n_blocks, padding=padding, n_tx=int(x.size),
               frozen_ok=code.validate_frozen_set(), encoder_ok=encoder_matches_matrix(N, trials=4))

    # ---------------------------------------------------------------- [4] channel
    if channel.kind == "awgn" and abs(channel.rate - code.R_eff) > 1e-12:
        raise ValueError("AWGN channel must be created with rate = R_eff of the code")
    rx = channel.transmit(x, rng)
    res.update(rx_hard=rx["hard"], chan_errors=rx["errors"], chan_ber=rx["errors"] / x.size,
               uncoded_expected=channel.uncoded_bit_error_prob() * comp.size)

    # ---------------------------------------------------------------- [5] receiver
    out = code.decode_scl(rx["llr"], L)
    rx_payload = out["payload"]
    blocks_ok = (rx_payload == payload).all(axis=1)
    res.update(crc_ok=out["crc_ok"], blocks_ok=blocks_ok)
    if compare_sc:
        sc = code.decode_sc(rx["llr"])
        sc_ok = (sc["payload"] == payload).all(axis=1)
        res["rescued"] = int((~sc_ok & blocks_ok).sum())
        res["sc_crc_ok"] = sc["crc_ok"]
    x_hat = out["x_hat"]
    res["corrected"] = int(((rx["hard"] != x) & (x_hat == x)).sum())
    res["residual_cw_errors"] = int((x_hat != x).sum())

    rec_bits = rx_payload.reshape(-1)
    fail = []
    if not out["crc_ok"].all():
        bad = [str(i) for i in np.flatnonzero(~out["crc_ok"])]
        fail.append(f"CRC FAILED in block(s) {', '.join(bad)}")
    recovered = None
    try:
        rmode, rwarm, recovered = cm_coder.decompress(rec_bits.tolist())
        if rmode != mode or rwarm != warm:
            fail.append("header corrupted (mode / warm-start flag mismatch)")
        if len(recovered) != len(raw):
            fail.append(f"length mismatch: {len(recovered)} symbols decoded, {len(raw)} sent")
    except ValueError as e:
        fail.append(f"source decoding failed: {e}")
        recovered = b"" if mode == "text" else []
    if mode == "text":
        rec_src = bytes_to_bits(bytes(recovered))
        res["recovered_text"] = bytes(recovered).decode("utf-8", errors="replace")
    else:
        rec_src = np.array(recovered, dtype=np.uint8)
        res["recovered_text"] = "".join(map(str, recovered))
    m = min(rec_src.size, n_src)
    res["src_errors"] = int((rec_src[:m] != src_bits[:m]).sum()) + abs(rec_src.size - n_src)
    res["src_ber"] = res["src_errors"] / n_src if n_src else 0.0
    res["match"] = (bytes(recovered) == raw) if mode == "text" else (list(recovered) == raw)
    if not res["match"] and not fail:
        fail.append("decoded message differs from the input (undetected error)")
    res["fail_reasons"] = fail if not res["match"] else [f for f in fail if f.startswith("CRC")]

    # ---------------------------------------------------------------- [6] theory
    res["capacity"] = channel.capacity()
    if channel.kind == "bsc":
        res["fer_pred"] = theory.bsc_normal_approx_fer(channel.p, N, code.R_eff)
    elif channel.kind == "awgn":
        res["fer_pred"] = theory.biawgn_normal_approx_fer(channel.ebn0_db, N, code.R_eff)
    else:
        res["fer_pred"] = None
    return res


# =============================================================================== report
def _kv(label, value):
    """One aligned 'label : value' report line."""
    print(f"{label:<23}: {value}")


def print_report(res):
    """Print the six report sections in the required format."""
    code, ch = res["code"], res["channel"]
    text = res["mode"] == "text"
    print("\n[1] SOURCE")
    _kv("Input mode", "TEXT" if text else "BINARY")
    _kv("Original message" if text else "Original sequence", res["message"])
    if text:
        _kv("UTF-8 message size", f"{res['n_symbols']} bytes")
    else:
        _kv("Sequence length", f"{res['n_symbols']} bits")
    _kv("Source bits", res["n_src"])
    _kv("Source P(0)", f"{res['p0']:.6f}")
    _kv("Source P(1)", f"{res['p1']:.6f}")
    unit = "bits/symbol (symbol = byte)" if text else "bits/symbol (symbol = bit)"
    _kv("Order-0 entropy H0", f"{res['H0']:.6f} {unit}")
    _kv("Order-1 entropy H1", f"{res['H1']:.6f} {unit}")

    print("\n[2] CONTEXT-MIXING SOURCE CODING")
    _kv("Model", "bitwise context mixing (orders 0-3 + match model)" if text
        else "bitwise context mixing (bit-history orders 0-8)")
    _kv("Warm start", "ON" if res["warm"] else "OFF")
    _kv("Original source bits", res["n_src"])
    _kv("Compressed bits", f"{res['n_comp']}  (includes 32-bit length header)")
    ratio = res["n_src"] / res["n_comp"]
    _kv("Compression ratio", f"{ratio:.3f} : 1")
    _kv("Size reduction", f"{100.0 * (1.0 - res['n_comp'] / res['n_src']) if res['n_src'] else 0.0:.2f}%")
    _kv("Average code length", f"{res['n_comp'] / max(res['n_symbols'], 1):.4f} bits/symbol")
    if res["n_comp_other"] is not None:
        _kv("Without warm start" if res["warm"] else "With warm start", f"{res['n_comp_other']} bits")
    if res["n_comp"] >= res["n_src"]:
        print("NOTE: for this short input the model cannot beat the raw size: the 32-bit header and the\n"
              "      cost of learning the statistics outweigh the savings.")

    print("\n[3] POLAR CHANNEL ENCODING")
    _kv("Block length N", code.N)
    _kv("Info bits/block (K)", code.K)
    _kv("CRC", f"{code.crc}, payload bits/block = {code.k_payload}")
    _kv("Codeword bits/block", code.N)
    _kv("Effective code rate", f"{code.R_eff:.3f}  (= (K - r)/N payload bits per channel use)")
    _kv("Frozen-set design", res["design_text"])
    _kv("Compressed bits", res["n_comp"])
    _kv("Padding bits", res["padding"])
    _kv("Polar blocks", res["n_blocks"])
    _kv("Transmitted bits", res["n_tx"])
    _kv("Frozen-set validation", "PASSED" if res["frozen_ok"] else "FAILED")
    _kv("Encoder vs G-matrix", "PASSED" if res["encoder_ok"] else "FAILED")

    print("\n[4] CHANNEL TRANSMISSION")
    _kv("Channel type", ch.describe())
    _kv("Channel bit errors", res["chan_errors"])
    _kv("Channel BER", f"{res['chan_ber']:.6f}")
    print(f"Uncoded-equivalent errors expected : {res['uncoded_expected']:.2f}  "
          f"(if the {res['n_comp']} compressed bits were sent without coding)")

    print("\n[5] RECEIVER AND SOURCE DECODING")
    L = res.get("L", 8)
    _kv("Decoder", f"CRC-aided Successive Cancellation List (L={L})")
    _kv("CRC checks passed", f"{int(res['crc_ok'].sum())} / {res['n_blocks']} blocks")
    if "rescued" in res:
        _kv("Blocks rescued by list", f"{res['rescued']}  (blocks plain SC would have failed)")
    _kv("Recovered message" if text else "Recovered sequence", res["recovered_text"])
    _kv("Message matches input", res["match"])

    print("\n[6] FINAL RESULTS")
    print(SUB)
    _kv("Source bits", res["n_src"])
    _kv("Compressed bits", res["n_comp"])
    _kv("Transmitted bits", res["n_tx"])
    _kv("Channel bit errors", res["chan_errors"])
    print(f"Errors corrected by decoder : {res['corrected']}")
    _kv("Source bit errors", res["src_errors"])
    _kv("Source BER", f"{res['src_ber']:.10f}")
    C, R = res["capacity"], code.R_eff
    rel = "<" if R < C else ">="
    if ch.kind == "bsc":
        _kv("Capacity check", f"R_eff = {R:.3f} {rel} C = 1 - h2(p) = {C:.4f}")
    elif ch.kind == "awgn":
        _kv("Capacity check", f"R_eff = {R:.3f} {rel} C_BIAWGN(sigma^2 = {ch.sigma2:.4f}) = {C:.4f}")
    else:
        _kv("Capacity check", f"R_eff = {R:.3f} < C = 1 (noiseless channel)")
    if res["fer_pred"] is None:
        _kv("Finite-blocklength", f"predicted FER of ideal code (N={code.N}) = 0 (noiseless channel)")
    else:
        _kv("Finite-blocklength", f"predicted FER of ideal code (N={code.N}) = {res['fer_pred']:.3e}"
            "  (normal approximation)")
    print()
    if res["match"]:
        print("RESULT: PASS")
        print("The original input was recovered exactly.")
        for f in res["fail_reasons"]:
            print(f"(note: {f}, but the message was still recovered)")
    else:
        print("RESULT: FAIL")
        for f in res["fail_reasons"]:
            print(f"Reason: {f}")
    print(LINE)


# =============================================================================== input helpers
def ask(prompt, parse, default=None):
    """Prompt until parse(answer) succeeds; empty answer -> default (if given)."""
    while True:
        try:
            ans = input(prompt).strip()
        except EOFError:
            print("\nInput closed - exiting.")
            sys.exit(0)
        if ans == "" and default is not None:
            return default
        try:
            return parse(ans)
        except ValueError as e:
            print(f"  Invalid input: {e}")


def _choice(options):
    """Parser accepting one of the given strings."""
    def parse(s):
        """Validate against the allowed options."""
        if s not in options:
            raise ValueError(f"enter one of {', '.join(options)}")
        return s
    return parse


def _binary(s):
    """Parser for a non-empty 0/1 string (spaces ignored)."""
    s = s.replace(" ", "")
    if not s or set(s) - {"0", "1"}:
        raise ValueError("binary sequence must be non-empty and contain only 0 and 1")
    return [int(c) for c in s]


def _text(s):
    """Parser for a non-empty message."""
    if s == "":
        raise ValueError("message must not be empty")
    return s


def _float_range(lo, hi, lo_open=True, hi_open=True):
    """Parser for a float inside an (open/closed) interval."""
    def parse(s):
        """Validate a float against the interval."""
        v = float(s)
        if not math.isfinite(v) or (v <= lo if lo_open else v < lo) or (v >= hi if hi_open else v > hi):
            raise ValueError(f"value must lie in {'(' if lo_open else '['}{lo}, {hi}{')' if hi_open else ']'}")
        return v
    return parse


def _int_choice(values):
    """Parser for an integer from a fixed set."""
    def parse(s):
        """Validate an integer against the allowed set."""
        v = int(s)
        if v not in values:
            raise ValueError(f"choose one of {values}")
        return v
    return parse


def _seed(s):
    """Parser for a non-negative integer seed."""
    v = int(s)
    if v < 0:
        raise ValueError("seed must be >= 0")
    return v


def interactive_once():
    """Ask for one input and all parameters, run the pipeline and print the report."""
    print("\nChoose your live input:")
    print("1. Text message")
    print("2. Binary sequence")
    choice = ask("Enter choice (1 or 2): ", _choice(["1", "2"]))
    if choice == "1":
        mode, data = "text", ask("\nType your message: ", _text)
    else:
        mode, data = "binary", ask("\nType your binary sequence (0/1 only): ", _binary)

    print("\nChoose the channel:")
    print("1. Lossless")
    print("2. BSC (binary symmetric channel, hard-decision wired link)")
    print("3. BPSK over AWGN (thermal noise on copper/coax)")
    ch = ask("Enter choice (1, 2 or 3) [2]: ", _choice(["1", "2", "3"]), "2")
    p = ebn0 = None
    design_ebn0 = DEFAULTS["design_ebn0"]
    if ch == "2":
        p = ask("Crossover probability p, 0 < p < 0.5 [0.05]: ", _float_range(0.0, 0.5), 0.05)
    elif ch == "3":
        ebn0 = ask("Eb/N0 in dB [2.5]: ", _float_range(-20.0, 30.0, False, False), 2.5)
        design_ebn0 = ask("Design Eb/N0 for the frozen set in dB [2.5]: ",
                          _float_range(-5.0, 15.0, False, False), 2.5)
    seed = ask("Random seed [42]: ", _seed, 42)

    N, K, crc_name, L, warm = (DEFAULTS[k] for k in ("N", "K", "crc", "L", "warm"))
    if ask("Use default code (N=256, K=128, CRC-11, L=8, warm start ON)? (y/n) [y]: ",
           _choice(["y", "n", "Y", "N"]), "y").lower() == "n":
        N = ask("Block length N (64/128/256/512) [256]: ", _int_choice([64, 128, 256, 512]), 256)
        crc_name = CRC_CHOICES[ask("CRC (6, 11 or 24) [11]: ", _choice(["6", "11", "24"]), "11")]
        r = CRC(crc_name).r

        def parse_k(s):
            """Validate r < K <= N."""
            v = int(s)
            if not r < v <= N:
                raise ValueError(f"K must satisfy {r} < K <= {N}")
            return v
        K = ask(f"Info bits per block K ({r} < K <= {N}) [{N // 2}]: ", parse_k, N // 2)
        L = ask("List size L (1/2/4/8/16/32) [8]: ", _int_choice([1, 2, 4, 8, 16, 32]), 8)
        warm = ask("Warm start (y/n) [y]: ", _choice(["y", "n", "Y", "N"]), "y").lower() == "y"
    run_and_report(data, mode, ch, p, ebn0, design_ebn0, seed, N, K, crc_name, L, warm)


def run_and_report(data, mode, ch, p, ebn0, design_ebn0, seed, N, K, crc_name, L, warm):
    """Build the channel, run the pipeline and print the report."""
    r_eff = (K - CRC(crc_name).r) / N
    channel = (Channel("lossless") if ch == "1" else Channel("bsc", p=p) if ch == "2"
               else Channel("awgn", ebn0_db=ebn0, rate=r_eff))
    res = run_pipeline(data, mode, channel, seed=seed, N=N, K=K, crc_name=crc_name, L=L, warm=warm,
                       design_ebn0=design_ebn0)
    res["L"] = L
    print_report(res)
    return res


def banner():
    """Print the title box."""
    print(LINE)
    print("     ITC LIVE COMMUNICATION SYSTEM")
    print("     CONTEXT-MIXING SOURCE CODING + CRC-AIDED POLAR (SCL) + NOISY CHANNEL")
    print(LINE)


def main(argv=None):
    """Entry point: interactive loop, or a single scripted run when --text/--binary is given."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="run once on this text message (no prompts)")
    ap.add_argument("--binary", help="run once on this 0/1 sequence (no prompts)")
    ap.add_argument("--channel", choices=["lossless", "bsc", "awgn"], default="bsc")
    ap.add_argument("--p", type=float, default=0.05, help="BSC crossover probability")
    ap.add_argument("--ebn0", type=float, default=2.5, help="AWGN Eb/N0 in dB")
    ap.add_argument("--design-ebn0", type=float, default=2.5, help="GA design Eb/N0 in dB")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--N", type=int, choices=[64, 128, 256, 512], default=256)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--crc", choices=["6", "11", "24"], default="11")
    ap.add_argument("--L", type=int, choices=[1, 2, 4, 8, 16, 32], default=8)
    ap.add_argument("--no-warm", action="store_true", help="disable the warm start")
    args = ap.parse_args(argv)

    banner()
    if args.text is not None or args.binary is not None:
        crc_name = CRC_CHOICES[args.crc]
        if not CRC(crc_name).r < args.K <= args.N:
            ap.error("need r < K <= N")
        if args.text is not None:
            mode, data = "text", _text(args.text)
        else:
            mode, data = "binary", _binary(args.binary)
        ch = {"lossless": "1", "bsc": "2", "awgn": "3"}[args.channel]
        if ch == "2" and not 0 < args.p < 0.5:
            ap.error("p must lie in (0, 0.5)")
        print(f"\nInput ({mode}): {data if mode == 'text' else ''.join(map(str, data))}")
        print(f"Channel: {args.channel}, seed = {args.seed}")
        res = run_and_report(data, mode, ch, args.p, args.ebn0, args.design_ebn0, args.seed, args.N, args.K,
                             crc_name, args.L, not args.no_warm)
        return 0 if res["match"] else 1
    while True:
        interactive_once()
        again = ask("\nRun again? (y/n): ", _choice(["y", "n", "Y", "N"]))
        if again.lower() != "y":
            print("Goodbye.")
            return 0
        banner()


if __name__ == "__main__":
    sys.exit(main())
