"""plot_results.py - Monte-Carlo performance figures, results.csv and the summary table.

Usage:
    python plot_results.py            # full run (up to 50 000 frames per point)
    python plot_results.py --quick    # fast low-frame-count run (up to 2 000 frames per point)
Options: --seed S (default 42), --workers W, --outdir DIR, --no-show

Every number in the figures, the CSV and the printed tables comes from the simulations
and computations in this file.  Each simulation point stops after TARGET_ERRORS frame errors
or after the maximum number of frames, whichever comes first; the frames and errors actually
used are reported.  Each point uses its own generator seeded with (seed, point id), so the
results are reproducible for a given seed regardless of how the work is scheduled.
"""
import argparse
import csv
import json
import lzma
import math
import multiprocessing as mp
import os
import time
import zlib

import numpy as np

import cm_coder
import polar
import theory
import utils
from channel import Channel
from crc import CRC

N, K, CRC_NAME = 256, 128, "CRC-11"
R_CRC = CRC(CRC_NAME).r
KP = K - R_CRC                        # 117 payload bits per block
R_EFF = KP / N
DESIGN_EBN0 = 2.5                     # GA design point (dB) used for every AWGN point
L_FIXED_EBN0 = 2.0                    # Eb/N0 of the list-size experiment (Fig. 4)
L_VALUES = [1, 2, 4, 8, 16, 32]
TARGET_ERRORS = 100

# fixed categorical slots (reference palette, light mode) - colour follows the scheme, never its rank
C_UNCODED, C_SC, C_CASCL, C_SCLNOCRC, C_CRC6, C_PAC = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                       "#6250d6")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3de", "#fcfcfb"

SCHEMES = {
    # name: (K_info, crc, decoder, L, colour, marker, linestyle)
    "Uncoded BPSK": (None, None, "uncoded", 0, C_UNCODED, "o", "-"),
    "Polar SC (L=1, no CRC)": (KP, None, "sc", 1, C_SC, "s", "-"),
    "Polar CA-SCL (L=8, CRC-11)": (K, CRC_NAME, "scl", 8, C_CASCL, "D", "-"),
    "Polar SCL (L=8, no CRC)": (KP, None, "scl", 8, C_SCLNOCRC, "^", "--"),
    "Polar CA-SCL (L=8, CRC-6)": (KP + 6, "CRC-6", "scl", 8, C_CRC6, "P", "-."),
}
# OPTIONAL TIER 2: PAC (no CRC, K = KP payload bits, same R_eff) vs CA-polar at the same L
PAC_SCHEMES = {
    "PAC (L=8, RM profile)": (KP, None, "pac", 8, C_PAC, "X", "-"),
    "PAC (L=32, RM profile)": (KP, None, "pac", 32, C_PAC, "x", "--"),
    "Polar CA-SCL (L=32, CRC-11)": (K, CRC_NAME, "scl", 32, C_CASCL, "d", "--"),
}
ALL_SCHEMES = {**SCHEMES, **PAC_SCHEMES}


# =============================================================================== simulation (workers)
_CODES = {}


def get_code(k, crc_name, design, param):
    """Per-process cache of designed polar codes."""
    key = (k, crc_name, design, param)
    if key not in _CODES:
        if crc_name == "PAC":
            _CODES[key] = polar.PACCode(N, k, design=design, design_param=param)
        else:
            _CODES[key] = polar.PolarCode(N, k, CRC(crc_name) if crc_name else None, design=design,
                                          design_param=param)
    return _CODES[key]


def run_point(task):
    """Simulate one (scheme, channel parameter) point until TARGET_ERRORS frame errors or max_frames.

    Frames carry KP random payload bits.  Coded schemes: encode -> channel -> decode; the uncoded
    reference sends the KP bits directly with BPSK at R = 1 (or over the same BSC).
    Returns counts plus the per-frame sum of squared bit errors (for a frame-level BER confidence interval).
    """
    rng = np.random.default_rng([task["seed"], task["id"]])
    k, crc_name, decoder, L = task["k"], task["crc"], task["decoder"], task["L"]
    if decoder == "uncoded":
        code, kp, rate = None, KP, 1.0
    else:
        design = "awgn" if task["channel"] == "awgn" else "bsc"
        code = get_code(k, "PAC" if decoder == "pac" else crc_name, design,
                        DESIGN_EBN0 if design == "awgn" else task["param"])
        kp, rate = code.k_payload, code.R_eff
    ch = (Channel("awgn", ebn0_db=task["param"], rate=rate) if task["channel"] == "awgn"
          else Channel("bsc", p=task["param"]))
    frames = fe = be = inlist = 0
    be_sq = 0.0
    t_dec = 0.0
    batch = 100
    while frames < task["max_frames"] and fe < task["target"]:
        nb = min(batch, task["max_frames"] - frames)
        pay = rng.integers(0, 2, (nb, kp), dtype=np.uint8)
        x = pay if code is None else code.encode(pay)
        rx = ch.transmit(x, rng)
        t0 = time.perf_counter()
        if code is None:
            est = rx["hard"]
        elif decoder == "sc":
            est = code.decode_sc(rx["llr"])["payload"]
        else:
            out = code.decode_scl(rx["llr"], L)
            est = out["payload"]
        t_dec += time.perf_counter() - t0
        e = (est != pay).sum(axis=1)
        if decoder == "scl":
            # diagnostic: frame errors although the transmitted info vector survived in the final list
            truth = polar.polar_transform(x)[:, code.info]
            in_list = (out["list_info_bits"] == truth[:, None, :]).all(axis=2).any(axis=1)
            inlist += int(((e > 0) & in_list).sum())
        be += int(e.sum())
        be_sq += float((e.astype(float) ** 2).sum())
        fe += int((e > 0).sum())
        frames += nb
        batch = 500
    out = dict(task)
    out.update(frames=frames, frame_errors=fe, inlist_errors=inlist if decoder == "scl" else None,
               bit_errors=be, bit_err_sq=be_sq, kp=kp, rate=rate,
               BER=be / (frames * kp), FER=fe / frames, time_per_block=t_dec / frames)
    return out


# =============================================================================== statistics
def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def ber_interval(r, z=1.96):
    """95% CLT interval for the BER treating frames (not bits) as i.i.d.: bit errors cluster inside frames."""
    n, kp = r["frames"], r["kp"]
    mean = r["BER"]
    var = max(r["bit_err_sq"] / (kp * kp) / n - mean * mean, 0.0)
    h = z * math.sqrt(var / n)
    lo, hi = max(mean - h, 0.0), mean + h
    if 0 < r["frame_errors"] < 20:            # few error frames: CLT is meaningless, scale the Wilson FER interval
        a, b = wilson(r["frame_errors"], n)
        lo, hi = mean * a / r["FER"], mean * b / r["FER"]
    return lo, hi


def ebn0_at_ber(rows, target=1e-3, metric="BER"):
    """Eb/N0 where the simulated BER (or FER) curve crosses `target` (log-linear interpolation), or None."""
    pts = sorted((r["param"], r[metric]) for r in rows if r["bit_errors"] > 0)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 >= target > y1:
            t = (math.log10(target) - math.log10(y0)) / (math.log10(y1) - math.log10(y0))
            return x0 + t * (x1 - x0)
    return None


# =============================================================================== plotting helpers
def _style(plt):
    """Recessive axes/grid, thin marks, light surface."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "lines.markersize": 6.5, "legend.frameon": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.fontsize": 8.5,
    })


def _plot_series(ax, rows, metric, label, color, marker, ls):
    """Plot one simulated curve with 95% error bars; points with zero errors are drawn as
    hollow down-triangles at the 95% upper bound 3/frames ('rule of three')."""
    rows = sorted(rows, key=lambda r: r["param"])
    xs, ys, lo, hi = [], [], [], []
    floor_x, floor_y = [], []
    for r in rows:
        errs = r["frame_errors"] if metric == "FER" else r["bit_errors"]
        if errs == 0:
            floor_x.append(r["param"])
            floor_y.append(3.0 / (r["frames"] * (1 if metric == "FER" else r["kp"])))
            continue
        v = r[metric]
        a, b = wilson(r["frame_errors"], r["frames"]) if metric == "FER" else ber_interval(r)
        xs.append(r["param"])
        ys.append(v)
        lo.append(max(v - max(a, v * 1e-3), 0.0))
        hi.append(max(b - v, 0.0))
    if xs:
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, ls=ls, label=label, capsize=2.5,
                    elinewidth=1.0, markeredgecolor=SURFACE, markeredgewidth=1.0)
    if floor_x:
        ax.plot(floor_x, floor_y, ls="none", marker="v", mfc="none", mec=color, mew=1.5, ms=8)


def _floor_legend(ax):
    """Legend entry explaining the zero-error floor marker."""
    ax.plot([], [], ls="none", marker="v", mfc="none", mec=INK2, mew=1.5, ms=8,
            label="0 errors observed: 95% upper bound (3/frames)")


def _vline(ax, x, text, y_frac=0.30):
    """Labelled vertical reference line in muted ink."""
    ax.axvline(x, color=INK2, ls=":", lw=1.5)
    ax.text(x, y_frac, " " + text, transform=ax.get_xaxis_transform(), color=INK2, fontsize=8.5,
            rotation=90, va="bottom", ha="right")


# =============================================================================== figures
def fig_awgn(plt, rows, metric, path, shannon_db, ebn0_grid):
    """Fig. 1 (BER) / Fig. 2 (FER) versus Eb/N0 over BI-AWGN."""
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    dense = np.linspace(min(ebn0_grid), max(ebn0_grid), 101)
    pb = theory.uncoded_bpsk_ber(dense)
    if metric == "BER":
        ax.plot(dense, pb, color=C_UNCODED, ls=":", lw=1.5, label="Uncoded BPSK theory Q(sqrt(2Eb/N0))")
    else:
        ax.plot(dense, 1 - (1 - pb) ** KP, color=C_UNCODED, ls=":", lw=1.5,
                label=f"Uncoded theory 1-(1-Pb)^{KP} ({KP}-bit frame)")
        na = [theory.biawgn_normal_approx_fer(e, N, R_EFF) for e in dense]
        ax.plot(dense, na, color=INK2, ls="--", lw=1.5,
                label=f"Normal approx., ideal ({N},{KP}) code")
    for name, (_, _, _, _, col, mk, ls) in SCHEMES.items():
        sr = [r for r in rows if r["scheme"] == name]
        _plot_series(ax, sr, metric, name + (" (sim.)" if name == "Uncoded BPSK" else ""), col, mk, ls)
    ax.set_yscale("log")
    ax.set_xlabel("Eb/N0 (dB)   [Eb = energy per payload bit; coded R_eff = (K-r)/N = %.3f]" % R_EFF)
    ax.set_ylabel(metric)
    ax.set_ylim(bottom=max(1e-7, ax.get_ylim()[0]), top=1.5)
    _vline(ax, shannon_db, f"BI-AWGN limit for R={R_EFF:.3f}: {shannon_db:.3f} dB")
    _floor_legend(ax)
    num = 1 if metric == "BER" else 2
    ax.set_title(f"Fig. {num}: {metric} vs Eb/N0, BI-AWGN, N={N}, R_eff={R_EFF:.3f} (GA design @ {DESIGN_EBN0} dB)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def fig_bsc(plt, rows, path, p_grid):
    """Fig. 3: FER vs crossover probability p over the BSC."""
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    dense = np.linspace(min(p_grid), max(p_grid), 120)
    ax.plot(dense, [theory.bsc_normal_approx_fer(p, N, R_EFF) for p in dense], color=INK2, ls="--", lw=1.5,
            label=f"Normal approx., ideal ({N},{KP}) code")
    for name in ("Polar SC (L=1, no CRC)", "Polar CA-SCL (L=8, CRC-11)"):
        _, _, _, _, col, mk, ls = SCHEMES[name]
        _plot_series(ax, [r for r in rows if r["scheme"] == name], "FER", name, col, mk, ls)
    pstar = theory.bsc_threshold(R_EFF)
    ax.set_yscale("log")
    ax.set_xlim(0, max(max(p_grid), pstar) * 1.05)
    _vline(ax, pstar, f"capacity threshold 1-h2(p)={R_EFF:.3f}: p={pstar:.4f}")
    _floor_legend(ax)
    ax.set_xlabel("BSC crossover probability p   (frozen set designed at the channel p)")
    ax.set_ylabel("FER")
    ax.set_ylim(bottom=max(1e-7, ax.get_ylim()[0]), top=1.5)
    ax.set_title(f"Fig. 3: FER vs p, BSC, N={N}, R_eff={R_EFF:.3f}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def fig_list(plt, rows, timing, path):
    """Fig. 4: FER and decoding time versus list size L (two panels, one y-axis each)."""
    rows = sorted(rows, key=lambda r: r["L"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    Ls = [r["L"] for r in rows]
    fer = [r["FER"] for r in rows]
    ci = [wilson(r["frame_errors"], r["frames"]) for r in rows]
    nz = [i for i, r in enumerate(rows) if r["frame_errors"] > 0]
    a1.errorbar([Ls[i] for i in nz], [fer[i] for i in nz],
                yerr=[[fer[i] - max(ci[i][0], fer[i] * 1e-3) for i in nz], [ci[i][1] - fer[i] for i in nz]],
                color=C_CASCL, marker="D", capsize=2.5, elinewidth=1.0, markeredgecolor=SURFACE,
                label="CA-SCL (CRC-11), simulated")
    z = [i for i, r in enumerate(rows) if r["frame_errors"] == 0]
    if z:
        a1.plot([Ls[i] for i in z], [3.0 / rows[i]["frames"] for i in z], ls="none", marker="v", mfc="none",
                mec=C_CASCL, mew=1.5, ms=8, label="0 errors: 95% upper bound")
    na = theory.biawgn_normal_approx_fer(L_FIXED_EBN0, N, R_EFF)
    a1.axhline(na, color=INK2, ls="--", lw=1.5, label=f"Normal approx. ideal code: {na:.2e}")
    for i in nz:
        a1.annotate(f"{rows[i]['frame_errors']}/{rows[i]['frames']}", (Ls[i], fer[i]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5, color=INK2)
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.set_xticks(Ls, [str(v) for v in Ls])
    a1.set_xlabel("List size L")
    a1.set_ylabel("FER")
    a1.set_title(f"Fig. 4a: FER vs L at Eb/N0 = {L_FIXED_EBN0} dB")
    a1.legend(loc="upper right")
    a2.plot(L_VALUES, [timing[L]["batched"] * 1e3 for L in L_VALUES], color=C_CASCL, marker="D",
            label="batched (200 blocks per call)")
    a2.plot(L_VALUES, [timing[L]["single"] * 1e3 for L in L_VALUES], color=C_UNCODED, marker="o", ls="--",
            label="one block per call (live latency)")
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xticks(L_VALUES, [str(v) for v in L_VALUES])
    a2.set_xlabel("List size L")
    a2.set_ylabel("Decoding time per block (ms)")
    a2.set_title("Fig. 4b: CA-SCL decoding time per block")
    a2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def source_coding_results(seed):
    """Fig. 5 data: bits/symbol of CM (warm/cold), H0, H1, raw deflate and raw LZMA2 on five inputs."""
    rng = np.random.default_rng([seed, 999])
    inputs = [(f"English {n} B", utils.english_sample(n), "text") for n in (16, 64, 256, 1024)]
    inputs.append(("Binary P(1)=0.1, 10000 b", utils.skewed_binary(10000, 0.1, rng), "binary"))
    out = []
    for name, data, mode in inputs:
        if mode == "text":
            n_sym, src_bits, packed, syms = len(data), 8 * len(data), data, list(data)
        else:
            n_sym, src_bits, syms = len(data), len(data), data
            packed = np.packbits(np.array(data, dtype=np.uint8)).tobytes()
        cm_w = len(cm_coder.compress(data, mode, True))
        cm_c = len(cm_coder.compress(data, mode, False))
        assert cm_coder.decompress(cm_coder.compress(data, mode, True))[2] == (data if mode == "text" else list(data))
        zl = 8 * len(zlib.compress(packed, 9, wbits=-15))
        lz = 8 * len(lzma.compress(packed, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9}]))
        out.append(dict(input=name, mode=mode, symbols=n_sym, source_bits=src_bits, cm_warm=cm_w, cm_cold=cm_c,
                        zlib=zl, lzma=lz, H0=utils.entropy_h0(syms), H1=utils.entropy_h1(syms)))
    return out


def _tick(name):
    """Two-line tick label for a Fig. 5 input name."""
    return name.replace("English ", "English\n").replace(", ", "\n")


def fig_source(plt, res, path):
    """Fig. 5: bits/symbol (text inputs; binary input) and compression ratio of every method."""
    methods = [("CM, warm start", "cm_warm", C_UNCODED), ("CM, no warm start", "cm_cold", C_SC),
               ("zlib raw deflate (ref.)", "zlib", C_CASCL), ("LZMA2 raw (ref.)", "lzma", C_SCLNOCRC)]
    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.0, 2.4])
    a1, a2, a3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for ax, subset, unit in ((a1, [r for r in res if r["mode"] == "text"], "bits/byte"),
                             (a2, [r for r in res if r["mode"] == "binary"], "bits/bit")):
        x = np.arange(len(subset))
        w = 0.8 / (len(methods) + 2)
        for j, (lab, key, col) in enumerate(methods):
            ax.bar(x + (j - 2.5) * w, [r[key] / r["symbols"] for r in subset], w * 0.92, color=col, label=lab)
        ax.bar(x + 1.5 * w, [r["H0"] for r in subset], w * 0.92,
               color="none", edgecolor=INK, hatch="////", lw=0.8, label="empirical H0")
        ax.bar(x + 2.5 * w, [r["H1"] for r in subset], w * 0.92, color="none", edgecolor=INK2, hatch="....",
               lw=0.8, label="empirical H1 (plug-in)")
        ax.set_xticks(x, [_tick(r["input"]) for r in subset], fontsize=8.5)
        ax.set_ylabel(f"{unit} (incl. headers)")
        if ax is a1:
            ax.axhline(8, color=INK2, ls=":", lw=1.2)
            ax.text(3.45, 8.15, "raw ASCII = 8", color=INK2, fontsize=8, ha="right")
        else:
            ax.axhline(1, color=INK2, ls=":", lw=1.2)
            ax.text(0.45, 1.02, "raw = 1", color=INK2, fontsize=8, ha="right")
    a1.set_title("Fig. 5a: English text")
    a2.set_title("Fig. 5b: skewed binary")
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8.5)
    x = np.arange(len(res))
    w = 0.8 / len(methods)
    for j, (lab, key, col) in enumerate(methods):
        a3.bar(x + (j - 1.5) * w, [r["source_bits"] / r[key] for r in res], w * 0.92, color=col, label=lab)
    a3.axhline(1, color=INK2, ls=":", lw=1.2)
    a3.set_xticks(x, [_tick(r["input"]) for r in res], fontsize=8)
    a3.set_ylabel("compression ratio  (original bits / compressed bits)")
    a3.set_title("Fig. 5c: compression ratio (>1 = smaller)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    return fig


def fig_polarization(plt, path, p=0.05):
    """Fig. 6: polarization of the N synthetic channels (Bhattacharyya for BSC(p), GA mean for BI-AWGN)."""
    code_b = polar.PolarCode(N, K, CRC(CRC_NAME), design="bsc", design_param=p)
    code_g = polar.PolarCode(N, K, CRC(CRC_NAME), design="awgn", design_param=DESIGN_EBN0)
    lz = polar.bhattacharyya_log_bsc(N, p) / math.log(10)
    lm = polar.ga_log_means(N, theory.ebn0_to_sigma2(DESIGN_EBN0, R_EFF)) / math.log(10)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.4))
    idx = np.arange(N)
    fr = code_b.frozen
    a1.scatter(idx[fr], lz[fr], s=10, color=C_SC, label=f"frozen ({N - K})")
    a1.scatter(idx[~fr], lz[~fr], s=10, color=C_UNCODED, marker="D", label=f"information ({K})")
    a1.set_xlabel("synthetic channel index i (natural order)")
    a1.set_ylabel("log10 Z(W_N^(i))")
    a1.set_ylim(bottom=max(lz.min(), -30) - 1)
    a1.set_title(f"Fig. 6a: Bhattacharyya Z, BSC(p={p}), by index")
    a1.legend(loc="lower left")
    order = np.argsort(-lz, kind="stable")             # least reliable first
    a2.plot(np.arange(N), np.clip(lz[order], -30, 0), color=C_UNCODED, lw=2)
    a2.axvline(N - K - 0.5, color=INK2, ls=":", lw=1.5)
    a2.text(N - K - 3, -29, "frozen | information", ha="center", color=INK2, fontsize=8.5,
            bbox=dict(fc=SURFACE, ec="none"))
    a2.set_xlabel("rank (least -> most reliable)")
    a2.set_ylabel("log10 Z (clipped at -30)")
    a2.set_title("Fig. 6b: sorted Bhattacharyya parameters")
    order_g = np.argsort(lm, kind="stable")
    a3.plot(np.arange(N), lm[order_g], color=C_CASCL, lw=2)
    a3.axvline(N - K - 0.5, color=INK2, ls=":", lw=1.5)
    a3.text(N - K - 3, lm.min() + 0.3, "frozen | information", ha="center", color=INK2, fontsize=8.5,
            bbox=dict(fc=SURFACE, ec="none"))
    a3.set_xlabel("rank (least -> most reliable)")
    a3.set_ylabel("log10 mean LLR (GA)")
    a3.set_title(f"Fig. 6c: sorted GA means, BI-AWGN @ {DESIGN_EBN0} dB")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def fig_pac(plt, rows, path):
    """Fig. 7 (OPTIONAL TIER 2): PAC (RM profile, conv. precoder 133 octal, SCL) vs CA-polar at the same R_eff."""
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    xs = sorted({r["param"] for r in rows})
    dense = np.linspace(min(xs), max(xs), 61)
    ax.plot(dense, [theory.biawgn_normal_approx_fer(e, N, R_EFF) for e in dense], color=INK2, ls="--", lw=1.5,
            label=f"Normal approx., ideal ({N},{KP}) code")
    names = ["Polar CA-SCL (L=8, CRC-11)", "Polar CA-SCL (L=32, CRC-11)", "PAC (L=8, RM profile)",
             "PAC (L=32, RM profile)"]
    for name in names:
        _, _, _, _, col, mk, ls = ALL_SCHEMES[name]
        _plot_series(ax, [r for r in rows if r["scheme"] == name], "FER", name, col, mk, ls)
    _floor_legend(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("FER")
    ax.set_ylim(bottom=max(1e-7, ax.get_ylim()[0]), top=1.5)
    ax.set_title(f"Fig. 7 (optional): PAC vs CRC-aided polar, BI-AWGN, N={N}, R_eff={R_EFF:.3f}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


# =============================================================================== main
def measure_timing(seed):
    """Decoding time per block vs L in the main process (no worker contention), at L_FIXED_EBN0."""
    rng = np.random.default_rng([seed, 777])
    code = get_code(K, CRC_NAME, "awgn", DESIGN_EBN0)
    ch = Channel("awgn", ebn0_db=L_FIXED_EBN0, rate=code.R_eff)
    pay = rng.integers(0, 2, (200, code.k_payload), dtype=np.uint8)
    llr = ch.transmit(code.encode(pay), rng)["llr"]
    out = {}
    for L in L_VALUES:
        code.decode_scl(llr[:2], L)                       # warm-up
        t0 = time.perf_counter()
        code.decode_scl(llr, L)
        batched = (time.perf_counter() - t0) / 200
        t0 = time.perf_counter()
        for i in range(20):
            code.decode_scl(llr[i:i + 1], L)
        single = (time.perf_counter() - t0) / 20
        out[L] = {"batched": batched, "single": single}
    t0 = time.perf_counter()
    code.decode_sc(llr)
    out["sc_batched"] = (time.perf_counter() - t0) / 200
    return out


def build_tasks(args, ebn0_grid, p_grid):
    """All simulation points with their stopping rules and unique ids."""
    tasks = []
    mf_coded = 2000 if args.quick else 50000

    def add(scheme, channel, param, L=None):
        """Append one simulation point with its own id (seed stream)."""
        k, crc_name, decoder, Ls, *_ = ALL_SCHEMES[scheme]
        tasks.append(dict(id=len(tasks), seed=args.seed, scheme=scheme, channel=channel, param=float(param),
                          k=k, crc=crc_name, decoder=decoder, L=L if L is not None else Ls,
                          max_frames=mf_coded, target=TARGET_ERRORS))
    for e in ebn0_grid:
        for s in SCHEMES:
            add(s, "awgn", e)
    for p in p_grid:
        add("Polar SC (L=1, no CRC)", "bsc", p)
        add("Polar CA-SCL (L=8, CRC-11)", "bsc", p)
    for e in ebn0_grid:
        if 1.0 <= e <= 4.0:
            for s in PAC_SCHEMES:
                add(s, "awgn", e)
                tasks[-1]["group"] = "pac"
    for L in L_VALUES:
        add("Polar CA-SCL (L=8, CRC-11)", "awgn", L_FIXED_EBN0, L=L)
        tasks[-1]["scheme"] = f"CA-SCL list sweep (L={L}, CRC-11)"
        tasks[-1]["group"] = "list"
    return tasks


def main():
    """Run all simulations in parallel, then draw the figures and write results.csv."""
    ap = argparse.ArgumentParser(description="Generate the performance figures and results.csv")
    ap.add_argument("--quick", action="store_true", help="fast low-frame-count run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    ap.add_argument("--replot", action="store_true",
                    help="redraw figures/CSV from OUTDIR/raw_results.json of an earlier run (same --quick setting)")
    args = ap.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    os.makedirs(args.outdir, exist_ok=True)

    ebn0_grid = np.arange(0.0, 5.01, 1.0 if args.quick else 0.5)
    p_grid = ([0.02, 0.04, 0.06, 0.08, 0.10, 0.12] if args.quick
              else [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12])
    raw_path = os.path.join(args.outdir, "raw_results.json")
    if args.replot:
        with open(raw_path) as f:
            rows = json.load(f)
        print(f"Re-plotting {len(rows)} simulation points from {raw_path} (no new simulation)")
    else:
        tasks = build_tasks(args, ebn0_grid, p_grid)
        tasks.sort(key=lambda t: -(t["L"] * (1 + t["param"] if t["channel"] == "awgn" else 1)))
        print(f"Running {len(tasks)} simulation points on {args.workers} worker processes "
              f"({'quick' if args.quick else 'full'} mode, seed {args.seed}, target {TARGET_ERRORS} frame errors, "
              f"max {tasks[0]['max_frames']} frames per point) ...")
        t0 = time.time()
        with mp.Pool(args.workers) as pool:
            rows = []
            for i, r in enumerate(pool.imap_unordered(run_point, tasks), 1):
                rows.append(r)
                print(f"  [{i:3d}/{len(tasks)}] {r['scheme']:<34} {r['channel']:>4} {r['param']:<5g} "
                      f"frames={r['frames']:<6d} FE={r['frame_errors']:<4d} BER={r['BER']:.3e} "
                      f"FER={r['FER']:.3e}", flush=True)
        rows.sort(key=lambda r: r["id"])
        print(f"Simulation time: {time.time() - t0:.1f} s")
        with open(raw_path, "w") as f:
            json.dump(rows, f, indent=1)

    timing = measure_timing(args.seed)
    shannon_db = theory.biawgn_min_ebn0_db(R_EFF)
    unc_db = theory.uncoded_ebn0_for_ber(1e-3)
    awgn = [r for r in rows if r["channel"] == "awgn" and r.get("group") not in ("list", "pac")]
    pac = [r for r in rows if r.get("group") == "pac"]
    bsc = [r for r in rows if r["channel"] == "bsc"]
    lst = [r for r in rows if r.get("group") == "list"]
    gains = {}
    for s in SCHEMES:
        x = ebn0_at_ber([r for r in awgn if r["scheme"] == s])
        gains[s] = (unc_db - x) if x is not None else None

    figs = [fig_awgn(plt, awgn, "BER", os.path.join(args.outdir, "fig1_ber_awgn.png"), shannon_db, ebn0_grid),
            fig_awgn(plt, awgn, "FER", os.path.join(args.outdir, "fig2_fer_awgn.png"), shannon_db, ebn0_grid),
            fig_bsc(plt, bsc, os.path.join(args.outdir, "fig3_fer_bsc.png"), p_grid),
            fig_list(plt, lst, timing, os.path.join(args.outdir, "fig4_list_size.png"))]
    src = source_coding_results(args.seed)
    figs.append(fig_source(plt, src, os.path.join(args.outdir, "fig5_source_coding.png")))
    figs.append(fig_polarization(plt, os.path.join(args.outdir, "fig6_polarization.png")))
    figs.append(fig_pac(plt, pac + [r for r in awgn if r["scheme"] == "Polar CA-SCL (L=8, CRC-11)"
                                    and 1.0 <= r["param"] <= 4.0],
                        os.path.join(args.outdir, "fig7_pac_optional.png")))

    # ---------------------------------------------------------------- results.csv + table
    hdr = ["scheme", "N", "K", "crc_bits", "R_eff", "L", "channel", "EbN0_dB_or_p", "frames", "frame_errors",
           "bit_errors", "BER", "FER", "time_per_block_ms", "gain_dB_at_BER_1e-3",
           "errors_with_correct_path_in_list"]
    table = []
    for r in rows:
        unc = r["decoder"] == "uncoded"
        g = gains.get(r["scheme"]) if r["channel"] == "awgn" and r.get("group") is None else None
        table.append([r["scheme"], "-" if unc else N, "-" if unc else r["k"],
                      CRC(r["crc"]).r if r["crc"] else 0,
                      f"{r['rate']:.4f}", "-" if unc else r["L"], r["channel"].upper(), f"{r['param']:g}",
                      r["frames"], r["frame_errors"], r["bit_errors"], f"{r['BER']:.3e}", f"{r['FER']:.3e}",
                      f"{r['time_per_block'] * 1e3:.4f}", "n/a" if g is None else f"{g:.2f}",
                      "-" if r["inlist_errors"] is None else r["inlist_errors"]])
    with open(os.path.join(args.outdir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(table)
    widths = [max(len(str(x)) for x in col) for col in zip(hdr, *table)]
    print("\n" + "  ".join(h.ljust(wd) for h, wd in zip(hdr, widths)))
    for row in table:
        print("  ".join(str(x).ljust(wd) for x, wd in zip(row, widths)))
    print("(time_per_block_ms: measured inside the parallel workers on batches of up to 500 blocks)")

    with open(os.path.join(args.outdir, "source_coding.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input", "symbols", "source_bits", "cm_warm_bits", "cm_cold_bits", "zlib_raw_bits",
                    "lzma2_raw_bits", "H0", "H1", "cm_warm_bits_per_symbol", "cm_cold_bits_per_symbol"])
        for r in src:
            w.writerow([r["input"], r["symbols"], r["source_bits"], r["cm_warm"], r["cm_cold"], r["zlib"], r["lzma"],
                        f"{r['H0']:.4f}", f"{r['H1']:.4f}", f"{r['cm_warm'] / r['symbols']:.4f}",
                        f"{r['cm_cold'] / r['symbols']:.4f}"])

    print("\nSource coding (bits/symbol incl. headers; symbol = byte for text, bit for binary):")
    print(f"{'input':<26}{'CM warm':>9}{'CM cold':>9}{'deflate':>9}{'LZMA2':>9}{'H0':>8}{'H1':>8}")
    for r in src:
        s = r["symbols"]
        print(f"{r['input']:<26}{r['cm_warm'] / s:>9.3f}{r['cm_cold'] / s:>9.3f}{r['zlib'] / s:>9.3f}"
              f"{r['lzma'] / s:>9.3f}{r['H0']:>8.3f}{r['H1']:>8.3f}")

    print(f"\nDecoding time per block at Eb/N0 = {L_FIXED_EBN0} dB (main process):")
    print(f"  plain SC (batched): {timing['sc_batched'] * 1e3:.3f} ms")
    for L in L_VALUES:
        print(f"  CA-SCL L={L:<2d}: batched {timing[L]['batched'] * 1e3:8.3f} ms   "
              f"one block per call {timing[L]['single'] * 1e3:8.3f} ms")
    print(f"\nBI-AWGN Shannon limit for R_eff = {R_EFF:.4f}: {shannon_db:.3f} dB")
    print(f"Uncoded BPSK needs {unc_db:.3f} dB for BER = 1e-3 (theory); BSC threshold p* = "
          f"{theory.bsc_threshold(R_EFF):.4f}")
    for s, g in gains.items():
        print(f"  coding gain at BER 1e-3, {s:<28}: " + ("not reached in the simulated range" if g is None
                                                           else f"{g:.2f} dB"))
    lo, hi = 0.0, 6.0                                     # normal-approximation Eb/N0 at FER 1e-3 (bisection)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if theory.biawgn_normal_approx_fer(mid, N, R_EFF) > 1e-3 else (lo, mid)
    print(f"Eb/N0 at FER 1e-3: normal approximation (ideal ({N},{KP}) code) {lo:.2f} dB")
    for s in list(SCHEMES)[1:]:
        x = ebn0_at_ber([r for r in awgn if r["scheme"] == s], metric="FER")
        print(f"  {s:<28}: " + ("not reached" if x is None else f"{x:.2f} dB  (gap {x - lo:.2f} dB)"))
    for s in PAC_SCHEMES:
        x = ebn0_at_ber([r for r in pac if r["scheme"] == s], metric="FER")
        print(f"  {s:<28}: " + ("not reached" if x is None else f"{x:.2f} dB  (gap {x - lo:.2f} dB)"))
    print(f"\nFigures and CSV files written to {os.path.abspath(args.outdir)}")
    if not args.no_show:
        plt.show()
    for f in figs:
        plt.close(f)


if __name__ == "__main__":
    main()
