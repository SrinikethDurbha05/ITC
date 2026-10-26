"""Unit tests for the ITC live communication system.  Run with:  python -m unittest -v"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import channel  # noqa: E402
import cm_coder as cm  # noqa: E402
import crc as crcmod  # noqa: E402
import main  # noqa: E402
import polar  # noqa: E402
import theory  # noqa: E402
import utils  # noqa: E402


# ======================================================================== CRC
class TestCRC(unittest.TestCase):
    def test_remainder_zero_and_matrix(self):
        rng = np.random.default_rng(1)
        for name in crcmod.CRC_POLYS:
            c = crcmod.CRC(name)
            for k in (1, 7, 64, 117, 200):
                m = rng.integers(0, 2, k, dtype=np.uint8)
                w = np.concatenate([m, c.compute(m)])
                self.assertTrue(c.check(w), name)
                self.assertFalse(crcmod.poly_mod(w, c.poly, c.r).any())
                np.testing.assert_array_equal(c.compute_batch(m), c.compute(m))
            batch = rng.integers(0, 2, (5, 3, 40), dtype=np.uint8)
            words = np.concatenate([batch, c.compute_batch(batch)], axis=-1)
            self.assertTrue(c.check_batch(words).all())

    def test_known_value(self):
        # 1 followed by zeros: CRC of the single bit '1' is x^r mod g(x) = g(x) - x^r
        for name, (r, poly) in crcmod.CRC_POLYS.items():
            c = crcmod.CRC(name)
            expect = [(poly >> (r - 1 - i)) & 1 for i in range(r)]
            np.testing.assert_array_equal(c.compute([1]), expect)

    def test_single_bit_errors_detected(self):
        rng = np.random.default_rng(2)
        for name in crcmod.CRC_POLYS:
            c = crcmod.CRC(name)
            m = rng.integers(0, 2, 117, dtype=np.uint8)
            w = np.concatenate([m, c.compute(m)])
            for i in range(w.size):
                e = w.copy()
                e[i] ^= 1
                self.assertFalse(c.check(e), f"{name} missed single error at {i}")

    def test_bursts_detected(self):
        rng = np.random.default_rng(3)
        for name in crcmod.CRC_POLYS:
            c = crcmod.CRC(name)
            m = rng.integers(0, 2, 100, dtype=np.uint8)
            w = np.concatenate([m, c.compute(m)])
            for blen in range(1, c.r + 1):
                for start in range(0, w.size - blen + 1, 3):
                    e = np.zeros(w.size, np.uint8)
                    e[start:start + blen] = rng.integers(0, 2, blen)
                    e[start] = e[start + blen - 1] = 1
                    self.assertFalse(c.check(w ^ e), f"{name} missed burst {blen}@{start}")

    def test_self_test(self):
        for name in crcmod.CRC_POLYS:
            self.assertTrue(crcmod.self_test(name))


# ======================================================================== theory
class TestTheory(unittest.TestCase):
    def test_bsc_capacity(self):
        self.assertAlmostEqual(theory.bsc_capacity(0.11), 0.5, delta=1e-3)
        self.assertEqual(theory.bsc_capacity(0.5), 0.0)
        self.assertAlmostEqual(theory.bsc_threshold(0.5), 0.110028, delta=1e-4)

    def test_dispersion_and_normal_approx(self):
        p = 0.05
        self.assertAlmostEqual(theory.bsc_dispersion(p), p * (1 - p) * np.log2(19) ** 2)
        R, n = 117 / 256, 256
        fers = [theory.bsc_normal_approx_fer(p, n, R) for p in (0.12, 0.10, 0.08, 0.06, 0.04)]
        self.assertTrue(all(a > b for a, b in zip(fers, fers[1:])), fers)
        # inverse consistency: plugging the predicted eps back gives the rate
        eps = theory.bsc_normal_approx_fer(0.06, n, R)
        r_back = theory.normal_approx_rate(n, eps, theory.bsc_capacity(0.06), theory.bsc_dispersion(0.06))
        self.assertAlmostEqual(r_back, R, delta=1e-9)

    def test_Q_Qinv(self):
        self.assertAlmostEqual(theory.Q(0.0), 0.5)
        for e in (1e-1, 1e-3, 1e-6):
            self.assertAlmostEqual(theory.Q(theory.Qinv(e)) / e, 1.0, delta=1e-6)

    def test_biawgn_capacity(self):
        for s2 in (0.1, 0.5, 1.0, 3.0):
            self.assertAlmostEqual(theory.biawgn_capacity(s2), theory.biawgn_capacity_grid(s2), delta=1e-5)
        # BPSK Shannon limit at rate 1/2 is the textbook ~0.19 dB
        self.assertTrue(0.1 < theory.biawgn_min_ebn0_db(0.5) < 0.3)
        # capacity of BI-AWGN never exceeds the unconstrained Gaussian capacity
        for s2 in (0.2, 1.0):
            self.assertLess(theory.biawgn_capacity(s2), 0.5 * np.log2(1 + 1 / s2))

    def test_uncoded_bpsk_monte_carlo(self):
        rng = np.random.default_rng(4)
        for ebn0 in (0.0, 4.0):
            n = 2_000_000
            s = np.sqrt(theory.ebn0_to_sigma2(ebn0, 1.0))
            y = 1.0 + s * rng.standard_normal(n)       # all-zero bits -> +1
            ber_mc = np.mean(y < 0)
            ber_th = theory.uncoded_bpsk_ber(ebn0)
            self.assertAlmostEqual(ber_mc / ber_th, 1.0, delta=0.03)


# ======================================================================== context mixing
class TestContextMixing(unittest.TestCase):
    def roundtrip(self, data, mode, warm):
        bits = cm.compress(data, mode, warm)
        # the receiver starts from scratch: forget every cached model state first
        cm._WARM_CACHE.clear()
        rmode, rwarm, out = cm.decompress(list(bits))
        self.assertEqual((rmode, rwarm), (mode, warm))
        self.assertEqual(out, bytes(data) if mode == "text" else list(data))
        return bits

    def test_text_roundtrips(self):
        rng = np.random.default_rng(5)
        cases = [
            b"",
            b"a",
            bytes(rng.integers(0, 256, 300, dtype=np.uint8)),
            utils.EVAL_TEXT.encode(),
            "Hello \U0001F600 你好, naïve café — नमस्ते \U0001F680".encode("utf-8"),
            (utils.EVAL_TEXT * 3).encode()[:5200],
            bytes(200),            # long run of zeros (very skewed)
        ]
        for data in cases:
            for warm in (False, True):
                bits = self.roundtrip(data, "text", warm)
                self.assertEqual(cm.parse_header(bits)[2], len(data))

    def test_binary_roundtrips(self):
        rng = np.random.default_rng(6)
        cases = [[], [1], [0, 1, 1, 0], rng.integers(0, 2, 3000).tolist(),
                 utils.skewed_binary(5000, 0.1, rng), [0, 1] * 700]
        for data in cases:
            for warm in (False, True):
                self.roundtrip(data, "binary", warm)

    def test_header_and_padding(self):
        bits = cm.compress(b"padding test", "text", True)
        self.assertEqual(len(bits) >= 32, True)
        # zeros appended by the channel-coding stage must not change the result
        self.assertEqual(cm.decompress(bits + [0] * 100)[2], b"padding test")
        self.assertEqual(len(cm.compress(b"", "text", False)), cm.HEADER_BITS)

    def test_compresses_english(self):
        d = utils.english_sample(1024)
        self.assertLess(len(cm.compress(d, "text", False)), 0.6 * 8 * len(d))
        self.assertLess(len(cm.compress(d, "text", True)), len(cm.compress(d, "text", False)))

    def test_corrupted_header_rejected(self):
        bits = [0, 0] + [1] * 30 + [0] * 50
        with self.assertRaises(ValueError):
            cm.decompress(bits)

    def test_arithmetic_coder_extreme_probabilities(self):
        rng = np.random.default_rng(7)
        n = 4000
        probs = rng.choice([0, 1, 2, 4094, 4095, 5000, -3, 2048], size=n)   # out-of-range values get clamped
        bits = [int(rng.random() < min(max(p, 1), 4095) / 4096) for p in probs]
        bits[:50] = [0] * 25 + [1] * 25           # also code the 'impossible' outcome at p = 1/4096
        probs[:50] = 4095
        enc = cm.ArithmeticEncoder()
        for b, p in zip(bits, probs):
            enc.encode(b, int(p))
        code = enc.finish()
        dec = cm.ArithmeticDecoder(code)
        self.assertEqual([dec.decode(int(p)) for p in probs], bits)

    def test_arithmetic_coder_cost(self):
        # coding cost must approach the ideal -log2 P(x)
        rng = np.random.default_rng(8)
        p = 410                                    # P(1) ~ 0.1
        bits = (rng.random(20000) < p / 4096).astype(int)
        enc = cm.ArithmeticEncoder()
        for b in bits:
            enc.encode(int(b), p)
        ideal = -sum(np.log2(p / 4096) if b else np.log2(1 - p / 4096) for b in bits)
        self.assertLess(len(enc.finish()), ideal * 1.01 + 40)

    def test_squash_stretch(self):
        self.assertEqual(cm.squash(0), 2048)
        for p in range(1, 4095, 7):
            self.assertLessEqual(abs(cm.squash(cm.STRETCH[p]) - p), max(2, p // 20))
        self.assertEqual(sorted(cm.STRETCH), cm.STRETCH)


# ======================================================================== polar
class TestPolar(unittest.TestCase):
    def test_fast_encoder_equals_matrix(self):
        for N in (2, 4, 8, 16, 32):
            self.assertTrue(polar.encoder_matches_matrix(N, trials=50))
        # involution: G_N^2 = I over GF(2)
        u = np.random.default_rng(9).integers(0, 2, (10, 64), dtype=np.uint8)
        np.testing.assert_array_equal(polar.polar_transform(polar.polar_transform(u)), u)

    def test_bec_genie_matches_bhattacharyya(self):
        # For the BEC the recursion is exact: the genie-aided SC erasure rate of u_i equals Z_i.
        N, eps, T = 16, 0.4, 200_000
        code = polar.PolarCode(N, 1, design="bec", design_param=eps)
        rng = np.random.default_rng(10)
        llr = np.where(rng.random((T, N)) < eps, 0.0, 30.0)
        emp = (code._rate0_leaf_llrs(llr) == 0).mean(axis=0)
        Z = np.exp(polar.bhattacharyya_log_bec(N, eps))
        self.assertLess(np.abs(emp - Z).max(), 0.006)

    def test_phi_chung_vs_numeric(self):
        for x in (0.3, 1.0, 3.0, 8.0, 15.0, 30.0):
            approx = float(np.exp(polar.phi_chung(np.array([x])))[0])
            self.assertAlmostEqual(approx / polar.phi_numeric(x), 1.0, delta=0.05)

    def test_frozen_set_design(self):
        c11 = crcmod.CRC("CRC-11")
        for N in (64, 128, 256, 512):
            for design, par in (("bsc", 0.05), ("bsc", 0.01), ("awgn", 2.5), ("awgn", 0.0)):
                code = polar.PolarCode(N, N // 2, c11, design=design, design_param=par)
                self.assertTrue(code.validate_frozen_set())
                self.assertEqual(code.info.size, N // 2)
                self.assertEqual(int(np.argmax(code.reliability)), N - 1)
                self.assertEqual(int(np.argmin(code.reliability)), 0)

    def test_noiseless_roundtrip(self):
        rng = np.random.default_rng(11)
        for N in (64, 128, 256, 512):
            for crc_name in (None, "CRC-6", "CRC-11", "CRC-24C"):
                c = crcmod.CRC(crc_name) if crc_name else None
                code = polar.PolarCode(N, N // 2, c, design="bsc", design_param=0.05)
                pay = rng.integers(0, 2, (20, code.k_payload), dtype=np.uint8)
                llr = channel.Channel("lossless").transmit(code.encode(pay), rng)["llr"]
                np.testing.assert_array_equal(code.decode_sc(llr)["payload"], pay)
                out = code.decode_scl(llr, 4)
                np.testing.assert_array_equal(out["payload"], pay)
                if c is not None:
                    self.assertTrue(out["crc_ok"].all())

    def test_exact_boxplus_option(self):
        rng = np.random.default_rng(12)
        code = polar.PolarCode(128, 64, crcmod.CRC("CRC-6"), f_mode="exact")
        pay = rng.integers(0, 2, (50, code.k_payload), dtype=np.uint8)
        y = channel.Channel("awgn", ebn0_db=4.0, rate=code.R_eff).transmit(code.encode(pay), rng)
        self.assertLessEqual((code.decode_scl(y["llr"], 8)["payload"] != pay).any(axis=1).sum(), 2)
        a, b = rng.normal(0, 3, 1000), rng.normal(0, 3, 1000)
        np.testing.assert_allclose(polar.f_exact(a, b), 2 * np.arctanh(np.tanh(a / 2) * np.tanh(b / 2)),
                                   rtol=1e-6, atol=1e-9)

    def test_scl_L1_equals_sc(self):
        rng = np.random.default_rng(13)
        for N, ch in ((256, channel.Channel("bsc", p=0.06)), (128, channel.Channel("awgn", ebn0_db=1.0, rate=0.4)),
                      (512, channel.Channel("bsc", p=0.09))):
            for c in (None, crcmod.CRC("CRC-11")):
                code = polar.PolarCode(N, N // 2, c, design="bsc", design_param=0.06)
                pay = rng.integers(0, 2, (300, code.k_payload), dtype=np.uint8)
                llr = ch.transmit(code.encode(pay), rng)["llr"]
                sc, scl = code.decode_sc(llr), code.decode_scl(llr, 1)
                np.testing.assert_array_equal(sc["info_bits"], scl["info_bits"])
                np.testing.assert_array_equal(sc["x_hat"], scl["x_hat"])

    def test_list_rescues_sc_failures(self):
        rng = np.random.default_rng(14)
        code = polar.PolarCode(256, 128, crcmod.CRC("CRC-11"), design="bsc", design_param=0.05)
        pay = rng.integers(0, 2, (200, code.k_payload), dtype=np.uint8)
        x = code.encode(pay)
        y = x.copy()
        for row in y:                                         # inject exactly 14 errors per block
            row[rng.choice(256, 14, replace=False)] ^= 1
        llr = (1.0 - 2.0 * y) * np.log(0.95 / 0.05)
        sc_ok = (code.decode_sc(llr)["payload"] == pay).all(axis=1)
        scl = code.decode_scl(llr, 8)
        scl_ok = (scl["payload"] == pay).all(axis=1)
        self.assertGreater(int((~sc_ok & scl_ok).sum()), 0)     # rescued blocks exist
        self.assertGreater(scl_ok.sum(), sc_ok.sum())
        # whenever CA-SCL reports CRC pass the decision is (almost surely) right
        self.assertGreaterEqual(int((scl["crc_ok"] == scl_ok).sum()), 198)

    def test_list_size_monotone(self):
        rng = np.random.default_rng(15)
        code = polar.PolarCode(128, 64, crcmod.CRC("CRC-6"), design="awgn", design_param=2.0)
        pay = rng.integers(0, 2, (600, code.k_payload), dtype=np.uint8)
        llr = channel.Channel("awgn", ebn0_db=1.5, rate=code.R_eff).transmit(code.encode(pay), rng)["llr"]
        fer = [(code.decode_scl(llr, L)["payload"] != pay).any(axis=1).mean() for L in (1, 4, 16)]
        self.assertTrue(fer[0] > fer[1] >= fer[2], fer)


# ======================================================================== optional tier 2: PAC
class TestPAC(unittest.TestCase):
    def test_generator_and_invertibility(self):
        self.assertEqual(int("".join(map(str, polar.PAC_GENERATOR)), 2), 0o133)
        rng = np.random.default_rng(19)
        v = rng.integers(0, 2, (50, 128), dtype=np.uint8)
        np.testing.assert_array_equal(polar.conv_deprecode(polar.conv_precode(v)), v)
        # explicit Toeplitz matrix has full rank over GF(2) (unit diagonal, upper triangular)
        T = polar.conv_precode(np.eye(16, dtype=np.uint8))
        self.assertTrue(np.all(np.diag(T) == 1) and not np.tril(T, -1).any())

    def test_rm_profile(self):
        code = polar.PACCode(256, 117)
        wt = np.array([bin(i).count("1") for i in code.info])
        self.assertTrue((wt >= 4).all())                       # 93 indices of weight >= 5 + 24 of weight 4
        self.assertEqual(int((wt >= 5).sum()), 93)
        self.assertTrue(code.validate_frozen_set())

    def test_pac_noiseless_and_noisy(self):
        rng = np.random.default_rng(20)
        for N in (64, 128, 256, 512):
            code = polar.PACCode(N, N // 2 - 5)
            pay = rng.integers(0, 2, (10, code.K), dtype=np.uint8)
            llr = channel.Channel("lossless").transmit(code.encode(pay), rng)["llr"]
            np.testing.assert_array_equal(code.decode_scl(llr, 4)["payload"], pay)
        code = polar.PACCode(256, 117)
        pay = rng.integers(0, 2, (300, 117), dtype=np.uint8)
        llr = channel.Channel("awgn", ebn0_db=2.5, rate=code.R_eff).transmit(code.encode(pay), rng)["llr"]
        f1 = (code.decode_scl(llr, 1)["payload"] != pay).any(axis=1).mean()
        f8 = (code.decode_scl(llr, 8)["payload"] != pay).any(axis=1).mean()
        self.assertLess(f8, f1)


# ======================================================================== channel
class TestChannel(unittest.TestCase):
    def test_bsc_statistics(self):
        rng = np.random.default_rng(16)
        out = channel.Channel("bsc", p=0.1).transmit(np.zeros(400_000, np.uint8), rng)
        self.assertAlmostEqual(out["errors"] / 400_000, 0.1, delta=0.002)
        self.assertTrue(np.allclose(np.abs(out["llr"]), np.log(9)))

    def test_awgn_llr_and_sigma(self):
        rng = np.random.default_rng(17)
        ch = channel.Channel("awgn", ebn0_db=2.0, rate=0.5)
        self.assertAlmostEqual(ch.sigma2, 1 / (2 * 0.5 * 10 ** 0.2))
        out = ch.transmit(np.ones(200_000, np.uint8), rng)          # bit 1 -> -1
        self.assertAlmostEqual(np.mean(out["y"]), -1.0, delta=0.01)
        self.assertAlmostEqual(np.var(out["y"]), ch.sigma2, delta=0.01)
        np.testing.assert_allclose(out["llr"], 2 * out["y"] / ch.sigma2)

    def test_validation(self):
        for bad in (dict(kind="bsc", p=0.5), dict(kind="bsc", p=0.0), dict(kind="awgn"), dict(kind="x")):
            with self.assertRaises(ValueError):
                channel.Channel(**bad)


# ======================================================================== end to end
class TestEndToEnd(unittest.TestCase):
    def test_determinism(self):
        for ch in (channel.Channel("bsc", p=0.07), channel.Channel("awgn", ebn0_db=1.0, rate=117 / 256)):
            a = main.run_pipeline("determinism check: the same seed gives the same run", "text", ch, seed=123)
            b = main.run_pipeline("determinism check: the same seed gives the same run", "text", ch, seed=123)
            for k in ("n_comp", "chan_errors", "corrected", "src_errors", "match", "rescued"):
                self.assertEqual(a[k], b[k])
            np.testing.assert_array_equal(a["crc_ok"], b["crc_ok"])
            np.testing.assert_array_equal(a["rx_hard"], b["rx_hard"])
            c = main.run_pipeline("determinism check: the same seed gives the same run", "text", ch, seed=124)
            self.assertFalse(np.array_equal(a["rx_hard"], c["rx_hard"]))

    def test_lossless_200_random_inputs(self):
        rng = np.random.default_rng(18)
        alphabet = list("abcdefghijklmnopqrstuvwxyz ABCXYZ0123456789.,!?\n") + ["é", "ß", "中",
                                                                                     "\U0001F600", "न"]
        crcs = ["CRC-6", "CRC-11", "CRC-24C"]
        for i in range(200):
            N = int(rng.choice([64, 128, 256, 512]))
            crc_name = crcs[i % 3]
            K = N // 2
            warm = bool(i % 2)
            if i % 4 < 3:
                data, mode = "".join(rng.choice(alphabet, int(rng.integers(1, 120)))), "text"
            else:
                data, mode = rng.integers(0, 2, int(rng.integers(1, 300))).tolist(), "binary"
            res = main.run_pipeline(data, mode, channel.Channel("lossless"), seed=i, N=N, K=K, crc_name=crc_name,
                                    L=int(rng.choice([1, 4, 8])), warm=warm, compare_sc=False, cold_size=False)
            self.assertTrue(res["match"], f"case {i}: {data!r}")
            self.assertTrue(res["crc_ok"].all())
            self.assertEqual(res["src_errors"], 0)

    def test_noisy_pass_and_fail_reporting(self):
        msg = "my name is sunny"
        ok = main.run_pipeline(msg, "text", channel.Channel("bsc", p=0.03), seed=42)
        self.assertTrue(ok["match"])
        bad = main.run_pipeline(msg * 4, "text", channel.Channel("bsc", p=0.2), seed=1)
        self.assertFalse(bad["match"])
        self.assertTrue(bad["fail_reasons"])


if __name__ == "__main__":
    unittest.main()
