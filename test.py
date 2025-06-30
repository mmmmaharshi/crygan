import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from math import log2, sqrt

import numpy as np
from scipy.stats import chisquare
from randtest import random_score

from randomness_testsuite import (
    ApproximateEntropy,
    Complexity,
    CumulativeSum,
    FrequencyTest,
    RandomExcursions,
    RunTest,
    Serial,
    Spectral,
    TemplateMatching,
    Universal,
)


def load_key_bin(path):
    with open(path, "rb") as f:
        return np.unpackbits(np.frombuffer(f.read(), dtype=np.uint8))


def shannon_entropy(bits):
    c = Counter(bits)
    total = len(bits)
    return -sum((v / total) * log2(v / total) for v in c.values())


def frequency_monobit(bits):
    ones = int(bits.sum())
    zeros = len(bits) - ones
    diff = abs(ones - zeros)
    bias = diff / len(bits)
    s = diff / sqrt(len(bits))
    return s, ones, zeros, bias


def runs_stat(bits):
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i - 1]:
            runs += 1
    exp = ((2 * bits.sum()) * (len(bits) - bits.sum()) / len(bits)) + 1
    diff = abs(runs - exp) / exp
    return runs, exp, diff


def chi_square_stat(bits):
    cnt0 = int((bits == 0).sum())
    cnt1 = int((bits == 1).sum())
    chi, p = chisquare([cnt0, cnt1])
    return chi, p


def bientropy(bits):
    if isinstance(bits, list):
        bits = np.array(bits)
    n = len(bits)
    if n < 2:
        return 0
    e = 0
    current = bits.copy()
    for i in range(n.bit_length() - 1):
        p1 = np.sum(current) / len(current)
        p0 = 1 - p1
        h = 0
        if p0 > 0:
            h -= p0 * log2(p0)
        if p1 > 0:
            h -= p1 * log2(p1)
        e += h
        current = np.bitwise_xor(current[:-1], current[1:])
    return e / (n.bit_length() - 1)


def log_test_result(name, pval, passed, note=""):
    if passed is None:
        print(f"-  {name:<40}: {note}")
    else:
        status = "[PASSED]" if passed else "[FAILED]"
        pstr = f"{pval:.6f}" if pval is not None else "N/A"
        print(f"{status} {name:<40}: p = {pstr}")


def run_nist_test(name, fn, bitstr):
    try:
        p = fn(bitstr)
        if name in ("Random Excursion", "Random Excursion Variant"):
            if isinstance(p, (list, tuple)) and isinstance(p[0], str):
                return name, None, None, "Skipped: J < 500"
        pval = None
        if isinstance(p, (list, tuple)):
            flat = [
                x
                for part in p
                for x in (part if isinstance(part, (list, tuple)) else [part])
                if isinstance(x, (int, float))
            ]
            if flat:
                pval = flat[0]
        elif isinstance(p, (int, float)):
            pval = p
        passed = pval is not None and pval >= 0.01
        note = ""
    except Exception as e:
        pval = None
        msg = str(e).lower()
        if "insufficient data" in msg or "bits required" in msg:
            note = "Insufficient bits"
        elif "j < 500" in msg:
            return name, None, None, "Skipped: J < 500"
        else:
            note = msg
        passed = False
    return name, pval, passed, note


def main():
    start_time = time.time()
    path = "outputs/keys/gan_expanded_1mbit.bin"
    if not os.path.exists(path):
        print(f"[✗] Missing key file: {path}")
        return

    bits = load_key_bin(path)
    print(f"[✓] Loaded {len(bits)} bits\n")

    # Quick Tests
    print("[⚡] Quick Randomness Checks")
    rand_score = random_score(bits)
    bien = bientropy(bits)
    print(f"🔹 randtest score         : {rand_score}")
    print(f"🔹 BiEntropy              : {bien:.6f}\n")

    # Basic Tests
    print("[📊] Basic Key Strength Tests")
    entropy = shannon_entropy(bits)
    s, ones, zeros, bias = frequency_monobit(bits)
    r, exp_r, runs_diff = runs_stat(bits)
    chi, p_chi = chi_square_stat(bits)

    basic = [
        ("Shannon Entropy", entropy, entropy >= 0.997, "threshold ≥ 0.997"),
        ("Monobit Bias", bias, bias <= 0.01, f"ones={ones}, zeros={zeros}"),
        ("Runs Deviation", runs_diff, runs_diff <= 0.01, f"runs={r}, exp≈{int(exp_r)}"),
        ("Chi‑Square p", p_chi, p_chi >= 0.05, "threshold ≥ 0.05"),
    ]
    passed_basic = sum(ok for _, _, ok, _ in basic)
    for name, val, ok, note in basic:
        log_test_result(name, val, ok, "" if ok else note)
    print()

    # NIST SP800‑22
    print("[🧪] NIST SP800‑22 Full Battery")
    bitstr = "".join(map(str, bits[:2_000_000]))
    suite = [
        ("Monobit", FrequencyTest.FrequencyTest.monobit_test),
        ("Block Frequency", FrequencyTest.FrequencyTest.block_frequency),
        ("Runs", RunTest.RunTest.run_test),
        ("Longest Run", RunTest.RunTest.longest_one_block_test),
        ("DFT", Spectral.SpectralTest.spectral_test),
        (
            "Non-overlap Template",
            TemplateMatching.TemplateMatching.non_overlapping_test,
        ),
        (
            "Overlapping Template",
            TemplateMatching.TemplateMatching.overlapping_patterns,
        ),
        ("Universal", Universal.Universal.statistical_test),
        ("Linear Complexity", Complexity.ComplexityTest.linear_complexity_test),
        ("Serial", Serial.Serial.serial_test),
        (
            "Approximate Entropy",
            ApproximateEntropy.ApproximateEntropy.approximate_entropy_test,
        ),
        ("Cumulative Sums", CumulativeSum.CumulativeSums.cumulative_sums_test),
        ("Random Excursion", RandomExcursions.RandomExcursions.random_excursions_test),
        ("Random Excursion Variant", RandomExcursions.RandomExcursions.variant_test),
    ]

    with ProcessPoolExecutor() as exec:
        futures = [exec.submit(run_nist_test, name, fn, bitstr) for name, fn in suite]
        results = [f.result() for f in futures]

    passed_suite = sum(1 for _, _, ok, _ in results if ok)
    for name, pval, ok, note in results:
        log_test_result(name, pval, ok, note)

    total_time = time.time() - start_time

    print(f"\n📋 Basic: 4 total, {passed_basic} passed, {4 - passed_basic} failed")
    n_fail = sum(1 for _, _, ok, _ in results if ok is False)
    n_skip = sum(1 for _, _, ok, _ in results if ok is None)
    print(
        f"📋 NIST: {len(suite)} total, {passed_suite} passed, {n_fail} failed, {n_skip} skipped"
    )
    print(f"⏱ Total runtime: {total_time:.2f} sec")


if __name__ == "__main__":
    main()
