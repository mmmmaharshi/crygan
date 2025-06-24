import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from math import log2, sqrt

import numpy as np
from scipy.stats import chisquare

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
    return -sum((freq / total) * log2(freq / total) for freq in c.values())


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


def log_test_result(name, pval, passed, note=""):
    if passed is None and note:
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
                if "too small" in p[0].lower() or "j < 500" in p[0].lower():
                    return name, None, None, "Skipped: J < 500"
        pval = None
        if isinstance(p, (list, tuple)):
            flat = []
            for item in p:
                if isinstance(item, (float, int)):
                    flat.append(item)
                elif isinstance(item, (list, tuple)):
                    flat += [v for v in item if isinstance(v, (float, int))]
            if flat:
                pval = flat[0]
        elif isinstance(p, (float, int)):
            pval = p
        passed = (pval is not None) and (pval >= 0.01)
        note = ""
    except Exception as e:
        pval = None
        msg = str(e).lower()
        if "insufficient data" in msg or "bits required" in msg:
            note = "Insufficient bits"
        elif "j too small" in msg or "j < 500" in msg:
            note = "Skipped: J < 500"
            return name, None, None, note
        else:
            note = str(e)
        passed = False
    return name, pval, passed, note


def main():
    path = "outputs/keys/gan_expanded_1mbit.bin"
    if not os.path.exists(path):
        print(f"[✗] Missing key file: {path}")
        return

    t0 = time.time()
    bits = load_key_bin(path)
    print(f"[✓] Loaded {len(bits)} bits")

    print("\n[📊] Basic Key Strength Tests")
    entropy = shannon_entropy(bits)
    s, ones, zeros, bias = frequency_monobit(bits)
    r, exp_r, runs_diff = runs_stat(bits)
    chi, p_chi = chi_square_stat(bits)

    basic = [
        ("Shannon Entropy", entropy, entropy >= 0.997, "≥0.997"),
        ("Monobit Bias", bias, bias <= 0.01, f"ones={ones},zeros={zeros}"),
        ("Runs Deviation", runs_diff, runs_diff <= 0.01, f"runs={r},exp≈{int(exp_r)}"),
        ("Chi-Square p", p_chi, p_chi >= 0.05, "≥0.05"),
    ]
    passed_basic = 0
    for name, val, ok, note in basic:
        log_test_result(name, val, ok, note if not ok else "")
        passed_basic += ok

    print("\n[🧪] NIST SP800-22 Full Test Battery (randomness_testsuite)")
    bitstr = "".join(map(str, bits[:2000000]))
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

    passed_suite = 0
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_nist_test, name, fn, bitstr) for name, fn in suite
        ]
        for f in futures:
            result = f.result()
            results.append(result)

    for name, pval, ok, note in results:
        log_test_result(name, pval, ok, note if not ok else "")
        if ok:
            passed_suite += 1

    total_time = time.time() - t0
    print(f"\n[📋] Basic: 4 total, {passed_basic} passed, {4 - passed_basic} failed")
    print(
        f"[📋] NIST: {len(suite)} total, {passed_suite} passed, {len([r for r in results if r[2] is False])} failed, {len([r for r in results if r[2] is None])} skipped"
    )
    print(f"[⏱️] Total runtime: {total_time:.2f} sec")


if __name__ == "__main__":
    main()
