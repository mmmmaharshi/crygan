import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from math import log2, sqrt

import numpy as np
from randtest import random_score
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


def min_entropy(bits):
    c = Counter(bits)
    total = len(bits)
    pmax = max(v / total for v in c.values())
    return -log2(pmax)


def collision_entropy(bits):
    c = Counter(bits)
    total = len(bits)
    return -log2(sum((v / total) ** 2 for v in c.values()))


def hartley_entropy(bits):
    c = Counter(bits)
    k = len(c)
    return log2(k) if k > 0 else 0


def renyi_entropy(bits, alpha=2):
    c = Counter(bits)
    total = len(bits)
    if alpha == 1:
        return shannon_entropy(bits)
    return 1 / (1 - alpha) * log2(sum((v / total) ** alpha for v in c.values()))


def tsallis_entropy(bits, q=2):
    c = Counter(bits)
    total = len(bits)
    if q == 1:
        return shannon_entropy(bits)
    return (1 - sum((v / total) ** q for v in c.values())) / (q - 1)


def permutation_entropy(bits, order=3):
    # For binary, this is a simple sliding window
    n = len(bits)
    if n < order:
        return 0
    patterns = Counter()
    for i in range(n - order + 1):
        patterns[tuple(bits[i : i + order])] += 1
    total = sum(patterns.values())
    return -sum((v / total) * log2(v / total) for v in patterns.values())


def sample_entropy(bits, m=2):
    # Approximate for binary
    n = len(bits)
    if n <= m + 1:
        return 0

    def _phi(m):
        patterns = Counter(tuple(bits[i : i + m]) for i in range(n - m + 1))
        return sum(v * (v - 1) for v in patterns.values()) / ((n - m + 1) * (n - m))

    try:
        return -log2(_phi(m + 1) / _phi(m))
    except (ZeroDivisionError, ValueError):
        return 0


def guessing_entropy(bits):
    c = Counter(bits)
    total = len(bits)
    p = sorted((v / total) for v in c.values())[::-1]
    return sum((i + 1) * pi for i, pi in enumerate(p))


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


def block_entropy(bits, block_size=8):
    n = len(bits)
    if n < block_size:
        return 0
    blocks = Counter(
        tuple(bits[i : i + block_size])
        for i in range(0, n - block_size + 1, block_size)
    )
    total = sum(blocks.values())
    return -sum((v / total) * log2(v / total) for v in blocks.values())


def conditional_entropy(bits):
    # H(X|X-1) for binary
    n = len(bits)
    if n < 2:
        return 0
    joint = Counter((bits[i - 1], bits[i]) for i in range(1, n))
    marginal = Counter(bits[:-1])
    total = n - 1
    h = 0
    for (x_prev, x), v in joint.items():
        p_joint = v / total
        p_prev = marginal[x_prev] / total
        h -= p_joint * log2(p_joint / p_prev)
    return h


def cross_entropy(bits, qbits):
    # Cross entropy between two bit sequences
    c = Counter(bits)
    q = Counter(qbits)
    total = len(bits)
    total_q = len(qbits)
    h = 0
    for k in c:
        p = c[k] / total
        qv = q.get(k, 1e-12) / total_q
        h -= p * log2(qv)
    return h


def relative_entropy(bits, qbits):
    # KL divergence D(P||Q)
    c = Counter(bits)
    q = Counter(qbits)
    total = len(bits)
    total_q = len(qbits)
    d = 0
    for k in c:
        p = c[k] / total
        qv = q.get(k, 1e-12) / total_q
        d += p * log2(p / qv)
    return d


def lempel_ziv_entropy(bits):
    # Lempel-Ziv complexity as entropy estimate (normalized)
    n = len(bits)
    s = "".join(map(str, bits))
    i, k, c = 0, 1, 1
    while True:
        if i + k > n:
            c += 1
            break
        if s[i : i + k] not in s[0:i]:
            c += 1
            i += k
            k = 1
        else:
            k += 1
            if i + k > n:
                c += 1
                break
    return c * log2(n) / n if n > 0 else 0


def spectral_entropy(bits):
    # Entropy of the normalized power spectrum
    n = len(bits)
    if n < 2:
        return 0
    x = np.array(bits) - np.mean(bits)
    spectrum = np.abs(np.fft.fft(x))[: n // 2]
    psd = spectrum**2
    psd /= np.sum(psd)
    return -np.sum(psd * np.log2(psd + 1e-12))


def multi_scale_entropy(bits, max_scale=5):
    # Average sample entropy over multiple scales
    entropies = []
    for scale in range(1, max_scale + 1):
        coarse = [
            np.mean(bits[i : i + scale]) > 0.5 for i in range(0, len(bits), scale)
        ]
        entropies.append(sample_entropy(np.array(coarse, dtype=int), m=2))
    return np.mean(entropies)


def markov_entropy(bits):
    # Entropy rate for a first-order Markov chain
    n = len(bits)
    if n < 2:
        return 0
    trans = Counter((bits[i - 1], bits[i]) for i in range(1, n))
    marginal = Counter(bits[:-1])
    total = n - 1
    h = 0
    for (x_prev, x), v in trans.items():
        p_joint = v / total
        p_prev = marginal[x_prev] / total
        h -= p_joint * log2(p_joint / p_prev)
    return h


def maurer_universal_test(bits, L=7, Q=1280, K=512):
    # Maurer's Universal Statistical Test for randomness (binary version)
    n = len(bits)
    if n < Q + K:
        return 0, 0
    T = [0] * (2**L)
    for i in range(Q):
        dec = 0
        for j in range(L):
            dec = (dec << 1) | bits[i * L + j]
        T[dec] = i + 1
    sum_ = 0
    for i in range(Q, Q + K):
        dec = 0
        for j in range(L):
            dec = (dec << 1) | bits[i * L + j]
        d = i + 1 - T[dec]
        T[dec] = i + 1
        sum_ += np.log2(d)
    fn = sum_ / K
    # Expected value and variance for L=7
    expected = 6.1962507
    variance = 3.125
    z = (fn - expected) / np.sqrt(variance / K)
    return fn, z


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
    minent = min_entropy(bits)
    collent = collision_entropy(bits)
    hartley = hartley_entropy(bits)
    renyi2 = renyi_entropy(bits, alpha=2)
    tsallis2 = tsallis_entropy(bits, q=2)
    perm_ent = permutation_entropy(bits, order=3)
    samp_ent = sample_entropy(bits, m=2)
    guess_ent = guessing_entropy(bits)
    block_ent = block_entropy(bits, block_size=8)
    cond_ent = conditional_entropy(bits)
    # For cross/relative entropy, use bits vs. reversed bits as a simple example
    cross_ent = cross_entropy(bits, bits[::-1])
    rel_ent = relative_entropy(bits, bits[::-1])
    lz_ent = lempel_ziv_entropy(bits)
    spec_ent = spectral_entropy(bits)
    ms_ent = multi_scale_entropy(bits, max_scale=5)
    markov_ent = markov_entropy(bits)
    maurer_fn, maurer_z = maurer_universal_test(bits)
    print(f"🔹 randtest score         : {rand_score}")
    print(f"🔹 BiEntropy              : {bien:.6f}")
    print(f"🔹 Min-Entropy            : {minent:.6f}")
    print(f"🔹 Collision Entropy      : {collent:.6f}")
    print(f"🔹 Hartley Entropy        : {hartley:.6f}")
    print(f"🔹 Rényi Entropy (α=2)    : {renyi2:.6f}")
    print(f"🔹 Tsallis Entropy (q=2)  : {tsallis2:.6f}")
    print(f"🔹 Permutation Entropy    : {perm_ent:.6f}")
    print(f"🔹 Sample Entropy         : {samp_ent:.6f}")
    print(f"🔹 Guessing Entropy       : {guess_ent:.6f}")
    print(f"🔹 Block Entropy (8)      : {block_ent:.6f}")
    print(f"🔹 Conditional Entropy    : {cond_ent:.6f}")
    print(f"🔹 Cross Entropy (rev)    : {cross_ent:.6f}")
    print(f"🔹 Relative Entropy (rev) : {rel_ent:.6f}")
    print(f"🔹 Lempel-Ziv Entropy     : {lz_ent:.6f}")
    print(f"🔹 Spectral Entropy       : {spec_ent:.6f}")
    print(f"🔹 Multi-Scale Entropy    : {ms_ent:.6f}")
    print(f"🔹 Markov Entropy         : {markov_ent:.6f}")
    print(f"🔹 Maurer's Universal Statistic: {maurer_fn:.6f} (z={maurer_z:.2f})\n")

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
