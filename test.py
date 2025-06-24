import os
import time
from collections import Counter
from math import log2, sqrt
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.stats import chisquare


def load_key_bin(path):
    with open(path, "rb") as f:
        return np.unpackbits(np.frombuffer(f.read(), dtype=np.uint8))


def shannon_entropy(bits):
    c = Counter(bits)
    total = len(bits)
    return -sum((freq / total) * log2(freq / total) for freq in c.values())


def frequency_monobit_test(bits):
    ones = np.sum(bits)
    zeros = len(bits) - ones
    s = abs(ones - zeros) / sqrt(len(bits))
    return s, ones, zeros


def runs_test(bits):
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i - 1]:
            runs += 1
    expected_runs = ((2 * np.sum(bits)) * (len(bits) - np.sum(bits)) / len(bits)) + 1
    return runs, expected_runs


def chi_square_test(bits):
    bit_counts = [np.sum(bits == 0), np.sum(bits == 1)]
    chi, p = chisquare(bit_counts)
    return chi, p


def parallel_tests(bits):
    with Pool(processes=cpu_count()) as pool:
        entropy_result = pool.apply_async(shannon_entropy, (bits,))
        monobit_result = pool.apply_async(frequency_monobit_test, (bits,))
        runs_result = pool.apply_async(runs_test, (bits,))
        chi_result = pool.apply_async(chi_square_test, (bits,))
        return (
            entropy_result.get(),
            monobit_result.get(),
            runs_result.get(),
            chi_result.get(),
        )


def verdict(label, value, threshold, higher_better=True, label_info=""):
    ok = (value >= threshold) if higher_better else (value <= threshold)
    symbol = "✅" if ok else "❌"
    return f"{symbol} {label:<22}: {value:.6f}   ({label_info})"


def main():
    bin_path = "outputs/keys/gan_expanded_1mbit.bin"
    if not os.path.exists(bin_path):
        print(f"[✗] Key not found at: {bin_path}")
        return

    start_time = time.time()

    bits = load_key_bin(bin_path)
    print(f"[✓] Loaded {len(bits)} bits")

    entropy, (s, ones, zeros), (runs, expected_runs), (chi, p_chi) = parallel_tests(
        bits
    )

    print("\n[📊] Key Strength Summary")
    print(verdict("Shannon Entropy", entropy, 0.997, True, "ideal ≥ 0.997"))
    print(
        verdict(
            "Monobit Bias",
            abs(ones - zeros) / len(bits),
            0.01,
            False,
            f"ones = {ones}, zeros = {zeros}",
        )
    )
    print(
        verdict(
            "Runs Deviation",
            abs(runs - expected_runs) / expected_runs,
            0.01,
            False,
            f"runs = {runs}, expected ≈ {int(expected_runs)}",
        )
    )
    print(verdict("Chi-Square p", p_chi, 0.05, True, "pass if ≥ 0.05"))

    elapsed = time.time() - start_time

    tests = [
        entropy >= 0.997,
        abs(ones - zeros) / len(bits) <= 0.01,
        abs(runs - expected_runs) / expected_runs <= 0.01,
        p_chi >= 0.05,
    ]
    passed = sum(tests)
    total = len(tests)

    print(
        f"\n[🧪] Tests Run: {total} | ✅ Passed: {passed} | ❌ Failed: {total - passed}"
    )
    print(f"[⏱️] Testing completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
