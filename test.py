import os
import time
import numpy as np
from collections import Counter
from math import log2
from scipy.stats import chisquare
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

from randomness_testsuite import (
    FrequencyTest,
    RunTest,
    Spectral,
    Universal,
)

def load_key_bin(path):
    with open(path, "rb") as f:
        return np.unpackbits(np.frombuffer(f.read(), dtype=np.uint8))

# === Entropy Functions ===
def shannon_entropy(bits):
    c = Counter(bits)
    total = len(bits)
    return -sum((v / total) * log2(v / total) for v in c.values())

def min_entropy(bits):
    c = Counter(bits)
    total = len(bits)
    pmax = max(v / total for v in c.values())
    return -log2(pmax)

def collision_entropy(bits):
    c = Counter(bits)
    total = len(bits)
    return -log2(sum((v / total) ** 2 for v in c.values()))

def permutation_entropy(bits, order=3):
    n = len(bits)
    if n < order:
        return 0
    patterns = Counter(tuple(bits[i:i+order]) for i in range(n - order + 1))
    total = sum(patterns.values())
    return -sum((v / total) * log2(v / total) for v in patterns.values())

def sample_entropy(bits, m=2):
    n = len(bits)
    if n <= m + 1:
        return 0

    def _phi(m):
        patterns = Counter(tuple(bits[i:i+m]) for i in range(n - m + 1))
        return sum(v * (v - 1) for v in patterns.values()) / ((n - m + 1) * (n - m))

    try:
        return -log2(_phi(m + 1) / _phi(m))
    except (ZeroDivisionError, ValueError):
        return 0

def markov_entropy(bits):
    trans = {'00': 0, '01': 0, '10': 0, '11': 0}
    for i in range(len(bits) - 1):
        trans[f"{bits[i]}{bits[i + 1]}"] += 1
    total = sum(trans.values())
    if total == 0:
        return 0
    return -sum((v / total) * log2(v / total) for v in trans.values() if v > 0)

def chi_square_stat(bits):
    cnt0 = int((bits == 0).sum())
    cnt1 = int((bits == 1).sum())
    chi, p = chisquare([cnt0, cnt1])
    return chi, p

# === Parallel Test Execution ===
def run_named_test(name, bits, bits_float):
    if name == "Monobit Test":
        return name, FrequencyTest.FrequencyTest.monobit_test(bits)
    elif name == "Chi-Square Test":
        return name, chi_square_stat(bits)[1]
    elif name == "Runs Test":
        return name, RunTest.RunTest.run_test("".join(map(str, bits)))
    elif name == "Permutation Entropy":
        return name, permutation_entropy(bits)
    elif name == "Sample Entropy":
        return name, sample_entropy(bits)
    elif name == "Shannon Entropy":
        return name, shannon_entropy(bits)
    elif name == "Min Entropy":
        return name, min_entropy(bits)
    elif name == "Collision Entropy":
        return name, collision_entropy(bits)
    elif name == "Markov Entropy":
        return name, markov_entropy(bits)
    elif name == "Maurer's Universal Test":
        return name, Universal.Universal.statistical_test("".join(map(str, bits)))
    elif name == "Spectral Entropy":
        return name, Spectral.SpectralTest.spectral_test(bits_float) if len(bits_float) >= 32 else 0.0
    else:
        return name, "N/A"

def is_pass(name, val):
    if name in ["Shannon Entropy", "Min Entropy", "Collision Entropy"]:
        return val >= 0.95
    elif name == "Markov Entropy":
        return val >= 1.90
    elif name == "Permutation Entropy":
        return val >= 2.5
    elif name == "Sample Entropy":
        return val >= 0.8
    elif isinstance(val, tuple):
        return val[1] is True if len(val) == 2 else None
    elif isinstance(val, float):
        return val >= 0.01
    return None

def main():
    start_time = time.time()
    path = "outputs/keys/gan_expanded_1mbit.bin"
    if not os.path.exists(path):
        print(f"[✗] Missing key file: {path}")
        return

    bits = load_key_bin(path)
    bits_float = bits.astype(float)

    print(f"Loaded {len(bits)} bits\n")

    test_names = [
        "Monobit Test",
        "Chi-Square Test",
        "Runs Test",
        "Permutation Entropy",
        "Sample Entropy",
        "Shannon Entropy",
        "Min Entropy",
        "Collision Entropy",
        "Markov Entropy",
        "Maurer's Universal Test",
        "Spectral Entropy",
    ]

    test_results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(run_named_test, name, bits, bits_float): name
            for name in test_names
        }

        pbar = tqdm(total=len(test_names), dynamic_ncols=True)
        pbar.set_description("🧪 Testing")

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                name = futures.pop(fut)
                try:
                    result_name, val = fut.result()
                    test_results.append((result_name, val))
                except Exception as e:
                    test_results.append((name, f"Error: {e}"))
                pbar.set_description(f"[🧪] Testing ({name})")
                pbar.update(1)
        pbar.close()

    print("\nResults:\n")
    for name in test_names:
        for rname, val in test_results:
            if rname == name:
                passed = is_pass(name, val)
                if isinstance(val, float):
                    val_str = f"{val:.10f}"
                elif isinstance(val, tuple):
                    val_str = f"{val[0]:.10f}" if isinstance(val[0], float) else str(val)
                else:
                    val_str = str(val)
                status = "✅ Passed" if passed else "❌ Failed" if passed is not None else "⚠️ N/A"
                print(f"{name:<25}: {val_str}  -->  {status}")
                break

    print(f"\n⏱ Total runtime: {time.time() - start_time:.2f} sec")

if __name__ == "__main__":
    main()
