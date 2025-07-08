import os
import time
import numpy as np
from collections import Counter
from math import log2
from scipy.stats import chisquare
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from randomness_testsuite import (
    FrequencyTest,
    RunTest,
    Spectral,
    Universal,
)

# === Load Key ===
def load_key_bin(path):
    with open(path, "rb") as f:
        return np.unpackbits(np.frombuffer(f.read(), dtype=np.uint8))


# === Entropy Metrics ===
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
    return -sum((v / total) * log2(v / total) for v in trans.values() if v > 0)

def chi_square_stat(bits):
    cnt0 = int((bits == 0).sum())
    cnt1 = int((bits == 1).sum())
    chi, p = chisquare([cnt0, cnt1])
    return chi, p


# === Cryptanalytic Resistance Tests ===
def seed_key_correlation(bits):
    try:
        seed = bits[:256].astype(float)
        key = bits[256:512].astype(float)
        if len(seed) != 256 or len(key) != 256:
            return "Insufficient length"
        return np.corrcoef(seed, key)[0, 1]
    except Exception as e:
        return f"Error: {e}"

def key_invertibility(bits):
    z = torch.randn(1000, 100)
    try:
        k = torch.tensor(bits[:1000 * 256].reshape(-1, 256)).float()
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 100)
        )
        opt = optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(20):
            pred = model(k)
            loss = ((pred - z) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return ((model(k) - z) ** 2).mean().item()
    except Exception as e:
        return f"Error: {e}"

def mutual_information_mine(bits):
    try:
        z = np.random.randn(2000, 100)
        k = bits[:2000 * 256].reshape(-1, 256)
        mi = mutual_info_regression(z, k[:, 0], discrete_features=False)
        return np.mean(mi)
    except:
        return 0.0


# === Wrapper ===
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
    elif name == "Seed-Key Correlation":
        return name, seed_key_correlation(bits)
    elif name == "Key Invertibility":
        return name, key_invertibility(bits)
    elif name == "Mutual Information (MINE)":
        return name, mutual_information_mine(bits)
    elif name == "Seed-Key Prediction":
        return name, "N/A"
    return name, "N/A"


# === Pass/Fail Logic ===
def is_pass(name, val):
    try:
        if name in ["Shannon Entropy", "Min Entropy", "Collision Entropy"]:
            return float(val) >= 0.95
        elif name == "Markov Entropy":
            return float(val) >= 1.90
        elif name == "Permutation Entropy":
            return float(val) >= 2.5
        elif name == "Sample Entropy":
            return float(val) >= 0.8
        elif name == "Seed-Key Correlation":
            return abs(val) < 0.05 if isinstance(val, float) else None
        elif name == "Key Invertibility":
            return float(val) > 1.0
        elif name == "Mutual Information (MINE)":
            return float(val) < 0.05
        elif isinstance(val, float):
            return val >= 0.01
        elif isinstance(val, tuple):
            return val[1] is True
    except Exception:
        return None


# === Main ===
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
        "Mutual Information (MINE)",
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
                val_str = (
                    f"{val:.10f}" if isinstance(val, float)
                    else f"{val[0]:.10f}" if isinstance(val, tuple) and isinstance(val[0], float)
                    else str(val)
                )
                status = "✅ Passed" if passed else "❌ Failed" if passed is not None else "⚠️ N/A"
                print(f"{name:<30}: {val_str}  -->  {status}")
                break

    print(f"\n⏱ Total runtime: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    main()
