# Hybrid GAN–Physical Chaotic Seeding for Cryptographic Key Generation

A research-grade, modular, and reproducible pipeline for cryptographic key generation using hybrid entropy from physical chaotic sources and deep generative adversarial networks (GANs). This system is designed for robust, high-entropy key synthesis, with comprehensive randomness and entropy validation.

---

## 📚 Table of Contents
- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Components](#components)
  - [1. Entropy Extraction](#1-entropy-extraction)
  - [2. GAN Training](#2-gan-training)
  - [3. Key Expansion](#3-key-expansion)
  - [4. Output Formats](#4-output-formats)
  - [5. Randomness & Entropy Testing](#5-randomness--entropy-testing)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Results & Validation](#results--validation)
- [References](#references)

---

## Overview
This project combines physical noise (WAV audio) with a GAN to generate high-entropy binary keys. A 256-bit GAN-generated seed is cryptographically expanded using HKDF (SHA-256) to produce a 1,000,000-bit bitstream. Final outputs are validated using a comprehensive suite of entropy and randomness tests, including NIST SP800-22 and pyentrp estimators.

---

## Pipeline Architecture

```
[Physical Entropy Sources (WAV)]
        │
        ▼
[Entropy Extraction & Hashing]
        │
        ▼
[GAN Training (WGAN-GP)]
        │
        ▼
[Best GAN Output (256 bits)]
        │
        ▼
[HKDF Expansion (SHA-256)]
        │
        ▼
[1,000,000-bit Key]
        │
        ▼
[Randomness & Entropy Test Suite]
```

---

## Components

### 1. Entropy Extraction
- **Sources:** Physical chaotic signals (e.g., `rain.wav`, `traffic.wav`)
- **Process:**
  - Audio is segmented and hashed (SHA-256) to extract unbiased, high-entropy bitstrings.
  - Segments are processed in parallel for speed.
- **Output:** Numpy array of binary segments, each segment = 256 bits.

### 2. GAN Training
- **Model:** Wasserstein GAN with Gradient Penalty (WGAN-GP)
- **Generator:** Maps latent noise (100D) to 256-bit binary output.
- **Discriminator:** Distinguishes real entropy segments from generated ones.
- **Training:**
  - Uses real entropy segments as positive samples.
  - Tracks entropy of generated samples during training.
  - Saves checkpoints and logs entropy/loss metrics.

### 3. Key Expansion
- **Seed:** Best GAN-generated 256-bit output.
- **Expansion:**
  - HKDF (HMAC-SHA256) expands the seed to 1,000,000 bits.
  - Salt is randomly generated and saved for reproducibility.
- **Output:** Expanded key as binary, hex, base64, and QR code.

### 4. Output Formats
- `gan_generated_key.bin` — Raw binary key (256 bits)
- `gan_generated_key.npy` — Numpy array (256 bits)
- `key.hex` — Hexadecimal string
- `key.b64` — Base64 string
- `key_qr.png` — QR code of base64 key
- `gan_expanded_1mbit.bin` — 1,000,000-bit expanded key
- `salt.bin` — HKDF salt
- `logs/training_metrics.csv` — Training/entropy logs

### 5. Randomness & Entropy Testing
- **Test Suite:** `test.py` runs all tests in parallel, results are categorized and minimal.
- **Categories:**
  - **Entropy Estimators:**
    - Min-Entropy, Collision, Hartley, Rényi, Tsallis, Sample, Permutation, Block, Conditional, Cross, Relative, Lempel-Ziv, Spectral, Multi-Scale, Markov, Maurer's Universal, pyentrp estimators
  - **Statistical/Randomness Tests:**
    - randtest score, Chi-Square p-value
  - **NIST SP800-22 Tests:**
    - Monobit, Block Frequency, Runs, Longest Run, DFT, Non-overlap Template, Overlapping Template, Universal, Linear Complexity, Serial, Approximate Entropy, Cumulative Sums, Random Excursion, Random Excursion Variant
- **Multiprocessing:** All tests run in parallel for speed.
- **Output:** Results are printed in original order, with clear PASS/FAIL/SKIP and error notes.

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Place entropy WAV files** in the project root (e.g., `rain.wav`, `traffic.wav`).
3. **Run the main pipeline:**
   ```bash
   python main.py
   ```
4. **Run the test suite:**
   ```bash
   python test.py
   ```

---

## Configuration
- All main parameters (GAN, entropy, expansion) are set at the top of `main.py`.
- Output and checkpoint paths are configurable.
- Add or change entropy sources by editing the `entropy_sources` list.

---

## Results & Validation
- All outputs are saved in the `outputs/` directory.
- The test suite provides a full breakdown of entropy and randomness properties.
- Training logs and metrics are available for reproducibility and analysis.

---

## References
- NIST SP800-22: A Statistical Test Suite for Random and Pseudorandom Number Generators
- pyentrp: Entropy and complexity estimators in Python
- WGAN-GP: Improved Training of Wasserstein GANs
- HKDF: RFC 5869 HMAC-based Extract-and-Expand Key Derivation Function
- [Project Author/Contact]