## 🚧 WIP 🚧

# CryGAN: Physical Chaotic Seeding for Cryptographic Key Generation

CryGAN is a lightweight adversarial system that generates high-entropy cryptographic keys using a WGAN-GP (Wasserstein GAN with Gradient Penalty), seeded from chaotic physical entropy sources such as environmental audio (e.g., rain, traffic). The model learns the entropy distribution and synthesizes 256-bit cryptographic keys, which are later expanded to 1 Mbit using HKDF. The system supports full testing, evaluation, and export of the keys in multiple formats.


## Methodology

- **Chaotic Entropy Seeding**: Audio files are segmented and passed through SHA-256 to extract randomness at the bit level.
- **Generator (G)**: MLP with dropout and Gaussian noise injection, designed to produce binary-like high-entropy vectors.
- **Discriminator (D)**: MLP with LeakyReLU activations trained under the WGAN-GP framework.
- **Training**: Conducted over adversarial epochs with gradient penalty for stable convergence.
- **Key Expansion**: Final 256-bit output from G is expanded to 1,000,000 bits using HKDF (SHA-256).
- **Export**: Keys are saved in `.bin`, `.hex`, `.b64`, `.npy`, and QR code formats.


## Pipeline Architecture

```
[Physical Entropy Sources (WAV)]
        │
        ▼
[Entropy Extraction via SHA-256 (bit-level)]
        │
        ▼
[Adversarial Training (WGAN-GP)]
        │
        ▼
[GAN Output (256-bit Key Candidate)]
        │
        ▼
[HKDF Expansion (SHA-256, IKM = GAN Output)]
        │
        ▼
[Expanded 1,000,000-bit Key]
        │
        ▼
[Randomness & Entropy Test Suite]
```

## Evaluation Metrics

CryGAN’s keys are evaluated with 12 randomness tests:
- **Statistical Tests**: Monobit, Chi-Square, Runs, Maurer’s Universal
- **Entropy Measures**: Shannon, Min, Collision, Markov, Spectral, Permutation, Sample
- **Information-Theoretic**: MINE (Mutual Information Neural Estimation)

## Usage & Results

This section demonstrates the key generation and evaluation process.

### 1. Key Generation

The `main.py` script runs the WGAN-GP model to generate a 256-bit key and then expands it to 1 Mbit using HKDF.

```bash
>> uv run main.py
```
```bash
H=0.9996: 100% | 1000/1000 [00:56<00:00, 17.67it/s]
[📦] Final entropy: 0.9996
[✅] 1Mbit expanded key saved.
```

### 2. Randomness Testing

The `test.py` script loads the generated 1 Mbit key and subjects it to a comprehensive suite of 12 statistical and entropy-based tests.

```bash
>> uv run test.py
```
```
Loaded 1000000 bits

[🧪] Testing : 100%|12/12 [00:10<00:00, 1.16it/s]

Results:

Monobit Test          : 1.0000000000  -->  ✅ Passed
Chi-Square Test       : 0.6269671659  -->  ✅ Passed
Runs Test             : 0.2208630293  -->  ✅ Passed
Permutation Entropy   : 2.9999962464  -->  ✅ Passed
Sample Entropy        : 1.0000010992  -->  ✅ Passed
Shannon Entropy       : 0.9999998296  -->  ✅ Passed
Min Entropy           : 0.9992990205  -->  ✅ Passed
Collision Entropy     : 0.9999996592  -->  ✅ Passed
Markov Entropy        : 1.9999985799  -->  ✅ Passed
Maurer's Universal Test : 0.9331389526  -->  ✅ Passed
Spectral Entropy      : 0.6863806151  -->  ✅ Passed
Mutual Information (MINE): 0.0046312641  -->  ✅ Passed

⏱ Total runtime: 11.23 sec
```

> This output confirms that the generated key successfully passes all randomness and entropy evaluations, validating the effectiveness of the CryGAN system.