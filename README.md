# Hybrid GAN–Physical Chaotic Seeding for Cryptographic Key Generation

A research-grade system for cryptographic key generation using hybrid entropy from physical chaotic sources and deep generative models.

---

## 🔍 Abstract

This project combines physical noise (WAV audio) with a GAN to generate high-entropy binary keys. A 256-bit GAN-generated seed is cryptographically expanded using SHA-256 chaining to produce a 1,000,000-bit bitstream. Final outputs are validated using ENT randomness tests.

---

## ⚙️ Pipeline

1. **Entropy Source**: Quantized WAV waveform ➝ binary samples
2. **Training**: GAN learns to model entropy structure (G vs. D)
3. **Key Synthesis**: Final 256-bit GAN output selected
4. **Expansion**: SHA-256 chaining ➝ 1M bits
5. **Evaluation**: ENT test validates entropy, randomness

---