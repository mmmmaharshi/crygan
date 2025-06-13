import os

import numpy as np
import scipy.io.wavfile as wav


def load_wav_as_entropy_bits(path, bit_length=256, threshold=0.001):
    # Step 1: Load audio
    sr, data = wav.read(path)

    # Step 2: Normalize (int16 → float)
    if data.dtype == np.int16:
        data = data / 32768.0
    if data.ndim > 1:
        data = data[:, 0]  # Use one channel if stereo

    # Step 3: Preprocess
    data = data[: bit_length * 4]
    data = data - np.mean(data)

    # Step 4: Threshold → binary
    bits = (data > threshold).astype(np.uint8)

    # Step 5: Pad or trim to fixed size
    if len(bits) < bit_length:
        bits = np.pad(bits, (0, bit_length - len(bits)))
    else:
        bits = bits[:bit_length]

    return bits, data


wav_path = "1-101296-B-19.wav"
entropy_path = "chaotic_entropy.npy"

# Extract entropy
bits, raw_data = load_wav_as_entropy_bits(wav_path)

# Save
np.save(entropy_path, bits)

# Display
print("Chaotic entropy bits:", bits)
print("Entropy quality (0s/1s):", np.sum(bits == 0), "/", np.sum(bits == 1))
print(f"Entropy saved to: {os.path.abspath(entropy_path)}")
