import base64
import os
from collections import Counter
from hashlib import sha256
from math import log2
from multiprocessing import Pool, cpu_count, freeze_support

import numpy as np
import qrcode
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# === CONFIG ===
latent_dim = 100
input_bits = 256
epochs = 3000
batch_size = 32
lr = 0.0002
wav_path = "1-101296-B-19.wav"


# === ENTROPY FUNCTIONS ===
def extract_bits(args):
    data, i, bit_length, threshold = args
    start = i * bit_length * 4
    end = start + bit_length * 4
    segment = data[start:end]
    if len(segment) < bit_length * 4:
        return None
    segment = segment - np.mean(segment)
    bits = (segment > threshold).astype(np.uint8)
    return bits[:bit_length]


def load_entropy_parallel(path, count=1024, bit_length=256, threshold=0.001):
    sr, data = wav.read(path)
    if data.dtype == np.int16:
        data = data / 32768.0
    if data.ndim > 1:
        data = data[:, 0]
    args_list = [(data, i, bit_length, threshold) for i in range(count)]
    with Pool() as pool:
        results = pool.map(extract_bits, args_list)
    return np.array([r for r in results if r is not None])


# === MODELS ===
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_bits),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_bits, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# === EXPANSION FUNCTION ===
def expand_key_sha256(seed_bits, target_bits=1_000_000):
    bits = []
    current = bytes(seed_bits.tolist())
    while len(bits) < target_bits:
        h = sha256(current).digest()
        chunk = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
        bits.extend(chunk)
        current = h
    return np.array(bits[:target_bits], dtype=np.uint8)


# === MAIN LOGIC ===
def main():
    torch.set_num_threads(cpu_count())
    os.makedirs("outputs/keys", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    real_keys = load_entropy_parallel(wav_path)
    print(f"[✓] Loaded {real_keys.shape[0]} samples × {real_keys.shape[1]} bits")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    real_tensor = torch.tensor(real_keys.astype(np.float32), device=device)
    real_tensor = real_tensor * 2 - 1

    def shannon_entropy(bits):
        c = Counter(bits)
        total = len(bits)
        return -sum((f / total) * log2(f / total) for f in c.values())

    entropy_log = []
    g_loss_log = []
    d_loss_log = []
    final_binary = None

    for epoch in tqdm(range(epochs)):
        idx = np.random.randint(0, real_tensor.shape[0], batch_size)
        real_batch = real_tensor[idx]

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_batch = G(z).detach()
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        D_loss = criterion(D(real_batch), real_labels) + criterion(
            D(fake_batch), fake_labels
        )
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_batch = G(z)
        G_loss = criterion(D(fake_batch), real_labels)
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if epoch % 100 == 0:
            z = torch.randn(1, latent_dim, device=device)
            with torch.no_grad():
                sample = G(z).squeeze().cpu().numpy()
            binary = (sample > 0).astype(np.uint8)
            entropy_log.append(shannon_entropy(binary))
            g_loss_log.append(G_loss.item())
            d_loss_log.append(D_loss.item())
            final_binary = binary

    # === FINAL OUTPUT ===
    if final_binary is None:
        z = torch.randn(1, latent_dim, device=device)
        with torch.no_grad():
            sample = G(z).squeeze().cpu().numpy()
        final_binary = (sample > 0).astype(np.uint8)

    np.save("outputs/keys/gan_generated_key.npy", final_binary)

    with open("outputs/keys/gan_generated_key.bin", "wb") as f:
        f.write(np.packbits(final_binary).tobytes())

    with open("outputs/keys/key.hex", "w") as f:
        f.write("".join(map(str, final_binary)))

    with open("outputs/keys/key.b64", "w") as f:
        f.write(base64.b64encode(np.packbits(final_binary)).decode())

    img = qrcode.make(base64.b64encode(np.packbits(final_binary)).decode())
    img.get_image().save("outputs/keys/key_qr.png")

    with open("outputs/logs/training_metrics.csv", "w") as f:
        f.write("Epoch,Entropy,G_Loss,D_Loss\n")
        for i in range(len(entropy_log)):
            f.write(f"{i * 100},{entropy_log[i]},{g_loss_log[i]},{d_loss_log[i]}\n")

    # === EXPAND TO 1M BIT FILE FOR ENT TESTING ===
    expanded = expand_key_sha256(final_binary, 1_000_000)
    with open("outputs/keys/gan_expanded_1mbit.bin", "wb") as f:
        f.write(np.packbits(expanded).tobytes())

    print("[✓] All outputs saved to /outputs/")
    print("[✓] 1Mbit expanded key saved to outputs/keys/gan_expanded_1mbit.bin")


# === WINDOWS MULTIPROCESSING GUARD ===
if __name__ == "__main__":
    freeze_support()
    main()
