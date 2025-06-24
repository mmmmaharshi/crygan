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
lr = 0.0001
wav_path = "rain.wav"
lambda_gp = 10
critic_iters = 5


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


def load_entropy_parallel(path, count=2048, bit_length=256, threshold=0.0005):
    print(f"[🎵] Extracting entropy from: {path}")
    sr, data = wav.read(path)
    if data.dtype == np.int16:
        data = data / 32768.0
    if data.ndim > 1:
        data = data[:, 0]

    args_list = [(data, i, bit_length, threshold) for i in range(count)]
    results = []
    with Pool() as pool:
        for r in tqdm(
            pool.imap_unordered(extract_bits, args_list),
            total=count,
            desc="[⚙️ ] Entropy Segments",
        ):
            if r is not None:
                results.append(r)
    return np.array(results)


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
        )

    def forward(self, x):
        return self.net(x)


# === GRADIENT PENALTY ===
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    alpha = alpha.expand_as(real_samples)

    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(real_samples.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[🧠] Device: {device}")
    print(f"[🔧] CPU Cores Used: {cpu_count()}")

    os.makedirs("outputs/keys", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    real_keys = load_entropy_parallel(wav_path)
    print(f"[✓] Loaded {real_keys.shape[0]} samples × {real_keys.shape[1]} bits")
    print("[🚀] Starting WGAN-GP Training...")

    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

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

    D_loss = torch.tensor(0.0)
    G_loss = torch.tensor(0.0)

    progress = tqdm(range(epochs), desc="Epochs")
    for epoch in progress:
        for _ in range(critic_iters):
            idx = np.random.randint(0, real_tensor.shape[0], batch_size)
            real_batch = real_tensor[idx]

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_batch = G(z).detach()

            d_real = D(real_batch)
            d_fake = D(fake_batch)

            gp = compute_gradient_penalty(D, real_batch, fake_batch, device)
            D_loss = -d_real.mean() + d_fake.mean() + lambda_gp * gp

            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_batch = G(z)
        G_loss = -D(fake_batch).mean()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if epoch % 100 == 0:
            z = torch.randn(1, latent_dim, device=device)
            with torch.no_grad():
                sample = G(z).squeeze().cpu().numpy()
            binary = (sample > 0).astype(np.uint8)
            ent = shannon_entropy(binary)
            entropy_log.append(ent)
            g_loss_log.append(G_loss.item())
            d_loss_log.append(D_loss.item())
            final_binary = binary
            progress.set_description(
                f"Epochs [D={D_loss.item():.3f}, G={G_loss.item():.3f}, H={ent:.4f}]"
            )

    if final_binary is None:
        z = torch.randn(1, latent_dim, device=device)
        with torch.no_grad():
            sample = G(z).squeeze().cpu().numpy()
        final_binary = (sample > 0).astype(np.uint8)

    print(f"[📦] Final binary entropy: {shannon_entropy(final_binary):.4f}")

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

    print("[📡] Expanding key to 1M bits via SHA-256...")
    expanded = expand_key_sha256(final_binary, 1_000_000)
    with open("outputs/keys/gan_expanded_1mbit.bin", "wb") as f:
        f.write(np.packbits(expanded).tobytes())

    print("[✅] 1Mbit expanded key saved to outputs/keys/gan_expanded_1mbit.bin")
    print("[✓] All outputs saved to /outputs/")


# === WINDOWS MULTIPROCESSING GUARD ===
if __name__ == "__main__":
    freeze_support()
    main()
