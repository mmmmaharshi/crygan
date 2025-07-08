import base64
import os
import random
from collections import Counter
from hashlib import sha256
from math import log2
from multiprocessing import cpu_count, freeze_support

import numpy as np
import qrcode
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import torch.optim as optim
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.hmac import HMAC
from tqdm import tqdm

# === CONFIG ===
latent_dim = 100
input_bits = 256
epochs = 3000
batch_size = 32
lr = 0.0001
lambda_gp = 10
critic_iters = 5
entropy_sources = ["rain.wav", "traffic.wav"]
seed = 42
checkpoint_path = "outputs/checkpoints"
output_path = "outputs/keys"

torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(1)


# === REPRODUCIBILITY ===
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# === ENTROPY EXTRACTION ===
def extract_bits_hashed(args):
    data, i, segment_len_samples = args
    start = i * segment_len_samples
    end = start + segment_len_samples
    segment = data[start:end]
    h = sha256(segment.tobytes()).digest()
    return np.unpackbits(np.frombuffer(h, dtype=np.uint8))


def _extract_file_entropy(args):
    path, count_per_file = args
    sr, data = wav.read(path)
    if data.dtype == np.int16:
        data = data / 32768.0
    if data.ndim > 1:
        data = data[:, 0]
    segment_len_samples = 1024
    args_list = [(data, i, segment_len_samples) for i in range(count_per_file)]
    file_segments = []
    for r in tqdm(args_list, leave=False):
        bits = extract_bits_hashed(r)
        if bits is not None:
            file_segments.append(bits)
    return os.path.basename(path), file_segments



def load_entropy(paths, count_per_file=1024):
    from multiprocessing import Pool as OuterPool
    from multiprocessing import cpu_count as mp_cpu_count

    with OuterPool() as pool:
        results = pool.map(
            _extract_file_entropy, [(path, count_per_file) for path in paths]
        )
    all_segments = []
    for _, file_segments in results:
        all_segments.extend(file_segments)
    return np.array(all_segments)

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


# === UTILITIES ===
def compute_gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, device=device).expand_as(real)
    interpolates = alpha * real + (1 - alpha) * fake
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
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def shannon_entropy(bits):
    c = Counter(bits)
    total = len(bits)
    return -sum((f / total) * log2(f / total) for f in c.values())


def expand_key_hkdf(seed_bits, target_bits=1_000_000, salt=None):
    seed_bytes = np.packbits(seed_bits).tobytes()
    if salt is None:
        salt = os.urandom(16)

    # === HKDF-Extract ===
    h = HMAC(salt, hashes.SHA256(), backend=default_backend())
    h.update(seed_bytes)
    prk = h.finalize()

    # === HKDF-Expand ===
    output = bytearray()
    prev = b""
    counter = 1
    info = b"gan-key-expansion"
    while len(output) < (target_bits + 7) // 8:
        h = HMAC(prk, hashes.SHA256(), backend=default_backend())
        h.update(prev + info + counter.to_bytes(4, "big"))
        prev = h.finalize()
        output.extend(prev)
        counter += 1

    final_bytes = bytes(output[: (target_bits + 7) // 8])
    expanded_bits = np.unpackbits(np.frombuffer(final_bytes, dtype=np.uint8))
    return expanded_bits[:target_bits], salt


def save_checkpoint(G, D, epoch):
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(
        {"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch},
        os.path.join(checkpoint_path, f"ckpt_epoch_{epoch}.pt"),
    )


# === MAIN ===
def main():
    set_seed(seed)
    torch.set_num_threads(cpu_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    real_keys = load_entropy(entropy_sources, count_per_file=1024)

    G, D = Generator().to(device), Discriminator().to(device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    real_tensor = torch.tensor(real_keys.astype(np.float32), device=device) * 2 - 1
    final_binary = None
    entropy_log, g_loss_log, d_loss_log = [], [], []
    D_loss = torch.tensor(0.0)
    G_loss = torch.tensor(0.0)

    progress = tqdm(range(epochs), desc="Epochs")
    for epoch in progress:
        for _ in range(critic_iters):
            idx = np.random.randint(0, real_tensor.shape[0], batch_size)
            real_batch = real_tensor[idx]
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_batch = G(z).detach()
            gp = compute_gradient_penalty(D, real_batch, fake_batch, device)
            loss_D = -D(real_batch).mean() + D(fake_batch).mean() + lambda_gp * gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_batch = G(z)
        loss_G = -D(fake_batch).mean()
        opt_G.zero_grad()
        loss_G.backward()
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
            progress.set_description(f"H={ent:.4f}")

    if final_binary is None:
        z = torch.randn(1, latent_dim, device=device)
        with torch.no_grad():
            final_binary = (G(z).squeeze().cpu().numpy() > 0).astype(np.uint8)

    print(f"[📦] Final entropy: {shannon_entropy(final_binary):.4f}")

    np.save(os.path.join(output_path, "gan_generated_key.npy"), final_binary)
    with open(os.path.join(output_path, "gan_generated_key.bin"), "wb") as f:
        f.write(np.packbits(final_binary).tobytes())
    with open(os.path.join(output_path, "key.hex"), "w") as f:
        f.write("".join(map(str, final_binary)))
    with open(os.path.join(output_path, "key.b64"), "w") as f:
        f.write(base64.b64encode(np.packbits(final_binary)).decode())

    qrcode.make(base64.b64encode(np.packbits(final_binary)).decode()).get_image().save(
        os.path.join(output_path, "key_qr.png")
    )

    with open("outputs/logs/training_metrics.csv", "w") as f:
        f.write("Epoch,Entropy,G_Loss,D_Loss\n")
        for i in range(len(entropy_log)):
            f.write(f"{i * 100},{entropy_log[i]},{g_loss_log[i]},{d_loss_log[i]}\n")

    expanded, salt = expand_key_hkdf(final_binary, 1_000_000)
    with open(os.path.join(output_path, "gan_expanded_1mbit.bin"), "wb") as f:
        f.write(np.packbits(expanded).tobytes())
    with open(os.path.join(output_path, "salt.bin"), "wb") as f:
        f.write(salt)

    print("[✅] 1Mbit expanded key saved.")
    print("[✓] All outputs saved to /outputs/")

if __name__ == "__main__":
    freeze_support()
    main()
