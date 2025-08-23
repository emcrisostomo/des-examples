import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

def simulate_chunk(chunk_size, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1, 3, chunk_size)
    Y = rng.uniform(2, 6, chunk_size)
    Z = rng.uniform(1, 8, chunk_size)
    T = rng.uniform(2, 5, chunk_size)
    U = rng.uniform(4, 6, chunk_size)
    V = rng.uniform(3, 6, chunk_size)
    return X + Y + Z + T + U + V

def process_chunk(seed, chunk_size, bins):
    S = simulate_chunk(chunk_size, seed)
    h, _ = np.histogram(S, bins=bins)
    return h

if __name__ == "__main__":
    n = 1_000_000_000
    num_workers = cpu_count()
    chunk_size = 10_000_000
    num_chunks = n // chunk_size

    bins = np.linspace(10, 40, 201)
    hist = np.zeros(len(bins) - 1)

    # Use partial to freeze chunk_size and bins
    worker = partial(process_chunk, chunk_size=chunk_size, bins=bins)

    with Pool(num_workers) as pool:
        for h in pool.imap_unordered(worker, range(num_chunks)):
            hist += h

    hist = hist / hist.sum() / np.diff(bins)

    plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge', alpha=0.7, color='skyblue')
    plt.title('Monte Carlo Simulation of S = X + Y + Z + T + U + V')
    plt.xlabel('S')
    plt.ylabel('Density')
    plt.show()