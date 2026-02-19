#!/usr/bin/env python3
"""
Monte Carlo simulation for shard overlap.

Demonstrates that for two random shards of size k sampled from n cells
without replacement, the expected overlap is k^2 / n.
"""

from __future__ import annotations

import argparse
import random
from statistics import mean


def trial_overlap(n: int, k: int, rng: random.Random) -> int:
    shard_a = set(rng.sample(range(n), k))
    shard_b = set(rng.sample(range(n), k))
    return len(shard_a & shard_b)


def simulate(n: int, k: int, trials: int, seed: int | None) -> float:
    rng = random.Random(seed)
    overlaps = [trial_overlap(n, k, rng) for _ in range(trials)]
    return mean(overlaps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate overlap between two random shards of size k over n cells "
            "and compare with expected value k^2/n."
        )
    )
    parser.add_argument("--n", type=int, default=1000, help="Total number of cells")
    parser.add_argument("--k", type=int, default=50, help="Shard size")
    parser.add_argument(
        "--trials", type=int, default=100_000, help="Number of Monte Carlo trials"
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("n must be positive")
    if args.k < 0:
        raise ValueError("k must be non-negative")
    if args.k > args.n:
        raise ValueError("k cannot be greater than n")
    if args.trials <= 0:
        raise ValueError("trials must be positive")

    empirical = simulate(args.n, args.k, args.trials, args.seed)
    theoretical = (args.k * args.k) / args.n
    abs_error = abs(empirical - theoretical)
    rel_error = abs_error / theoretical if theoretical != 0 else 0.0

    print(f"n={args.n}, k={args.k}, trials={args.trials}, seed={args.seed}")
    print(f"Empirical average overlap: {empirical:.6f}")
    print(f"Theoretical expectation k^2/n: {theoretical:.6f}")
    print(f"Absolute error: {abs_error:.6f}")
    print(f"Relative error: {rel_error:.4%}")


if __name__ == "__main__":
    main()
