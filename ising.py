#!/usr/bin/env python3
"""Square-lattice Ising Metropolis sampling; temperature = k_B T/J (J > 0)."""

import argparse
import csv
import json
import math
import random
from pathlib import Path


def energy(spins):
    """Total E/J, counting each periodic nearest-neighbour bond once."""
    L = len(spins)
    return -sum(spins[i][j] * (spins[(i + 1) % L][j] + spins[i][(j + 1) % L])
                for i in range(L) for j in range(L))


def delta_energy(spins, i, j):
    """Energy change in units of J for flipping one spin."""
    L = len(spins)
    neighbours = (spins[(i - 1) % L][j] + spins[(i + 1) % L][j]
                  + spins[i][(j - 1) % L] + spins[i][(j + 1) % L])
    return 2 * spins[i][j] * neighbours


def sweep(spins, E, M, weights, rng):
    """L^2 random-site proposals with replacement; return updated E/J and M."""
    L = len(spins)
    for _ in range(L * L):
        i, j = rng.randrange(L), rng.randrange(L)
        dE = delta_energy(spins, i, j)
        if dE <= 0 or rng.random() < weights[dE]:
            spins[i][j] *= -1
            E += dE
            M += 2 * spins[i][j]
    return E, M


def simulate(L, temperature, burn, sweeps, every, seed, start, output):
    """Run one independently initialized chain and stream production samples."""
    rng = random.Random(seed)
    spins = [[rng.choice((-1, 1)) if start == "random" else 1
              for _ in range(L)] for _ in range(L)]
    E, M = energy(spins), sum(map(sum, spins))
    weights = {dE: math.exp(-dE / temperature) for dE in (4, 8)}
    metadata = dict(L=L, N=L * L, temperature=temperature,
                    temperature_units="k_B T/J", energy_units="J",
                    burn=burn, sweeps=sweeps, every=every, seed=seed,
                    start=start, boundary="periodic", schema_version=1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of a previous run.
    with output.open("x", newline="") as handle:
        handle.write("# " + json.dumps(metadata, sort_keys=True) + "\n")
        writer = csv.writer(handle)
        writer.writerow(["sweep", "E_over_J", "M", "e_over_J", "m"])
        for _ in range(burn):
            E, M = sweep(spins, E, M, weights, rng)
        for step in range(1, sweeps + 1):
            E, M = sweep(spins, E, M, weights, rng)
            if step % every == 0:
                writer.writerow([step, E, M, E / (L * L), M / (L * L)])
    return spins, E, M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--T", type=float, default=2.3, help="reduced temperature k_B T/J")
    parser.add_argument("--burn", type=int, default=5000, help="equilibration sweeps discarded")
    parser.add_argument("--sweeps", type=int, default=20000, help="production sweeps")
    parser.add_argument("--every", type=int, default=1, help="save every n production sweeps")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--start", choices=["random", "up"], default="random")
    parser.add_argument("--output", type=Path, default=Path("data/samples.csv"))
    args = parser.parse_args()
    if args.L < 3:
        parser.error("L must be >= 3 to have four distinct periodic neighbours")
    if not math.isfinite(args.T) or args.T <= 0:
        parser.error("T must be finite and positive")
    if args.burn < 0 or args.sweeps < 1 or not 1 <= args.every <= args.sweeps:
        parser.error("require burn >= 0 and 1 <= every <= sweeps")
    try:
        simulate(args.L, args.T, args.burn, args.sweeps, args.every,
                 args.seed, args.start, args.output)
    except FileExistsError:
        parser.error(f"{args.output} already exists; choose another output filename")
    print(f"Saved {args.sweeps // args.every} samples to {args.output}")


if __name__ == "__main__":
    main()
