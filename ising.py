#!/usr/bin/env python3
"""Square-lattice Ising Metropolis sampling; temperature = k_B T/J (J > 0)."""

import argparse
import csv
from contextlib import ExitStack
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


def simulate(L, temperature, burn, sweeps, every, seed, start, output,
             snapshot_every=0, snapshot_output=None):
    """Run one independently initialized chain and stream production samples."""
    if snapshot_every < 0:
        raise ValueError("snapshot_every must be nonnegative")
    if snapshot_output is not None and snapshot_every == 0:
        raise ValueError("snapshot_output requires snapshot_every > 0")
    output = Path(output)
    if snapshot_every:
        import numpy as np  # Optional: normal simulations use only the standard library.
        snapshot_output = (Path(snapshot_output) if snapshot_output is not None
                           else output.with_suffix(".snapshots.npz"))
        if snapshot_output.resolve() == output.resolve():
            raise ValueError("samples and snapshots must use different paths")
        if snapshot_output.exists():
            raise FileExistsError(f"{snapshot_output} already exists")
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    frames, frame_sweeps = [], []

    def capture(step):
        if snapshot_every:
            frames.append(np.array(spins, dtype=np.int8))
            frame_sweeps.append(step)

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
    with ExitStack() as stack:
        if snapshot_every:
            snapshot_output.parent.mkdir(parents=True, exist_ok=True)
            snapshot_handle = stack.enter_context(snapshot_output.open("xb"))
        handle = stack.enter_context(output.open("x", newline=""))
        handle.write("# " + json.dumps(metadata, sort_keys=True) + "\n")
        writer = csv.writer(handle)
        writer.writerow(["sweep", "E_over_J", "M", "e_over_J", "m"])
        capture(0)
        for step in range(1, burn + 1):
            E, M = sweep(spins, E, M, weights, rng)
            if snapshot_every and (step % snapshot_every == 0 or step == burn):
                capture(step)
        for step in range(1, sweeps + 1):
            E, M = sweep(spins, E, M, weights, rng)
            total_step = burn + step
            if snapshot_every and (total_step % snapshot_every == 0 or step == sweeps):
                capture(total_step)
            if step % every == 0:
                writer.writerow([step, E, M, E / (L * L), M / (L * L)])
        if snapshot_every:
            np.savez_compressed(snapshot_handle, spins=np.stack(frames),
                                sweep=np.array(frame_sweeps, dtype=np.int64),
                                metadata=json.dumps(dict(metadata, snapshot_every=snapshot_every,
                                                         snapshot_schema_version=1)))
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
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="save full spins every n total sweeps; 0 disables snapshots")
    parser.add_argument("--snapshot-output", type=Path,
                        help="default: sample filename with .snapshots.npz suffix")
    args = parser.parse_args()
    if args.L < 3:
        parser.error("L must be >= 3 to have four distinct periodic neighbours")
    if not math.isfinite(args.T) or args.T <= 0:
        parser.error("T must be finite and positive")
    if args.burn < 0 or args.sweeps < 1 or not 1 <= args.every <= args.sweeps:
        parser.error("require burn >= 0 and 1 <= every <= sweeps")
    if args.snapshot_every < 0 or (args.snapshot_output and not args.snapshot_every):
        parser.error("snapshot-every must be >= 0; snapshot-output requires it to be > 0")
    try:
        simulate(args.L, args.T, args.burn, args.sweeps, args.every,
                 args.seed, args.start, args.output, args.snapshot_every, args.snapshot_output)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    except ImportError:
        parser.error("snapshots require NumPy: install requirements-analysis.txt")
    print(f"Saved {args.sweeps // args.every} samples to {args.output}")
    if args.snapshot_every:
        print(f"Saved spin snapshots to {args.snapshot_output or args.output.with_suffix('.snapshots.npz')}")


if __name__ == "__main__":
    main()
