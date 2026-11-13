#!/usr/bin/env python3
"""Unit-vector Heisenberg Monte Carlo on periodic square/simple-cubic lattices.

H/J = -sum_<ij> e_i.e_j - kappa*sum_<ij> (e_i.e_j)^2,
kappa=K/J and T*=kB*T/J, with J>0. Each bond is counted once.
"""
import argparse
from contextlib import ExitStack
import csv
import json
import math
from pathlib import Path
import numpy as np

COLUMNS = ['sweep', 'e_over_J', 'mx', 'my', 'mz', 'm_abs',
           'C1', 'B1', 'nematic', 'acceptance']


def neighbours(L, dim):
    """All 2*dim neighbours, positive/negative along each axis."""
    grid = np.arange(L**dim).reshape((L,)*dim)
    return np.column_stack([np.roll(grid, shift, axis=axis).ravel()
                            for axis in range(dim) for shift in (-1, 1)])


def random_vectors(rng, n):
    vectors = rng.normal(size=(n, 3))
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def pair_sums(spins, nb):
    # Each undirected bond is counted once, through positive-axis neighbours.
    dots = np.einsum('ij,ikj->ik', spins, spins[nb[:, ::2]])
    return float(dots.sum()), float((dots*dots).sum())


def total_energy(spins, nb, kappa):
    bilinear, biquadratic = pair_sums(spins, nb)
    return -bilinear-kappa*biquadratic


def delta_energy(spins, nb, site, proposed, kappa):
    near = spins[nb[site]]
    old = near @ spins[site]
    new = near @ proposed
    return float(-(new-old).sum()-kappa*(new*new-old*old).sum())


def sweep(spins, nb, T, kappa, rng, proposal='mixed', step_size=.5):
    """N sequential random-site attempts; no adaptive proposal tuning."""
    N = len(spins)
    sites = rng.integers(N, size=N)
    noises = rng.normal(size=(N, 3))
    modes, uniforms = rng.random(N), rng.random(N)
    accepted = 0
    for site, noise, mode, u in zip(sites, noises, modes, uniforms):
        if proposal == 'uniform' or mode < .1:
            trial = noise / np.linalg.norm(noise)
        elif mode < .2:
            trial = -spins[site]  # Useful for a strong collinearity preference.
        else:
            trial = spins[site] + step_size*noise
            trial /= np.linalg.norm(trial)
        dE = delta_energy(spins, nb, site, trial, kappa)
        if dE <= 0 or u < math.exp(-dE/T):
            spins[site] = trial
            accepted += 1
    return accepted/N


def measure(spins, nb, kappa):
    N = len(spins)
    bonds = N * nb.shape[1]//2
    c, b = pair_sums(spins, nb)
    m = spins.mean(axis=0)
    # Q=0 for an isotropic ensemble; largest eigenvalue=1 for collinearity.
    Q = 1.5*(spins.T @ spins)/N - .5*np.eye(3)
    return [(-c-kappa*b)/N, *m, np.linalg.norm(m), c/bonds, b/bonds,
            np.linalg.eigvalsh(Q)[-1]]


def simulate(L, dim, T, kappa, burn, sweeps, every, seed, start, output,
             proposal='mixed', step_size=.5, snapshot_every=0, snapshot_output=None):
    if L < 3 or dim not in (2, 3):
        raise ValueError('require L>=3 and dim=2 or 3')
    if not np.isfinite(T) or T <= 0 or not np.isfinite(kappa):
        raise ValueError('T must be positive and finite; K must be finite')
    if burn < 0 or sweeps < 1 or not 1 <= every <= sweeps:
        raise ValueError('require burn>=0 and 1<=every<=sweeps')
    if start not in ('random', 'up') or proposal not in ('mixed', 'uniform'):
        raise ValueError('unknown start or proposal')
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError('step-size must be positive and finite')
    if snapshot_every < 0 or (snapshot_output is not None and not snapshot_every):
        raise ValueError('snapshot-output requires a positive snapshot interval')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} already exists')
    if snapshot_every:
        snapshot_output = Path(snapshot_output) if snapshot_output else output.with_suffix('.vectors.npz')
        if snapshot_output.resolve() == output.resolve():
            raise ValueError('samples and snapshots must use different filenames')
        if snapshot_output.exists():
            raise FileExistsError(f'{snapshot_output} already exists')
    rng = np.random.default_rng(seed)
    N = L**dim
    spins = random_vectors(rng, N) if start == 'random' else np.tile([0.,0.,1.], (N,1))
    nb = neighbours(L, dim)
    meta = dict(model='classical_heisenberg', schema_version=1, L=L, dim=dim,
                N=N, temperature=T, kappa=kappa, temperature_units='kB T/J',
                energy_units='J', boundary='periodic', start=start, seed=seed,
                burn=burn, sweeps=sweeps, every=every, proposal=proposal,
                step_size=step_size, snapshot_every=snapshot_every,
                numpy_version=np.__version__)
    frames, frame_steps = [], []
    def capture(step):
        if snapshot_every:
            frames.append(spins.reshape((L,)*dim+(3,)).copy())
            frame_steps.append(step)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if snapshot_every:
            snapshot_output.parent.mkdir(parents=True, exist_ok=True)
            snapshots = stack.enter_context(snapshot_output.open('xb'))
        handle = stack.enter_context(output.open('x', newline=''))
        handle.write('# '+json.dumps(meta, sort_keys=True)+'\n')
        writer = csv.writer(handle); writer.writerow(COLUMNS)
        capture(0)
        for total_step in range(1, burn+sweeps+1):
            acceptance = sweep(spins, nb, T, kappa, rng, proposal, step_size)
            if snapshot_every and (total_step % snapshot_every == 0 or total_step in (burn,burn+sweeps)):
                capture(total_step)
            step = total_step-burn
            if step > 0 and step % every == 0:
                writer.writerow([step, *measure(spins, nb, kappa), acceptance])
        if snapshot_every:
            np.savez_compressed(snapshots, spins=np.stack(frames), sweep=frame_steps,
                                metadata=json.dumps(dict(meta, snapshot_schema_version=1)))
    return spins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--L', type=int, default=4)
    p.add_argument('--dim', type=int, choices=[2,3], default=3)
    p.add_argument('--T', type=float, default=1.45)
    p.add_argument('--K', type=float, default=0, help='kappa=K/J; minus sign in Hamiltonian')
    p.add_argument('--burn', type=int, default=1000)
    p.add_argument('--sweeps', type=int, default=5000)
    p.add_argument('--every', type=int, default=1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--start', choices=['random','up'], default='random')
    p.add_argument('--proposal', choices=['mixed','uniform'], default='mixed')
    p.add_argument('--step-size', type=float, default=.5)
    p.add_argument('--snapshot-every', type=int, default=0)
    p.add_argument('--snapshot-output', type=Path)
    p.add_argument('--output', type=Path, default=Path('data/heisenberg/samples.csv'))
    a = p.parse_args()
    try:
        simulate(a.L,a.dim,a.T,a.K,a.burn,a.sweeps,a.every,a.seed,a.start,a.output,
                 a.proposal,a.step_size,a.snapshot_every,a.snapshot_output)
    except (ValueError, FileExistsError) as exc:
        p.error(str(exc))
    print(f'Saved {a.sweeps//a.every} samples for {a.L**a.dim} vector spins to {a.output}')


if __name__ == '__main__':
    main()
