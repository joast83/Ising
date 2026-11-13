#!/usr/bin/env python3
"""Print or execute a reproducible set of extra convergence runs (serially)."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys


def plan(stage):
    if stage == 'quick':
        return [(64, 2.3, start, seed, 10000, 50000)
                for start in ('random', 'up') for seed in (2, 3)]
    if stage == 'extended':
        return [(L, T, start, seed, 10000, 50000)
                for L in (32, 64) for T in (2.25, 2.3)
                for start in ('random', 'up') for seed in (2, 3, 4)]
    return [(64, 2.3, start, seed, 50000, 200000)
            for start in ('random', 'up') for seed in (5, 6)]


def complete(path, settings):
    """Only skip an existing run if metadata and every expected row are present."""
    L, T, start, seed, burn, sweeps = settings
    try:
        with path.open() as f:
            first = f.readline()
            if not first.startswith('# '):
                return False
            meta = json.loads(first[2:])
            expected = dict(L=L, temperature=T, start=start, seed=seed,
                            burn=burn, sweeps=sweeps, every=1)
            if any(meta.get(k) != v for k, v in expected.items()):
                return False
            reader = csv.DictReader(f)
            count = 0
            for count, row in enumerate(reader, 1):
                if int(row['sweep']) != count:
                    return False
                E, M = int(row['E_over_J']), int(row['M'])
                if abs(E) > 2*L*L or abs(M) > L*L:
                    return False
                if abs(float(row['e_over_J'])-E/(L*L)) > 1e-12:
                    return False
                if abs(float(row['m'])-M/(L*L)) > 1e-12:
                    return False
            return count == sweeps
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['quick', 'extended', 'long'])
    p.add_argument('--data-dir', type=Path, default=Path('data/convergence'))
    p.add_argument('--run', action='store_true', help='execute; otherwise only print commands')
    a = p.parse_args()
    script = Path(__file__).resolve().with_name('ising.py')
    jobs = plan(a.stage)
    print(f'{len(jobs)} planned runs. Existing complete matching runs are skipped.', flush=True)
    for settings in jobs:
        L, T, start, seed, burn, sweeps = settings
        path = a.data_dir / f'L{L}_T{T}_{start}_seed{seed}_burn{burn}_n{sweeps}.csv'
        if path.exists():
            if complete(path, settings):
                print(f'Skip complete: {path}', flush=True)
                continue
            p.error(f'{path} exists but is incomplete or mismatched. Move it aside before retrying; it will not be overwritten.')
        command = [sys.executable, str(script), '--L', str(L), '--T', str(T),
                   '--start', start, '--seed', str(seed), '--burn', str(burn),
                   '--sweeps', str(sweeps), '--every', '1', '--output', str(path)]
        print(shlex.join(command), flush=True)
        if a.run:
            subprocess.run(command, check=True)
    if not a.run:
        print('Preview only. Add --run to execute these commands.')


if __name__ == '__main__':
    main()
