#!/usr/bin/env python3
"""Analyze existing Ising CSV files; never runs a simulation."""
import argparse
import csv
import glob
import json
from pathlib import Path
import sys

import numpy as np

OBSERVABLES = ('abs_m', 'm', 'e_over_J', 'chi_J', 'chi_abs_J', 'cv_over_kB')


def read_samples(path, discard=0):
    """Discard a number of saved rows, in addition to simulation equilibration."""
    with open(path) as f:
        line = f.readline()
        if not line.startswith('# '):
            raise ValueError(f'{path}: missing JSON metadata')
        meta = json.loads(line[2:])
        data = np.atleast_1d(np.genfromtxt(f, delimiter=',', names=True))
    if meta.get('schema_version') != 1 or meta.get('boundary') != 'periodic':
        raise ValueError(f'{path}: unsupported sample format')
    if meta['N'] != meta['L'] ** 2 or not np.isfinite(meta['temperature']) or meta['temperature'] <= 0:
        raise ValueError(f'{path}: invalid lattice size or temperature')
    if not {'sweep', 'm', 'e_over_J'}.issubset(data.dtype.names or ()):
        raise ValueError(f'{path}: missing sample columns')
    if discard < 0:
        raise ValueError('discard must be nonnegative')
    data = data[discard:]
    if len(data) < 2:
        raise ValueError(f'{path}: need at least two retained samples')
    for key in ('sweep', 'm', 'e_over_J'):
        if not np.all(np.isfinite(data[key])):
            raise ValueError(f'{path}: nonfinite {key}')
    if not np.all(np.diff(data['sweep']) == meta['every']):
        raise ValueError(f'{path}: samples are not regularly spaced')
    if np.any(np.abs(data['m']) > 1) or np.any(np.abs(data['e_over_J']) > 2):
        raise ValueError(f'{path}: observables outside physical bounds')
    return meta, data


def moments(data):
    m, e = data['m'], data['e_over_J']
    return np.column_stack((np.abs(m), m, e, m*m, e*e))


def observables(q, N, T):
    a, m, e, m2, e2 = np.moveaxis(q, -1, 0)
    return np.stack((a, m, e, N * np.maximum(0, m2-m*m) / T,
                     N * np.maximum(0, m2-a*a) / T,
                     N * np.maximum(0, e2-e*e) / T**2), axis=-1)


def summarize(path, discard=0, block_size=100):
    if block_size < 1:
        raise ValueError('block size must be positive')
    meta, data = read_samples(path, discard)
    q = moments(data)
    values = observables(q.mean(axis=0), meta['N'], meta['temperature'])
    # Equal contiguous blocks; only the error estimate drops an incomplete tail.
    blocks = len(q) // block_size
    errors = np.full(len(OBSERVABLES), np.nan)
    used = blocks * block_size
    if blocks >= 2:
        means = q[:used].reshape(blocks, block_size, 5).mean(axis=1)
        leave_one_out = (means.sum(axis=0) - means) / (blocks - 1)
        estimates = observables(leave_one_out, meta['N'], meta['temperature'])
        errors = np.sqrt((blocks-1)/blocks * ((estimates-estimates.mean(axis=0))**2).sum(axis=0))
    if blocks < 10:
        print(f'Warning: {path}: only {blocks} complete blocks; errors are preliminary'
              ' (NaN if fewer than two).', file=sys.stderr)
    if not (np.any(data['m'] > 0) and np.any(data['m'] < 0)):
        print(f'Warning: {path}: both magnetization signs were not sampled.', file=sys.stderr)
    row = dict(source=str(path), L=meta['L'], T=meta['temperature'], seed=meta['seed'],
               start=meta['start'], n_samples=len(data), discard_samples=discard,
               every=meta['every'], block_size=block_size, n_blocks=blocks,
               error_samples=used if blocks >= 2 else 0)
    for name, value, error in zip(OBSERVABLES, values, errors):
        row[name], row[name + '_err'] = float(value), float(error)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+', help='sample files or quoted glob patterns')
    p.add_argument('--discard', type=int, default=0, help='additional saved rows to discard per file')
    p.add_argument('--block-size', type=int, default=100, help='saved samples per error block')
    p.add_argument('--output', type=Path, default=Path('results/summary.csv'))
    a = p.parse_args()
    paths = set()
    for pattern in a.inputs:
        matches = glob.glob(pattern)
        if not matches:
            p.error(f'no files match {pattern!r}; run simulations first')
        paths.update(str(Path(x).resolve()) for x in matches)
    if str(a.output.resolve()) in paths:
        p.error('output must not be an input file')
    try:
        rows = [summarize(path, a.discard, a.block_size) for path in sorted(paths)]
    except (ValueError, KeyError, OSError) as exc:
        p.error(str(exc))
    rows.sort(key=lambda r: (r['L'], r['T'], r['seed'], r['source']))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} per-run summaries to {a.output}')
    print('Check error stability with several block sizes; blocks must exceed correlation times.')


if __name__ == '__main__':
    main()
