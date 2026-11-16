#!/usr/bin/env python3
"""Inspect one existing run for drift and magnetization reversals."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze import read_samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('samples', type=Path)
    p.add_argument('--discard', type=int, default=0)
    p.add_argument('--output', type=Path, default=Path('results/timeseries.png'))
    a = p.parse_args()
    if a.output.resolve() == a.samples.resolve():
        p.error('output must not overwrite the input')
    try:
        meta, data = read_samples(a.samples, a.discard)
    except (ValueError, KeyError, OSError) as exc:
        p.error(str(exc))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for ax, key, label in zip(axes, ['m', 'e_over_J'], [r'$m$', r'$E/(NJ)$']):
        y = data[key]
        ax.plot(data['sweep'], y, color='black', lw=0.5, alpha=0.6, label='Samples')
        ax.plot(data['sweep'], np.cumsum(y)/np.arange(1, len(y)+1),
                color='#1f77b4', lw=1.4, label='Running mean')
        ax.set_ylabel(label)
    axes[0].legend()
    axes[0].set_title(f"L = {meta['L']}, T* = {meta['temperature']}, seed = {meta['seed']}")
    axes[1].set_xlabel('Production sweep')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=180)
    plt.close(fig)
    print(f'Saved {a.output}')


if __name__ == '__main__':
    main()
