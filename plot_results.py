#!/usr/bin/env python3
"""Plot an existing summary.csv; each run remains a separate point."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('results'))
    a = p.parse_args()
    with a.summary.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        p.error('summary contains no runs')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panels = [('abs_m', r'$\langle |m|\rangle$'), ('chi_J', r'$J\chi$ (signed $m$)'),
              ('cv_over_kB', r'$c_V/k_B$'), ('chi_abs_J', r'$J\chi_{|m|}$ (alternative)')]
    colors = ['black', '#1f77b4', '#E69F00', '#009E73']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    sizes = sorted({int(r['L']) for r in rows})
    for ax, (key, label) in zip(axes.flat, panels):
        for i, L in enumerate(sizes):
            group = sorted((r for r in rows if int(r['L']) == L), key=lambda r: float(r['T']))
            T = [float(r['T']) for r in group]
            y = [float(r[key]) for r in group]
            err = np.array([float(r[key + '_err']) for r in group])
            color, marker = colors[i % len(colors)], markers[i % len(markers)]
            ax.plot(T, y, linestyle='none', marker=marker, color=color, label=f'L = {L}')
            valid = np.isfinite(err)
            ax.errorbar(np.array(T)[valid], np.array(y)[valid], yerr=err[valid],
                        fmt='none', ecolor=color, capsize=2)
        ax.axvline(2/np.log(1+np.sqrt(2)), color='0.4', ls='--', label='Onsager')
        ax.axvline(4, color='0.6', ls=':', label='Mean field')
        ax.set(xlabel=r'$T^*=k_BT/J$', ylabel=label)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(a.output_dir / f'observables.{suffix}', dpi=180)
    plt.close(fig)
    print(f'Saved observables.png and observables.pdf in {a.output_dir}')


if __name__ == '__main__':
    main()
