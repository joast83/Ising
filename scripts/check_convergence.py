#!/usr/bin/env python3
"""Make a convergence report from existing CSVs; never launches simulations."""
import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze import read_samples, moments, observables, OBSERVABLES

KEYS = ['abs_m', 'e_over_J', 'chi_abs_J', 'cv_over_kB']
LABELS = [r'$\langle|m|\rangle$', r'$\langle E/(NJ)\rangle$', r'$J\chi_{|m|}$', r'$c_V/k_B$']


def estimate(q, N, T, block):
    values = observables(q.mean(axis=0), N, T)
    n = len(q)//block
    err = np.full(len(OBSERVABLES), np.nan)
    if n >= 2:
        means = q[:n*block].reshape(n, block, 5).mean(axis=1)
        jack = observables((means.sum(axis=0)-means)/(n-1), N, T)
        err = np.sqrt((n-1)/n * ((jack-jack.mean(axis=0))**2).sum(axis=0))
    return values, err, n


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def entry(meta, source, q, block, discard=0):
    values, errors, n = estimate(q, meta['N'], meta['temperature'], block)
    row = dict(source=str(source), L=meta['L'], T=meta['temperature'],
               start=meta['start'], seed=meta['seed'], burn=meta['burn'],
               sweeps=meta['sweeps'], every=meta['every'], samples=len(q),
               discard=discard, block_size=block, blocks=n)
    for key, val, error in zip(OBSERVABLES, values, errors):
        row[key], row[key+'_err'] = float(val), float(error)
    return row


def diagnostics(path, meta, data, folder, block_sizes, discards, error_block):
    folder.mkdir(parents=True, exist_ok=True)
    q = moments(data)
    blocks = [entry(meta, path, q, b) for b in block_sizes]
    cuts = [entry(meta, path, q[d:], error_block, d) for d in discards if d <= len(q)-2]
    summary = entry(meta, path, q, error_block)
    write_csv(folder/'blocking.csv', blocks)
    write_csv(folder/'discard.csv', cuts)
    # Trace panels use the retained production samples, not the discarded burn-in.
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for ax, values, label in zip(axes, [data['m'], np.abs(data['m']), data['e_over_J']],
                                  ['m', '|m|', 'E/(NJ)']):
        ax.plot(data['sweep'], values, color='0.55', lw=0.5, label='Samples')
        ax.plot(data['sweep'], np.cumsum(values)/np.arange(1, len(values)+1),
                color='#1f77b4', label='Running mean')
        ax.set_ylabel(label)
    axes[0].legend()
    axes[0].set_title(f"L={meta['L']}, T*={meta['temperature']}, {meta['start']}, seed={meta['seed']}")
    axes[-1].set_xlabel('Production sweep')
    fig.savefig(folder/'traces.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for j, (key, label) in enumerate(zip(KEYS, LABELS)):
        ax = axes[0, j]
        good = [r for r in blocks if r['blocks'] >= 20 and np.isfinite(r[key+'_err'])]
        few = [r for r in blocks if 2 <= r['blocks'] < 20 and np.isfinite(r[key+'_err'])]
        ax.plot([r['block_size'] for r in good], [r[key+'_err'] for r in good], 'o-', color='black', label='At least 20 blocks')
        ax.plot([r['block_size'] for r in few], [r[key+'_err'] for r in few], 'x', color='#E69F00', label='Fewer than 20 blocks')
        ax.set(xscale='log', xlabel='Block size (saved samples)', ylabel='Estimated standard error', title=label)
        ax = axes[1, j]
        x, y = [r['discard'] for r in cuts], [r[key] for r in cuts]
        ax.plot(x, y, 'o-', color='#1f77b4')
        good = [r for r in cuts if np.isfinite(r[key+'_err'])]
        ax.errorbar([r['discard'] for r in good], [r[key] for r in good],
                    yerr=[r[key+'_err'] for r in good], fmt='none', ecolor='#1f77b4', capsize=2)
        ax.set(xlabel='Additional saved samples discarded', ylabel=label)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(folder/'sensitivity.png', dpi=130)
    plt.close(fig)
    notes = []
    if summary['blocks'] < 20:
        notes.append(f"Only {summary['blocks']} blocks for comparison error bars; use longer runs or interpret these errors cautiously.")
    if not (np.any(data['m'] > 0) and np.any(data['m'] < 0)):
        notes.append('Both magnetization signs were not observed. Signed-m susceptibility may not represent the full finite-system ensemble; this alone does not invalidate energy or |m|.')
    for key in KEYS:
        usable = [r for r in blocks if r['blocks'] >= 20]
        if len(usable) < 2:
            notes.append(f'{key}: too few block sizes with at least 20 blocks to assess error stability.')
        else:
            old, new = usable[-2][key+'_err'], usable[-1][key+'_err']
            if old == 0 or new == 0:
                notes.append(f'{key}: zero block error encountered; a frozen chain is not evidence of convergence.')
            else:
                change = abs(new/old-1)
                notes.append(f'{key}: error changes by {100*change:.1f}% between the two largest block sizes with at least 20 blocks' +
                             ('; inspect further (over 25% heuristic).' if change > .25 else '; compatible with a plateau over these two points only.'))
    first = observables(q[:len(q)//2].mean(axis=0), meta['N'], meta['temperature'])
    last = observables(q[len(q)//2:].mean(axis=0), meta['N'], meta['temperature'])
    notes.append('Second-half minus first-half estimates: ' + ', '.join(
        f'{k}={last[OBSERVABLES.index(k)]-first[OBSERVABLES.index(k)]:+.4g}' for k in KEYS) +
        '. These are drift diagnostics, not independent significance tests.')
    return summary, notes


def comparisons(rows, output):
    groups = sorted({(r['L'], r['T']) for r in rows})
    links = []
    for L, T in groups:
        group = sorted([r for r in rows if (r['L'], r['T']) == (L, T)],
                       key=lambda r: (r['sweeps'], r['start'], r['seed'], r['source']))
        if len(group) < 2:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(max(9, .7*len(group)), 8), constrained_layout=True)
        labels = [f"{r['start']} s{r['seed']}\nburn {r['burn']}\nn {r['sweeps']}" for r in group]
        for ax, key, label in zip(axes.flat, KEYS, LABELS):
            for i, r in enumerate(group):
                err = r[key+'_err']
                ax.errorbar(i, r[key], yerr=err if np.isfinite(err) else None,
                            fmt='o', color='#1f77b4' if r['start']=='random' else 'black', capsize=3)
            ax.set_xticks(range(len(group)), labels, rotation=70, ha='right', fontsize=7)
            ax.set_ylabel(label)
        fig.suptitle(f'L={L}, T*={T}: each point is one run; errors require block-size checks')
        name = f'comparison_L{L}_T{T}.png'
        fig.savefig(output/name, dpi=130)
        plt.close(fig)
        links.append(f'- [L={L}, T*={T}: starts, seeds and lengths]({name})')
    return links


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+', help='files or quoted globs; ** recurses')
    p.add_argument('--output-dir', type=Path, default=Path('results/convergence'))
    p.add_argument('--L', type=int, nargs='+', help='only these lattice sizes')
    p.add_argument('--T', type=float, nargs='+', help='only these reduced temperatures')
    p.add_argument('--blocks', type=int, nargs='+', default=[50,100,200,500,1000,2000,5000,10000])
    p.add_argument('--discards', type=int, nargs='+', default=[0,5000,10000,20000])
    p.add_argument('--error-block', type=int, default=1000, help='block size for discard and run-comparison error bars')
    a = p.parse_args()
    if min(a.blocks) < 1 or a.error_block < 1 or min(a.discards) < 0:
        p.error('block sizes must be positive and discards nonnegative')
    paths = set()
    for pattern in a.inputs:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            p.error(f'no files match {pattern!r}')
        paths.update(Path(x).resolve() for x in matches)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows, sections, skipped, seen = [], [], [], set()
    for path in sorted(paths):
        try:
            with path.open() as f:
                meta = json.loads(f.readline()[2:])
            if a.L and meta['L'] not in a.L:
                continue
            if a.T and not any(np.isclose(meta['temperature'], t, rtol=0, atol=1e-10) for t in a.T):
                continue
            meta, data = read_samples(path)
            expected = meta['sweeps']//meta['every']
            if len(data) != expected or data['sweep'][0] != meta['every'] or data['sweep'][-1] != expected*meta['every']:
                raise ValueError('incomplete production data; wait until the run finishes')
            # Identical parameters and seed reproduce the same trajectory, not a new replica.
            identity = tuple(meta[k] for k in ['L','temperature','start','seed','burn','sweeps','every'])
            duplicate = identity in seen
            seen.add(identity)
            name = path.stem + '_' + hashlib.sha256(str(path).encode()).hexdigest()[:8]
            row, notes = diagnostics(path, meta, data, a.output_dir/name,
                                     sorted(set(a.blocks)), sorted(set([0]+a.discards)), a.error_block)
            rows.append(row)
            if duplicate:
                notes.append('Duplicate run settings and seed: do not count this file as an independent replica.')
            sections += [f'## {path.name}', '', f'Source: `{path}`', '',
                         f'[Traces]({name}/traces.png) | [Block/discard plots]({name}/sensitivity.png) | '
                         f'[Blocking data]({name}/blocking.csv) | [Discard data]({name}/discard.csv)', '']
            sections += ['- '+note for note in notes] + ['']
            print(f'Analyzed {path}', flush=True)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            skipped.append(f'- `{path}`: {exc}')
    if not rows:
        p.error('no complete matching runs. ' + '; '.join(skipped))
    write_csv(a.output_dir/'summary.csv', rows)
    links = comparisons(rows, a.output_dir)
    report = ['# Ising convergence checks', '',
              f'Analyzed {len(rows)} runs. Comparison/discard error block: {a.error_block} saved samples.', '',
              '**These diagnostics do not certify convergence.** Inspect the plots, replica scatter, and longer runs together.', '',
              '## How to read the results', '',
              '- Traces: look for drift in energy and |m|; sign reversals alone are not drift.',
              '- Blocking: error estimates should stabilize while many blocks remain. The 25% flag is a rough screening rule, not a statistical test.',
              '- Discard: look for persistent changes as more early samples are removed. Estimates overlap in data and are correlated.',
              '- Comparisons: random/up starts and different seeds should give compatible energy and |m| estimates; longer runs should support the same result.',
              '- Comparison errors use the requested block size, not an automatically selected plateau. Rerun with larger --error-block after inspecting blocking plots.',
              '- The plotted susceptibility uses |m|. Signed-m susceptibility is also retained in CSVs and is sensitive to sampling both signs.',
              '- Burn-in was never saved. All traces and discard checks concern production data.',
              '- This report does not estimate autocorrelation times or pool replicas. Equal seeds across starts or temperatures are not independent randomness.', '',
              '## Run comparisons', ''] + (links or ['Only one run per L,T: add the extra simulations to compare starts and lengths.'])
    report += ['', '## Numerical summary', '', '[Full summary CSV](summary.csv)', '',
               '| L | T* | Start | Seed | Burn | Production | mean abs(m) | <E/(NJ)> | chi_abs J | cV/kB |',
               '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        cells = [str(r[k]) for k in ['L','T','start','seed','burn','sweeps']]
        cells += [f"{r[k]:.5g} ± {r[k+'_err']:.2g}" for k in KEYS]
        report.append('| '+' | '.join(cells)+' |')
    report += ['', '## Skipped files', ''] + (skipped or ['None.']) + [''] + sections
    (a.output_dir/'report.md').write_text('\n'.join(report)+'\n')
    print(f'Read {a.output_dir / "report.md"}')


if __name__ == '__main__':
    main()
