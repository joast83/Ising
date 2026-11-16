#!/usr/bin/env python3
"""Animate saved Ising snapshots as fixed blue/red dots; no simulation is run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, writers
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np


def load_snapshots(path):
    with np.load(path, allow_pickle=False) as data:
        spins, sweeps = data['spins'], data['sweep']
        meta = json.loads(str(data['metadata'].item()))
    L = meta['L']
    if (meta.get('snapshot_schema_version') != 1 or spins.ndim != 3
            or spins.shape[1:] != (L, L) or len(spins) == 0
            or sweeps.shape != (len(spins),)):
        raise ValueError('invalid snapshot format or lattice shape')
    if not np.all(np.isin(spins, [-1, 1])):
        raise ValueError('spins must be +1 or -1')
    if not np.all(np.isfinite(sweeps)) or np.any(np.diff(sweeps) <= 0):
        raise ValueError('snapshot sweeps must increase strictly')
    return meta, spins, sweeps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshots', type=Path)
    p.add_argument('--output', type=Path, default=Path('results/spins.gif'))
    p.add_argument('--fps', type=int, default=10, help='display frames per second')
    p.add_argument('--stride', type=int, default=1, help='use every nth saved frame')
    p.add_argument('--max-frames', type=int, default=500, help='maximum displayed frames, evenly selected')
    p.add_argument('--production-only', action='store_true', help='omit initialization/equilibration')
    a = p.parse_args()
    if min(a.fps, a.stride, a.max_frames) < 1:
        p.error('fps, stride, and max-frames must be positive')
    if a.output.suffix.lower() not in ('.gif', '.mp4'):
        p.error('output must end in .gif or .mp4')
    if a.output.exists():
        p.error(f'{a.output} exists; choose another filename')
    if a.output.suffix.lower() == '.mp4' and not writers.is_available('ffmpeg'):
        p.error('MP4 needs FFmpeg installed; use --output results/spins.gif instead')
    try:
        meta, spins, sweeps = load_snapshots(a.snapshots)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    indices = np.flatnonzero(sweeps > meta['burn']) if a.production_only else np.arange(len(spins))
    indices = indices[::a.stride]
    if len(indices) == 0:
        p.error('no frames selected')
    if len(indices) > a.max_frames:
        indices = indices[np.linspace(0, len(indices)-1, a.max_frames, dtype=int)]
    L = meta['L']
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.08, right=0.92)
    yy, xx = np.indices((L, L))
    red, blue = '#d73027', '#2166ac'
    dots = ax.scatter(xx.ravel(), yy.ravel(), c=spins[indices[0]].ravel(),
                      cmap=ListedColormap([red, blue]), norm=BoundaryNorm([-2, 0, 2], 2),
                      s=(240/L)**2, edgecolors='none')
    ax.set(xlim=(-0.7, L-0.3), ylim=(L-0.3, -0.7), aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title('')
    handles = [Line2D([], [], marker='o', linestyle='', color=c, label=label)
               for c, label in [(blue, '+1'), (red, '-1')]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.04), ncol=2)
    fig.text(0.5, 0.025, 'Periodic boundaries | Monte Carlo sweeps, not physical time',
             ha='center', fontsize=9)

    def update(index):
        step = int(sweeps[index])
        phase = ('Initial state' if step == 0 else
                 'Equilibration' if step <= meta['burn'] else
                 f"Production sweep {step-meta['burn']}")
        dots.set_array(spins[index].ravel())
        title.set_text(f"L = {L}, T* = {meta['temperature']}\n"
                       f"Total sweep {step} | {phase}\nm = {spins[index].mean():+.3f}")
        return dots, title

    animation = FuncAnimation(fig, update, frames=indices, interval=1000/a.fps,
                              blit=False, repeat=True)
    writer = PillowWriter(fps=a.fps) if a.output.suffix.lower() == '.gif' else FFMpegWriter(fps=a.fps)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(a.output), writer=writer)
    plt.close(fig)
    print(f'Saved {len(indices)} frames to {a.output}')
    print('Frames play at constant speed; see sweep labels for simulation intervals.')


if __name__ == '__main__':
    main()
