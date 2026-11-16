#!/usr/bin/env python3
"""Raw periodic spin correlations and structure factors from saved snapshots."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from animate_spins import load_snapshots


def correlations(spins):
    N=spins.size
    power=np.abs(np.fft.fft2(spins))**2
    return np.fft.ifft2(power).real/N,power/N


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshots',type=Path)
    p.add_argument('--discard-sweeps',type=int,default=0,help='additional production sweeps to discard')
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--output-dir',type=Path,default=Path('results/spatial'))
    a=p.parse_args()
    if a.discard_sweeps<0 or a.stride<1:p.error('discard must be >=0; stride >=1')
    meta,spins,sweeps=load_snapshots(a.snapshots)
    indices=np.flatnonzero(sweeps>meta['burn']+a.discard_sweeps)[::a.stride]
    if len(indices)<2:p.error('need at least two retained production snapshots')
    L=meta['L']; C=np.zeros((L,L)); S=np.zeros_like(C)
    for index in indices:
        c,s=correlations(spins[index]); C+=c;S+=s
    C/=len(indices);S/=len(indices)
    # Average the positive and negative x/y directions. Not a radial-shell average.
    distances=np.arange(L//2+1)
    axial=np.array([(C[0,r]+C[0,(-r)%L]+C[r,0]+C[(-r)%L,0])/4 for r in distances])
    q=2*np.pi*np.fft.fftshift(np.fft.fftfreq(L))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output_dir/'correlations.npz',C=C,S=S,q_unshifted=2*np.pi*np.fft.fftfreq(L),
                        retained_sweeps=sweeps[indices],L=L,temperature=meta['temperature'])
    with (a.output_dir/'axial_C.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['distance_over_a','C_axial']);w.writerows(zip(distances,axial))
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    axes[0].plot(distances,axial,'o-',color='black');axes[0].set(xlabel='Axial separation / a',ylabel='Raw C(r)')
    im=axes[1].imshow(np.fft.fftshift(C),origin='lower',cmap='coolwarm',vmin=-1,vmax=1)
    axes[1].set(title='C(displacement), zero at center',xlabel='Shifted displacement index');fig.colorbar(im,ax=axes[1])
    im=axes[2].pcolormesh(q,q,np.log10(1+np.fft.fftshift(S)),shading='nearest',cmap='cividis')
    axes[2].set(title='log10(1 + S(q))',xlabel='qx a',ylabel='qy a');fig.colorbar(im,ax=axes[2])
    fig.savefig(a.output_dir/'spatial.png',dpi=160);plt.close(fig)
    c1=(C[0,1]+C[1,0])/2
    (a.output_dir/'notes.txt').write_text(
        f'L={L}, T*={meta["temperature"]}, retained frames={len(indices)}\n'
        f'C(0)={C[0,0]:.8g}; C1={c1:.8g}; S(0)={S[0,0]:.8g}\n'
        'C is raw <s_i s_j>, not connected and not normalized by C1.\n'
        'S(q)=<|sum_j s_j exp(i q.r_j)|^2>/N; S(0)=N<m^2>.\n'
        'Only production snapshots are used. Sparse frames can remain correlated.\n'
        'All retained frames are equally weighted; boundary snapshots can be irregularly spaced.\n'
        'These are exploratory means, without error bars or an automatic correlation-length fit.\n')
    print(f'Saved spatial diagnostics from {len(indices)} frames to {a.output_dir}')


if __name__=='__main__':main()
