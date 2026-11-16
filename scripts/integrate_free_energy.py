#!/usr/bin/env python3
"""Exploratory Ising magnetic free energy from a temperature-dependent energy summary."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ising_theory import integrate_energy, mean_field


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('results/free_energy'))
    p.add_argument('--allow-coarse-reference',action='store_true',help='allow Tmax<10; diagnostic use only')
    a=p.parse_args()
    with a.summary.open() as f:rows=list(csv.DictReader(f))
    if not rows:p.error('empty summary')
    results=[]
    for L in sorted({int(r['L']) for r in rows}):
        group=[r for r in rows if int(r['L'])==L]
        T=[float(r['T']) for r in group];u=[float(r['e_over_J']) for r in group]
        if len(set(T))!=len(T):p.error(f'L={L}: more than one run per T; select a single consistent scan first (replicas are not pooled automatically)')
        if max(T)<10 and not a.allow_coarse_reference:p.error(f'L={L}: Tmax={max(T):g}<10; add high-T data or explicitly use --allow-coarse-reference')
        try:temperatures,energy,free,entropy=integrate_energy(T,u)
        except ValueError as exc:p.error(str(exc))
        for t,e,f,s in zip(temperatures,energy,free,entropy):
            mf=mean_field(float(t))
            results.append(dict(L=L,T=t,e_over_J=e,f_over_J=f,s_over_kB=s,
                                f_MF_over_J=mf['f_over_J'],s_MF_over_kB=mf['s_over_kB']))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'free_energy.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for i,L in enumerate(sorted({r['L'] for r in results})):
        group=sorted([r for r in results if r['L']==L],key=lambda r:r['T'])
        color=['black','#1f77b4','#E69F00','#009E73'][i%4]
        axes[0].plot([r['T'] for r in group],[r['f_over_J']+r['T']*np.log(2) for r in group],'o-',color=color,label=f'L={L}')
        axes[1].plot([r['T'] for r in group],[r['s_over_kB'] for r in group],'o-',color=color,label=f'L={L}')
    temps=np.linspace(min(r['T'] for r in results),max(r['T'] for r in results),500)
    mf=[mean_field(float(t)) for t in temps]
    axes[0].plot(temps,[r['f_over_J']+r['T']*np.log(2) for r in mf],'--',color='0.5',label='MF')
    axes[1].plot(temps,[r['s_over_kB'] for r in mf],'--',color='0.5',label='MF')
    axes[0].set(xlabel='T*',ylabel='F/(NJ) + T* ln 2')
    axes[1].set(xlabel='T*',ylabel='S/(N kB)')
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(a.output_dir/'free_energy.png',dpi=160);plt.close(fig)
    warning=('Exploratory trapezoidal integration: the unsampled interval from beta=0 to 1/Tmax is approximated linearly. '
             'Tmax>=10 is a screening rule, not a convergence guarantee. Repeat with higher Tmax and finer temperatures. '
             'No statistical or integration-error bars are supplied. Free-energy/entropy differences can be biased by the reference and grid. '
             'Entropy outside [0,ln(2)] indicates sampling or numerical problems; it is not clipped.\n')
    (a.output_dir/'integration_notes.txt').write_text(warning)
    print(f'Saved exploratory free energies to {a.output_dir}\n{warning}')


if __name__=='__main__':main()
