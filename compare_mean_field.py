#!/usr/bin/env python3
"""Compare an analyze.py summary with mean field and extract C1 from energy."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ising_theory import mean_field


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('results/mean_field'))
    a=p.parse_args()
    with a.summary.open() as f:
        rows=list(csv.DictReader(f))
    if not rows:
        p.error('empty summary')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r['C1']=-float(r['e_over_J'])/2
        r['C1_err']=float(r['e_over_J_err'])/2
        r['C1_MF']=mean_field(float(r['T']))['C1']
    with (a.output_dir/'sro.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    temps=np.linspace(min(.5,min(float(r['T']) for r in rows)),max(5,max(float(r['T']) for r in rows)),600)
    theory=[mean_field(float(t)) for t in temps]
    with (a.output_dir/'mean_field.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(theory[0]));w.writeheader();w.writerows(theory)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,key,mfkey,label in zip(axes.flat,['abs_m','e_over_J','C1','cv_over_kB'],
                                  ['m','e_over_J','C1','cv_over_kB'],
                                  ['Mean |m| / MF m','Energy per spin / J','Nearest-neighbour C1','Heat capacity per spin / kB']):
        # Split at Tc so the heat-capacity discontinuity is not interpolated across.
        for mask in (temps<4,temps>4):
            ax.plot(temps[mask],np.array([r[mfkey] for r in theory])[mask],color='black',ls='--',label='Mean field' if mask[0] else None)
        for i,L in enumerate(sorted({int(r['L']) for r in rows})):
            group=sorted([r for r in rows if int(r['L'])==L],key=lambda r:float(r['T']))
            x=np.array([float(r['T']) for r in group]); y=np.array([float(r[key]) for r in group])
            err=np.array([float(r[key+'_err']) for r in group])
            color=['#1f77b4','#E69F00','#009E73','black'][i%4]
            ax.plot(x,y,linestyle='none',marker=['o','s','^','D','v','P','X','*'][i%8],color=color,label=f'MC L={L}')
            good=np.isfinite(err)
            ax.errorbar(x[good],y[good],yerr=err[good],fmt='none',ecolor=color,capsize=2)
        ax.axvline(2/np.log(1+np.sqrt(2)),ls=':',color='0.6')
        ax.set(xlabel='T* = kB T / J',ylabel=label)
    axes[0,0].legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(a.output_dir/f'comparison.{ext}',dpi=160)
    plt.close(fig)
    print(f'Saved comparison plots, sro.csv and mean_field.csv to {a.output_dir}')
    print('MF m is a broken-symmetry order parameter; finite-cell mean |m| is not identical.')
    print('MF chi in the table is a within-branch response, not finite-cell reversal susceptibility.')


if __name__=='__main__':main()
