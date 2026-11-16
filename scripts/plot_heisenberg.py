#!/usr/bin/env python3
"""Plot vector-spin summaries, separating lattice dimension and K/J."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def langevin(x):
    if abs(x)<1e-3:return x/3-x**3/45+2*x**5/945
    return 1/np.tanh(x)-1/x


def mean_field_m(T,coordination):
    if T>=coordination/3:return 0.
    lo,hi=0.,1.
    for _ in range(80):
        mid=(lo+hi)/2
        if langevin(coordination*mid/T)>mid:lo=mid
        else:hi=mid
    return (lo+hi)/2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('results/heisenberg'))
    a=p.parse_args()
    with a.summary.open() as f:rows=list(csv.DictReader(f))
    if not rows:p.error('empty summary')
    panels=[('m_abs',r'$\langle|\mathbf{m}|\rangle$'),('chi_magnitude_J',r'$J\chi_{\rm magnitude}$'),
            ('cv_over_kB',r'$c_V/k_B$'),('e_over_J',r'$\langle E\rangle/(NJ)$'),
            ('chi_iso_J',r'$J\chi_{\rm iso}$'),('binder',r'$U_4$ (O(3), normalized)'),
            ('C1',r'$\langle\mathbf{e}_i\cdot\mathbf{e}_j\rangle_{NN}$'),
            ('B1',r'$\langle(\mathbf{e}_i\cdot\mathbf{e}_j)^2\rangle_{NN}$'),('nematic','Largest eigenvalue of Q')]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for dim,kappa in sorted({(int(r['dim']),float(r['kappa'])) for r in rows}):
        subset=[r for r in rows if int(r['dim'])==dim and float(r['kappa'])==kappa]
        fig,axes=plt.subplots(3,3,figsize=(12,10),constrained_layout=True)
        for ax,(key,label) in zip(axes.flat,panels):
            for i,L in enumerate(sorted({int(r['L']) for r in subset})):
                group=sorted([r for r in subset if int(r['L'])==L],key=lambda r:float(r['T']))
                x=np.array([float(r['T']) for r in group]);y=np.array([float(r[key]) for r in group])
                err=np.array([float(r[key+'_err']) for r in group])
                color=['black','#1f77b4','#E69F00','#009E73'][i%4]
                ax.plot(x,y,linestyle='none',marker=['o','s','^','D','v','P','X'][i%7],color=color,label=f'L={L}, N={L**dim}')
                good=np.isfinite(err)&np.isfinite(y)
                ax.errorbar(x[good],y[good],yerr=err[good],fmt='none',ecolor=color,capsize=2)
            if kappa==0:
                ax.axvline(2*dim/3,ls=':',color='0.5',label='MF Tc (K=0)')
                if dim==3:ax.axvline(1.443,ls='--',color='0.7',label='MC literature Tc ~1.443')
            ax.set(xlabel='T* = kB T / J',ylabel=label)
        if kappa==0:
            temps=np.linspace(min(.2,min(float(r['T']) for r in subset)),max(2.2,max(float(r['T']) for r in subset)),400)
            m=np.array([mean_field_m(t,2*dim) for t in temps])
            axes[0,0].plot(temps,m,color='#009E73',ls='--',label='MF m (K=0)')
            axes[1,0].plot(temps,-dim*m*m,color='#009E73',ls='--')
            axes[2,0].plot(temps,m*m,color='#009E73',ls='--')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'{dim}D lattice, three-component unit spins, K/J={kappa:g}' +
                     (' | no finite-T ferromagnetic LRO for 2D K=0 in the thermodynamic limit' if dim==2 and kappa==0 else ''))
        stem=f'vectors_d{dim}_K{kappa:g}'
        for ext in ['png','pdf']:fig.savefig(a.output_dir/f'{stem}.{ext}',dpi=140)
        plt.close(fig)
        print(f'Saved {a.output_dir/stem}.png and .pdf')


if __name__=='__main__':main()
