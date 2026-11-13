#!/usr/bin/env python3
"""Generate uncorrelated and exactly compensated reference spin configurations."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def metrics(s):
    c1=np.mean(s*(np.roll(s,1,axis=0)+np.roll(s,1,axis=1)))/2
    return float(s.mean()),float(c1),float(-2*c1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--L',type=int,default=16)
    p.add_argument('--samples',type=int,default=2000)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--output-dir',type=Path,default=Path('results/references'))
    a=p.parse_args()
    if a.L<4 or a.L%2 or a.samples<2:p.error('use even L>=4 and samples>=2')
    rng=np.random.default_rng(a.seed);N=a.L*a.L
    rows=[]; best=None;best_score=float('inf')
    for kind in ['independent','M_zero']:
        for index in range(a.samples):
            if kind=='independent':s=rng.choice([-1,1],size=(a.L,a.L))
            else:s=rng.permutation(np.r_[np.ones(N//2,dtype=int),-np.ones(N//2,dtype=int)]).reshape(a.L,a.L)
            m,c,e=metrics(s)
            rows.append(dict(ensemble=kind,sample=index,m=m,C1=c,e_over_J=e))
            if kind=='M_zero' and abs(c)<best_score:best=s.copy();best_score=abs(c)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.save(a.output_dir/'representative.npy',best)
    summary=[]
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for kind,color in [('independent','#1f77b4'),('M_zero','#E69F00')]:
        group=[r for r in rows if r['ensemble']==kind];c=np.array([r['C1'] for r in group])
        summary.append(dict(ensemble=kind,L=a.L,samples=a.samples,seed=a.seed,C1_mean=c.mean(),
                            C1_standard_error=c.std(ddof=1)/np.sqrt(a.samples),
                            C1_expected=0 if kind=='independent' else -1/(N-1)))
        ax.hist(c,bins=30,alpha=.5,label=kind,color=color,density=True)
    ax.set(xlabel='C1',ylabel='Probability density');ax.legend()
    fig.savefig(a.output_dir/'C1_distribution.png',dpi=160);plt.close(fig)
    with (a.output_dir/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(f'Saved references to {a.output_dir}; these are not finite-temperature equilibrium samples.')
    print(f'Representative M=0 configuration selected for smallest |C1|: {best_score:g}.')
    print('This is a nearest-neighbour-only selection, not a full magnetic SQS or thermal ensemble.')


if __name__=='__main__':main()
