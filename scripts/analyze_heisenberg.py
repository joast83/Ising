#!/usr/bin/env python3
"""Vector-specific thermodynamics and block-jackknife errors from existing runs."""
import argparse
import csv
import glob
import json
from pathlib import Path
import sys
import numpy as np

OBS = ['m_abs','e_over_J','chi_x_J','chi_y_J','chi_z_J','chi_iso_J',
       'chi_magnitude_J','cv_over_kB','binder','C1','B1','P2','nematic','acceptance']


def read_samples(path, discard=0):
    with open(path) as f:
        first = f.readline()
        if not first.startswith('# '):
            raise ValueError(f'{path}: missing metadata')
        meta = json.loads(first[2:])
        data = np.atleast_1d(np.genfromtxt(f, delimiter=',', names=True))
    if meta.get('model') != 'classical_heisenberg' or meta.get('schema_version') != 1:
        raise ValueError(f'{path}: requires vector-spin data from heisenberg.py')
    if meta['dim'] not in (2,3) or meta['N'] != meta['L']**meta['dim']:
        raise ValueError(f'{path}: inconsistent cell size')
    if meta['temperature'] <= 0 or not np.isfinite(meta['temperature']):
        raise ValueError(f'{path}: invalid temperature')
    required = ['sweep','e_over_J','mx','my','mz','m_abs','C1','B1','nematic','acceptance']
    if not set(required).issubset(data.dtype.names or ()):
        raise ValueError(f'{path}: missing measurement columns')
    if any(not np.all(np.isfinite(data[k])) for k in required):
        raise ValueError(f'{path}: nonfinite measurements')
    expected = meta['sweeps']//meta['every']
    if len(data) != expected or not np.array_equal(data['sweep'],np.arange(1,expected+1)*meta['every']):
        raise ValueError(f'{path}: incomplete or irregular run; wait until it finishes')
    if discard < 0 or len(data)-discard < 2:
        raise ValueError(f'{path}: need at least two samples after discard')
    if np.any(data['m_abs'] > 1+1e-10) or not np.allclose(
            data['m_abs']**2,data['mx']**2+data['my']**2+data['mz']**2,atol=1e-10):
        raise ValueError(f'{path}: inconsistent vector magnetization')
    if not np.allclose(data['e_over_J'],-meta['dim']*(data['C1']+meta['kappa']*data['B1']),atol=1e-10):
        raise ValueError(f'{path}: inconsistent energy and correlations')
    return meta, data[discard:]


def moments(data):
    x,y,z = data['mx'],data['my'],data['mz']
    m2 = x*x+y*y+z*z
    e = data['e_over_J']
    return np.column_stack([x,y,z,m2,data['m_abs'],m2*m2,e,e*e,
                            data['C1'],data['B1'],data['nematic'],data['acceptance'],x*x,y*y,z*z])


def observables(q, N, T):
    x,y,z,m2,a,m4,e,e2,c,b,nem,acc,x2,y2,z2 = np.moveaxis(q,-1,0)
    chis = [N*np.maximum(0,v2-v*v)/T for v,v2 in [(x,x2),(y,y2),(z,z2)]]
    ratio = np.divide(m4,m2*m2,out=np.full_like(m2,np.nan),where=m2>0)
    return np.stack([a,e,*chis,sum(chis)/3,N*np.maximum(0,m2-a*a)/T,
                     N*np.maximum(0,e2-e*e)/T**2,(5-3*ratio)/2,
                     c,b,(3*b-1)/2,nem,acc],axis=-1)


def summarize(path, discard=0, block_size=100):
    if block_size < 1:
        raise ValueError('block-size must be positive')
    meta,data = read_samples(path,discard)
    q = moments(data)
    values = observables(q.mean(axis=0),meta['N'],meta['temperature'])
    blocks = len(q)//block_size
    errors = np.full(len(OBS),np.nan)
    if blocks >= 2:
        means = q[:blocks*block_size].reshape(blocks,block_size,q.shape[1]).mean(axis=1)
        estimates = observables((means.sum(axis=0)-means)/(blocks-1),meta['N'],meta['temperature'])
        errors = np.sqrt((blocks-1)/blocks*((estimates-estimates.mean(axis=0))**2).sum(axis=0))
    if blocks < 20:
        print(f'Warning: {path}: only {blocks} complete blocks; errors are preliminary.',file=sys.stderr)
    row = dict(source=str(path),L=meta['L'],dim=meta['dim'],N=meta['N'],T=meta['temperature'],
               kappa=meta['kappa'],seed=meta['seed'],start=meta['start'],burn=meta['burn'],
               sweeps=meta['sweeps'],every=meta['every'],proposal=meta['proposal'],
               step_size=meta['step_size'],samples=len(q),discard=discard,
               block_size=block_size,blocks=blocks,error_samples=blocks*block_size if blocks>=2 else 0)
    for key,value,error in zip(OBS,values,errors):
        row[key],row[key+'_err'] = float(value),float(error)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+')
    p.add_argument('--discard',type=int,default=0,help='additional saved rows')
    p.add_argument('--block-size',type=int,default=100,help='saved samples per block')
    p.add_argument('--output',type=Path,default=Path('results/heisenberg/summary.csv'))
    a=p.parse_args();paths=set()
    for pattern in a.inputs:
        matches=glob.glob(pattern)
        if not matches:p.error(f'no files match {pattern!r}')
        paths.update(Path(x).resolve() for x in matches)
    if a.output.resolve() in paths:p.error('output must not overwrite an input')
    try:rows=[summarize(path,a.discard,a.block_size) for path in sorted(paths)]
    except (ValueError,KeyError,OSError) as exc:p.error(str(exc))
    rows.sort(key=lambda r:(r['dim'],r['kappa'],r['L'],r['T'],r['seed']))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Saved {len(rows)} run summaries to {a.output}')
    print('Check several block sizes and independent starts. Magnitude fluctuations are not the Cartesian response susceptibility.')


if __name__=='__main__':main()
