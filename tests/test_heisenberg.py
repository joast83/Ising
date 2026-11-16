import tempfile
import unittest
from pathlib import Path
import numpy as np
from heisenberg import neighbours, random_vectors, total_energy, delta_energy, measure, simulate, sweep
from analyze_heisenberg import moments, observables, summarize, read_samples, OBS
from plot_heisenberg import mean_field_m, langevin


class HeisenbergTests(unittest.TestCase):
    def test_bonds_and_ordered_state(self):
        for dim in [2,3]:
            nb=neighbours(4,dim);N=4**dim
            self.assertTrue(all(len(set(row))==2*dim for row in nb))
            self.assertTrue(all(i in nb[j] for i in range(N) for j in nb[i]))
            s=np.tile([0.,0.,1.],(N,1))
            for k in [-.5,0,.5]:
                self.assertAlmostEqual(total_energy(s,nb,k),-dim*N*(1+k))
                values=measure(s,nb,k)
                np.testing.assert_allclose(values,[-dim*(1+k),0,0,1,1,1,1,1])

    def test_rotation_and_local_energy(self):
        rng=np.random.default_rng(2)
        Q,_=np.linalg.qr(rng.normal(size=(3,3)))
        for dim in [2,3]:
            nb=neighbours(3,dim);s=random_vectors(rng,3**dim)
            for k in [-1.,0.,.7]:
                before=total_energy(s,nb,k)
                self.assertAlmostEqual(before,total_energy(s@Q,nb,k),places=11)
                for site in range(len(s)):
                    new=random_vectors(rng,1)[0];old=s[site].copy()
                    change=delta_energy(s,nb,site,new,k)
                    s[site]=new
                    self.assertAlmostEqual(total_energy(s,nb,k)-before,change,places=10)
                    s[site]=old

    def test_uniform_sphere_and_biquadratic_limits(self):
        rng=np.random.default_rng(5)
        x=random_vectors(rng,100000);y=random_vectors(rng,100000)
        np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-14)
        np.testing.assert_allclose(x.mean(axis=0),0,atol=.006)
        np.testing.assert_allclose((x*x).mean(axis=0),1/3,atol=.006)
        dots=np.sum(x*y,axis=1)
        self.assertLess(abs(dots.mean()),.006)
        self.assertLess(abs(np.mean(dots*dots)-1/3),.006)

    def test_sampling_and_snapshot_invariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.csv',Path(tmp)/'b.csv'
            args=(3,3,1.4,.2,5,30,2,7,'random')
            s=simulate(*args,a)
            v=simulate(*args,b,snapshot_every=7)
            np.testing.assert_array_equal(s,v)
            np.testing.assert_allclose(np.linalg.norm(s,axis=1),1,atol=1e-14)
            meta,data=read_samples(a)
            _,other=read_samples(b)
            np.testing.assert_array_equal(data,other)
            with np.load(b.with_suffix('.vectors.npz'),allow_pickle=False) as snapshots:
                np.testing.assert_array_equal(snapshots['spins'][-1].reshape(-1,3),s)
                np.testing.assert_array_equal(snapshots['sweep'],[0,5,7,14,21,28,35])
            nb=neighbours(3,3)
            self.assertAlmostEqual(data['e_over_J'][-1],total_energy(s,nb,.2)/27)
            with self.assertRaises(FileExistsError):simulate(*args,a)
            with self.assertRaises(ValueError):simulate(2,*args[1:],Path(tmp)/'bad.csv')

    def test_vector_susceptibility_normalization(self):
        # One unit spin sampled equally on the six axes: isotropic first/second moments.
        dtype=[(key,float) for key in ['mx','my','mz','m_abs','e_over_J','C1','B1','nematic','acceptance']]
        data=np.zeros(6,dtype=dtype)
        directions=np.r_[np.eye(3),-np.eye(3)]
        for i,key in enumerate(['mx','my','mz']):data[key]=directions[:,i]
        data['m_abs']=1
        result=dict(zip(OBS,observables(moments(data).mean(axis=0),1,2)))
        for key in ['chi_x_J','chi_y_J','chi_z_J','chi_iso_J']:
            self.assertAlmostEqual(result[key],1/6)
        self.assertAlmostEqual(result['chi_magnitude_J'],0)
        self.assertAlmostEqual(result['binder'],1)
        self.assertAlmostEqual(result['cv_over_kB'],0)

    def test_jackknife(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.csv';simulate(3,2,1.4,0,5,101,1,7,'random',path)
            meta,data=read_samples(path);q=moments(data)
            row=summarize(path,block_size=10)
            samples=[]
            for i in range(10):
                remaining=np.delete(q[:100],np.s_[i*10:(i+1)*10],axis=0)
                samples.append(observables(remaining.mean(axis=0),9,1.4))
            samples=np.array(samples)
            expected=np.sqrt(.9*((samples-samples.mean(axis=0))**2).sum(axis=0))
            np.testing.assert_allclose([row[k+'_err'] for k in OBS],expected,atol=1e-12)
            self.assertEqual(row['samples'],101)
            self.assertEqual(row['error_samples'],100)

    def test_mean_field(self):
        for z in [4,6]:
            for t in [.1,1.,1.3,1.9,2.,3.]:
                m=mean_field_m(t,z)
                self.assertAlmostEqual(m,langevin(z*m/t),places=10)
                if t>=z/3:self.assertEqual(m,0)


if __name__=='__main__':unittest.main()
