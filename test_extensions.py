import unittest
import itertools
import numpy as np
from ising_theory import mean_field, integrate_energy
from spatial_correlations import correlations
from paramagnetic_references import metrics


class ExtensionTests(unittest.TestCase):
    def test_mean_field_self_consistency(self):
        for T in [.2,1,2.3,3.9,3.999,4,5,10]:
            r=mean_field(T)
            self.assertAlmostEqual(r['m'],np.tanh(4*r['m']/T),places=11)
            self.assertAlmostEqual(r['f_over_J'],r['e_over_J']-T*r['s_over_kB'])
        self.assertEqual(mean_field(5)['m'],0)
        self.assertAlmostEqual(mean_field(5)['chi_J'],1)
        self.assertTrue(np.isinf(mean_field(4)['chi_J']))

    def test_heat_capacity_derivative(self):
        for T in [1,2.3,3.9,5]:
            h=1e-5
            derivative=(mean_field(T+h)['e_over_J']-mean_field(T-h)['e_over_J'])/(2*h)
            self.assertAlmostEqual(mean_field(T)['cv_over_kB'],derivative,places=7)

    def test_spatial_sum_rules(self):
        for s in [np.ones((4,4)),np.indices((4,4)).sum(axis=0)%2*2-1,
                  np.random.default_rng(1).choice([-1,1],size=(4,4))]:
            C,S=correlations(s)
            self.assertAlmostEqual(C[0,0],1)
            self.assertAlmostEqual(S[0,0],s.size*s.mean()**2)
            self.assertAlmostEqual(S.sum(),s.size)
            self.assertAlmostEqual((C[0,1]+C[1,0])/2,metrics(s)[1])
            for dx in range(4):
                for dy in range(4):
                    self.assertAlmostEqual(C[dx,dy],np.mean(s*np.roll(s,(dx,dy),(0,1))))

    def test_compensated_reference_exact_enumeration(self):
        # All M=0 configurations on a 4x4 lattice, equally weighted.
        values=[]
        for up in itertools.combinations(range(16),8):
            s=-np.ones(16,dtype=int);s[list(up)]=1
            values.append(metrics(s.reshape(4,4))[1])
        self.assertAlmostEqual(np.mean(values),-1/15)

    def test_integration_known_curve(self):
        T=np.array([2,4,8,16.])
        # For u=-2 beta, the integration is exactly -beta^2.
        temps,u,f,s=integrate_energy(T,-2/T)
        np.testing.assert_allclose(f,-temps*np.log(2)-1/temps)
        np.testing.assert_allclose(s,np.log(2)-1/temps**2)
        with self.assertRaises(ValueError):integrate_energy([2,2],[0,0])


if __name__=='__main__':unittest.main()
