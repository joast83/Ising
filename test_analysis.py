import tempfile
import unittest
from pathlib import Path
import numpy as np
from analyze import observables, summarize, read_samples
from ising import simulate


class AnalysisTests(unittest.TestCase):
    def test_known_moments(self):
        # Two equally weighted states: m=+/-1, e=-2 or 0.
        result = observables(np.array([1, 0, -1, 1, 2]), 16, 2)
        np.testing.assert_allclose(result, [1, 0, -1, 8, 0, 4])

    def test_constant_ordered_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'samples.csv'
            simulate(4, 0.001, 0, 20, 1, 1, 'up', path)
            row = summarize(path, block_size=2)
            self.assertEqual(row['abs_m'], 1)
            for key in ('chi_J', 'chi_abs_J', 'cv_over_kB', 'abs_m_err'):
                self.assertEqual(row[key], 0)
            with self.assertRaises(ValueError):
                read_samples(path, discard=19)
            self.assertEqual(len(read_samples(path, discard=5)[1]), 15)

    def test_jackknife_against_explicit_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'samples.csv'
            simulate(4, 3, 10, 101, 1, 4, 'random', path)
            row = summarize(path, block_size=10)
            meta, data = read_samples(path)
            from analyze import moments, OBSERVABLES
            q = moments(data[:100])
            estimates = []
            for i in range(10):
                remaining = np.delete(q, np.s_[i*10:(i+1)*10], axis=0)
                estimates.append(observables(remaining.mean(axis=0), 16, 3))
            estimates = np.array(estimates)
            errors = np.sqrt(9/10 * ((estimates-estimates.mean(axis=0))**2).sum(axis=0))
            np.testing.assert_allclose([row[k+'_err'] for k in OBSERVABLES], errors, atol=1e-12)
            self.assertEqual(row['n_samples'], 101)
            self.assertEqual(row['error_samples'], 100)
            short = summarize(path, block_size=200)
            self.assertTrue(np.isnan(short['chi_J_err']))


if __name__ == '__main__':
    unittest.main()
