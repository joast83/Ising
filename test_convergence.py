import tempfile
import unittest
from pathlib import Path
import numpy as np
from ising import simulate
from analyze import read_samples, moments, summarize, OBSERVABLES
from check_convergence import estimate
from run_convergence import plan, complete


class ConvergenceTests(unittest.TestCase):
    def test_plans_and_overlap(self):
        self.assertEqual(len(plan('quick')), 4)
        self.assertEqual(len(plan('extended')), 24)
        self.assertEqual(len(plan('long')), 4)
        self.assertTrue(set(plan('quick')).issubset(set(plan('extended'))))
        self.assertFalse(set(plan('quick')) & set(plan('long')))

    def test_resume_rejects_partial_and_wrong_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run.csv'
            settings = (4, 2.3, 'random', 2, 10, 100)
            simulate(4, 2.3, 10, 100, 1, 2, 'random', path)
            self.assertTrue(complete(path, settings))
            self.assertFalse(complete(path, (4,2.3,'up',2,10,100)))
            path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
            self.assertFalse(complete(path, settings))

    def test_errors_match_existing_analysis_and_short_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run.csv'
            simulate(4, 2.3, 10, 101, 1, 2, 'random', path)
            meta, data = read_samples(path)
            values, errors, blocks = estimate(moments(data), 16, 2.3, 10)
            row = summarize(path, block_size=10)
            np.testing.assert_allclose(values, [row[k] for k in OBSERVABLES])
            np.testing.assert_allclose(errors, [row[k+'_err'] for k in OBSERVABLES])
            self.assertEqual(blocks, 10)
            self.assertTrue(np.all(np.isnan(estimate(moments(data),16,2.3,200)[1])))


if __name__ == '__main__':
    unittest.main()
