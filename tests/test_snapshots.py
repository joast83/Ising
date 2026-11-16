import tempfile
import unittest
from pathlib import Path
import numpy as np
from ising import simulate, energy
from animate_spins import load_snapshots


class SnapshotTests(unittest.TestCase):
    def test_snapshots_do_not_change_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a.csv', Path(tmp)/'b.csv'
            args = (4, 2.3, 7, 13, 1, 42, 'random')
            final_a = simulate(*args, a)
            final_b = simulate(*args, b, snapshot_every=5)
            self.assertEqual(final_a, final_b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            meta, spins, steps = load_snapshots(b.with_suffix('.snapshots.npz'))
            np.testing.assert_array_equal(steps, [0, 5, 7, 10, 15, 20])
            np.testing.assert_array_equal(spins[-1], final_b[0])
            self.assertEqual(spins.dtype, np.int8)
            self.assertEqual(energy(spins[-1].tolist()), final_b[1])
            self.assertEqual(int(spins[-1].sum()), final_b[2])
            # For each captured production configuration, match the existing CSV.
            data = np.genfromtxt(b, delimiter=',', names=True, skip_header=1)
            for frame, step in zip(spins, steps):
                if step > meta['burn']:
                    row = data[data['sweep'] == step-meta['burn']][0]
                    self.assertEqual(energy(frame.tolist()), row['E_over_J'])
                    self.assertEqual(frame.sum(), row['M'])

    def test_no_burn_and_final_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'a.csv'
            simulate(3, 2, 0, 3, 1, 1, 'up', path, snapshot_every=10)
            _, spins, steps = load_snapshots(path.with_suffix('.snapshots.npz'))
            np.testing.assert_array_equal(steps, [0, 3])
            self.assertTrue(np.all(spins[0] == 1))

    def test_existing_snapshot_and_path_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, snapshots = Path(tmp)/'a.csv', Path(tmp)/'existing.npz'
            snapshots.write_bytes(b'keep me')
            args = (3, 2, 0, 3, 1, 1, 'up', path)
            with self.assertRaises(FileExistsError):
                simulate(*args, snapshot_every=1, snapshot_output=snapshots)
            self.assertFalse(path.exists())
            self.assertEqual(snapshots.read_bytes(), b'keep me')
            with self.assertRaises(ValueError):
                simulate(*args, snapshot_every=1, snapshot_output=path)
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
