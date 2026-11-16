"""Small deterministic physics and output checks; no long production runs."""

import csv
import itertools
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from ising import delta_energy, energy, simulate, sweep


class IsingTests(unittest.TestCase):
    def test_ordered_and_checkerboard_energies(self):
        self.assertEqual(energy([[1] * 4 for _ in range(4)]), -32)
        self.assertEqual(energy([[(-1) ** (i + j) for j in range(4)]
                                 for i in range(4)]), 32)

    def test_all_3x3_flips_and_detailed_balance(self):
        temperature = 2.3
        for values in itertools.product((-1, 1), repeat=9):
            spins = [list(values[i:i + 3]) for i in range(0, 9, 3)]
            before = energy(spins)
            for i, j in itertools.product(range(3), repeat=2):
                dE = delta_energy(spins, i, j)
                spins[i][j] *= -1
                after = energy(spins)
                self.assertEqual(after - before, dE)
                forward = min(1, math.exp(-dE / temperature))
                backward = min(1, math.exp(dE / temperature))
                self.assertAlmostEqual(math.exp(-before / temperature) * forward,
                                       math.exp(-after / temperature) * backward)
                spins[i][j] *= -1

    def test_incremental_observables(self):
        rng = random.Random(42)
        spins = [[rng.choice((-1, 1)) for _ in range(5)] for _ in range(5)]
        E, M = energy(spins), sum(map(sum, spins))
        weights = {dE: math.exp(-dE / 2.3) for dE in (4, 8)}
        for _ in range(100):
            E, M = sweep(spins, E, M, weights, rng)
            self.assertEqual(E, energy(spins))
            self.assertEqual(M, sum(map(sum, spins)))

    def test_output_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            args = (4, 2.3, 10, 30, 3, 123, "random")
            spins, E, M = simulate(*args, a)
            simulate(*args, b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with a.open() as f:
                self.assertEqual(json.loads(f.readline()[2:])["N"], 16)
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 10)
            self.assertEqual(int(rows[-1]["sweep"]), 30)
            self.assertEqual(float(rows[-1]["e_over_J"]), energy(spins) / 16)
            self.assertEqual(float(rows[-1]["m"]), sum(map(sum, spins)) / 16)
            self.assertEqual((E, M), (energy(spins), sum(map(sum, spins))))
            with self.assertRaises(FileExistsError):
                simulate(*args, a)


if __name__ == "__main__":
    unittest.main()
