# Square-lattice Ising model

A compact, readable Metropolis Monte Carlo simulation in Python 3, using only
the standard library. No packages need installing.

The Hamiltonian is H = -J sum_<ij> s_i s_j, with s_i = +/-1, J > 0,
zero external field, and each nearest-neighbour pair counted once.
The L x L lattice has periodic boundaries (L >= 3).
Temperature is supplied as **T* = k_B T/J**; energies are recorded in units of J.
One sweep is L^2 random-site flip attempts, chosen with replacement.
A flip costs Delta E/J = 2 s_i sum_neighbours s_j and is accepted with
probability min(1, exp[-(Delta E/J)/T*]).

## Run locally

From this repository's directory:

```bash
python3 ising.py --L 16 --T 2.3 --burn 5000 --sweeps 20000 --seed 1 --output data/L16_T2.3_seed1.csv
```

For a first temperature and size scan (bash):

```bash
for L in 8 16 24; do
  for T in 1.5 2.0 2.1 2.2 2.25 2.3 2.35 2.4 2.5 3.0 4.0 5.0; do
    python3 ising.py --L "$L" --T "$T" --burn 5000 --sweeps 20000 \
      --seed 1 --output "data/L${L}_T${T}_seed1.csv"
  done
done
```

Each command initializes a fresh chain. Repeat with different seeds for
independent replicas. `--start up` starts fully ordered instead of random;
comparing the two is useful for checking equilibration. These run lengths are
starting values, not a convergence guarantee. Single-spin Metropolis sampling
slows down near the transition, and this deliberately simple Python loop is
intended for modest lattice sizes. Increase equilibration and production lengths
after checking convergence and autocorrelation. Production cost scales as L^2
times the number of sweeps. `--every 10` reduces saved data, not simulation work,
and does not guarantee independent samples.

## Output and future analysis

Each CSV starts with a `#` comment containing JSON run metadata, followed by:

| Column | Meaning |
| --- | --- |
| sweep | Production sweep, starting after discarded equilibration |
| E_over_J | Total energy divided by J |
| M | Total signed magnetization, sum_i s_i |
| e_over_J | Energy per spin divided by J |
| m | Signed magnetization per spin, M/N, where N = L^2 |

Existing output files are never overwritten. The metadata records the seed,
temperature, lattice size, initialization and sampling settings. Equilibration
samples are not saved. A simple way to read a file is:

```python
import csv
import json
with open("data/L16_T2.3_seed1.csv") as f:
    metadata = json.loads(f.readline()[2:])
    rows = list(csv.DictReader(f))  # convert column strings to numbers for analysis
```

The saved time series supports these later estimates (averages over equilibrated
samples, e* = E/(NJ)):

- Mean absolute magnetization per spin: <|m|>.
- Susceptibility per spin: J chi = N (<m^2> - <m>^2)/T*.
  Here the conjugate field enters H as -h sum_i s_i and chi = d<m>/dh.
- Heat capacity per spin: c_V/k_B = N (<e*^2> - <e*>^2)/(T*)^2.

Keep the sign of m: replacing <m>^2 with <|m|>^2 produces a different,
often-used finite-size estimator, not the same zero-field susceptibility.
At low temperature a finite run may stay in one magnetization sector; this
affects the sampled susceptibility. Future postprocessing should handle
autocorrelation, blocking/error estimates and equilibration checks before
interpreting peaks. Raw samples also allow Binder cumulants and histogram
analysis without changing the simulation.

Finite lattices have rounded transitions. Later, compare size-dependent results
with the thermodynamic-limit values T*_c = 4 (mean-field theory) and
T*_c = 2/ln(1 + sqrt(2)) = 2.269185... (Onsager).
No transition temperature is inferred by the simulation itself.

## Checks

```bash
python3 -m unittest -v
```

Tests check bond counting, all single-spin energy differences on the 3 x 3
lattice, detailed balance, incremental energy/magnetization, and saved samples.
