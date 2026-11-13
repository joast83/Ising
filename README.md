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

## Separate postprocessing (run only after simulations)

The simulation never calls these scripts. They read existing files and do not
run or alter your simulations. Analysis requires NumPy; plotting additionally
requires Matplotlib. Install them in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

Analyze a single run, or all your sample CSV files:

```bash
python analyze.py data/L16_T2.3_seed1.csv --output results/single_run.csv
python analyze.py 'data/*.csv' --output results/summary.csv
python plot_results.py results/summary.csv
python plot_timeseries.py data/L16_T2.3_seed1.csv --output results/L16_T2.3_trace.png
```

`plot_results.py` writes `results/observables.png` and `.pdf`, showing each L
with its own marker/color and the Onsager and mean-field reference temperatures.
Each input file remains a separate point, including repeated seeds at the same
L and T; replicas are not silently combined. No transition temperature is fitted.
`plot_timeseries.py` plots signed m and energy against production sweep, plus
running averages. Inspect these and compare ordered/random starts and longer
runs; an apparently flat trace alone does not prove equilibration.

`analyze.py` writes one row per input file, taking L and T from file metadata,
not filenames. Columns `abs_m`, `m`, `e_over_J`, `chi_J`, `chi_abs_J`, and
`cv_over_kB` contain the mean absolute magnetization, mean signed magnetization,
mean energy per spin/J, signed-m fluctuation susceptibility times J,
N(<m^2>-<|m|>^2)/T*, and heat capacity per spin/k_B, respectively.
The `_err` columns are preliminary standard errors from a contiguous-block
leave-one-block-out jackknife. Run identifiers and sampling counts are retained.

By default a block contains 100 **saved samples**, or 100 times `--every`
simulation sweeps. Block size is not an automatically measured correlation time.
Repeat analysis with progressively larger blocks and check whether errors
stabilize while retaining many blocks (ideally at least 20). For example:

```bash
python analyze.py 'data/*.csv' --block-size 500 --output results/summary_b500.csv
python analyze.py 'data/*.csv' --block-size 1000 --output results/summary_b1000.csv
```

All retained samples enter central estimates. An incomplete final block is
excluded only from the error calculation; `error_samples` records its sample
count. Fewer than two complete blocks produces NaN errors, and fewer than ten
produces a warning. Plots omit unavailable error bars. Zero estimated error in
a frozen chain does not establish convergence. Errors may be underestimated
if blocks remain correlated or the run does not explore relevant states.

Use `--discard 1000` to exclude an additional 1000 saved rows from each input
file, beyond the burn-in already discarded by the simulator. The same option
is available for time-series plots. Do not apply a large discard blindly to
short runs. Analysis and plot outputs are replaced when rerun; raw simulation
files are left intact. Keep summaries in `results/`, outside the sample glob.

After installing analysis dependencies, `python -m unittest -v` also checks
observable formulas and block-jackknife errors against explicit block deletion.

## Animate spin configurations

Snapshot recording is optional. Ordinary runs and their CSV format are unchanged.
Install `requirements-analysis.txt` in the virtual environment shown above.
NumPy is needed only when recording snapshots; animation uses Matplotlib and
Pillow. The simulation and animation remain separate commands.

Try a short demonstration first (this is not a converged thermodynamic run):

```bash
python ising.py --L 32 --T 2.3 --burn 500 --sweeps 1000 --seed 1 \
  --output data/demo.csv --snapshot-every 5
python animate_spins.py data/demo.snapshots.npz --output results/spins.gif
```

Dots stay at fixed lattice positions: **blue = +1**, **red = -1**. The animation
shows L, reduced temperature, total sweep, simulation phase, and signed m.
It represents Monte Carlo sampling, not physical time. Periodic neighbours
across opposite edges still interact, even though the picture has a square frame.

New simulation options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--snapshot-every` | 0 | Snapshot interval in total sweeps; 0 disables recording |
| `--snapshot-output` | derived from CSV name | Optional snapshot path; default `demo.snapshots.npz` for `demo.csv` |

Snapshots include the initial state (total sweep 0), every requested interval
through equilibration and production, the end of equilibration, and the final
state. These extra boundary frames may produce unequal intervals. Recording
snapshots does not consume random numbers and does not change the sampled chain.
Existing sample and snapshot files are not overwritten.

The compressed NumPy archive contains `spins` (int8 array, frames x L x L),
`sweep` (total sweeps since initialization), and `metadata` (JSON text with
simulation settings and snapshot interval). Load with `np.load(..., allow_pickle=False)`.
Snapshots are held in memory until the simulation finishes, then saved. Memory
scales as frames times L squared bytes, with additional temporary copies during
saving. Use short demonstration runs or larger intervals for large lattices or
long runs. Interrupted runs may leave an incomplete snapshot file. Recording is
not a checkpoint/restart facility. Existing scalar CSV files cannot be animated
retroactively because they do not contain individual spins.

Animation options:

```bash
# Use only production frames; display every second recorded frame at 15 fps.
python animate_spins.py data/demo.snapshots.npz --production-only --stride 2 \
  --fps 15 --output results/production.gif

# MP4 requires the external FFmpeg executable installed and on PATH.
python animate_spins.py data/demo.snapshots.npz --output results/spins.mp4
```

GIF works without FFmpeg. The animation script refuses to overwrite an existing
output. By default it uses at most 500 frames, selecting evenly across the
chosen frames if necessary. Set `--max-frames 1000` to change this limit; GIF
encoding holds rendered frames in memory. `--fps` controls playback only;
`--stride` skips saved frames without changing any data. Read the sweep labels
for actual sampling intervals, especially at equilibration boundaries.

For comparison, repeat the demonstration at T*=1.5, 2.27 and 3.5, with distinct
output names. Starting randomly lets you observe ordering and domain growth.

## Convergence checks and extra runs

See [CONVERGENCE.md](CONVERGENCE.md) for the ordered command checklist for your
104-run scan. `run_convergence.py` previews or executes focused extra runs;
`check_convergence.py` generates a Markdown report with traces, discard and
block-size sensitivity, and comparisons between starts, seeds and run lengths.
The analysis never launches simulations. All extra runs require `--run`.
