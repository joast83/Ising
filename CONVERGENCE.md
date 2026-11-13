# Convergence workflow for your Ising scan

Your original scan has L=8,16,24,32,40,48,56,64 and 13 temperatures, with
10,000 burn-in and 50,000 production sweeps per run. Let those runs finish.
No snapshots or reruns are needed to analyze their existing CSV files.
Run the following commands from the repository directory.

## 0. Update and install dependencies

```bash
git pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

If you already have `.venv`, activate it and update dependencies; recreating it
is unnecessary. All subsequent commands assume the environment is activated.

## 1. Check existing data before launching more simulations

```bash
python check_convergence.py 'data/L64_T*_seed1.csv' \
  --T 1.0 2.2 2.25 2.3 2.35 2.4 \
  --output-dir results/check_existing
```

Read **results/check_existing/report.md** in VS Code's Markdown preview. Open
its linked plots and CSV tables. By default, every file receives:

- A trace plot of m, |m| and energy, including running averages.
- Error versus block size for mean |m|, mean energy, chi_abs and heat capacity.
- Estimates versus additional discarded rows (0, 5000, 10000, 20000).
- A numerical comparison of first-half and second-half estimates.
- Notes about sparse blocks, missing magnetization signs and error stability.

Blocks are 50,100,200,500,1000,2000,5000,10000 saved samples. Your runs save every
sweep. Block sizes yielding fewer than 20 complete blocks appear as orange crosses. With 50,000 samples, block sizes
up to 2000 retain at least 20 blocks. A plateau seen only after that point is
not well established. Zero errors in a frozen run do not prove convergence.

The discard and between-run plots use a block size of 1000 by default. If the
blocking plots suggest a larger value is needed, rerun with, for example,
`--error-block 2000`. This choice does not change the underlying means.

Incomplete files are skipped and named in the report. Prefer finished runs;
a file still being written cannot yet supply the planned full-length estimate.
Nothing edits the raw data. Rerunning a report replaces its derived outputs.

## 2. Run a focused random-versus-ordered check

First preview the four commands (no simulations are launched):

```bash
python run_convergence.py quick
```

Then execute them serially:

```bash
python run_convergence.py quick --run
```

| Stage | L | T* | Starts | Seeds | Burn | Production | Runs |
|---|---|---|---|---|---|---|---|
| quick | 64 | 2.3 | random, up | 2,3 | 10000 | 50000 | 4 |
| long | 64 | 2.3 | random, up | 5,6 | 50000 | 200000 | 4 |
| extended | 32,64 | 2.25,2.3 | random, up | 2,3,4 | 10000 | 50000 | 24 |

Outputs go to **data/convergence/**, with start, seed, burn-in and production
length included in each filename. These runs do not record spin snapshots.
The same seed across starts provides paired randomness, not independent runs
between the two starts; different seeds provide independent replicas within a
start. The report keeps every run separate and does not pool them.

The runner skips existing complete matching outputs, including the four quick
runs if you later request extended. That means extended adds 20 runs after
quick. If an existing file is partial or mismatched, the runner stops and asks
you to move it aside. It never overwrites a run and does not resume a partial
trajectory. Restarting after moving a partial file restarts that run from scratch.

## 3. Compare the new runs with your original result

```bash
python check_convergence.py 'data/L64_T2.3_seed1.csv' 'data/convergence/*.csv' \
  --L 64 --T 2.3 --output-dir results/check_quick
```

Read **results/check_quick/report.md**. Its run-comparison plot shows each
run separately, labeled by initial state, seed, burn and production length.
Compare the spread between seeds, random/up agreement, and the block errors.
Look for systematic separation between starts, rather than expecting identical
values. Magnetization sign alone is not a failure; focus on |m| and energy.
Fluctuation quantities often need longer runs than the means.

## 4. Test longer runs at the difficult point

```bash
python run_convergence.py long --run
python check_convergence.py 'data/L64_T2.3_seed1.csv' 'data/convergence/*.csv' \
  --L 64 --T 2.3 --error-block 2000 --output-dir results/check_lengths
```

Read **results/check_lengths/report.md**. The longer runs have 50,000 burn-in
and 200,000 production sweeps, so this tests a combined increase in equilibration
and sampling. It does not separately identify which of the two caused any change.
The longer run's block-size curve and discard test help distinguish remaining
sampling uncertainty from initial-condition sensitivity. The 2000-sample block
is a starting choice, not a guarantee that correlations have vanished.

A longer run is a fresh independent trajectory with a different seed, not an
extension of your original CSV. If a different temperature looks worse in your
initial report, prioritize that temperature with a custom `ising.py` command.
The scripts do not claim these proposed lengths guarantee convergence.

## 5. Optional broader check

If you want to check size and temperature dependence more thoroughly:

```bash
python run_convergence.py extended --run
python check_convergence.py 'data/L*_T*_seed1.csv' 'data/convergence/*.csv' \
  --L 32 64 --T 2.25 2.3 --output-dir results/check_extended
```

You can also run diagnostics on the entire completed original scan:

```bash
python check_convergence.py 'data/L*_T*_seed1.csv' \
  --output-dir results/check_all
```

This creates several plots for each of 104 runs, so start with the focused
checks above. Ordinary temperature-dependent thermodynamic plots remain:

```bash
python analyze.py 'data/L*_T*_seed1.csv' --block-size 1000 --output results/summary.csv
python plot_results.py results/summary.csv
```

## How to decide what to do next

| Finding | Next step |
|---|---|
| Persistent trace drift or estimates moving with discard | Increase burn-in, check ordered/random starts, and examine longer trajectories |
| Random and up runs systematically disagree in abs(m) or energy | Increase sampling/equilibration; check for trapped domain patterns |
| Errors grow with block size until few blocks remain | Run longer so large blocks can still be numerous |
| Seeds scatter much more than their error bars suggest | Reassess block size and run length; look for metastability or rare events |
| Starts/seeds/lengths agree and errors show a plausible plateau | Evidence supporting convergence for these observables and parameters, not a proof for the whole scan |

The Markdown report quantifies how errors change between the two largest block
sizes with at least 20 blocks. Its 25% screening threshold is only a heuristic:
read the entire curve. Discard estimates share data and are correlated, so their
differences are not independent significance tests. Reports do not automatically
estimate autocorrelation times or certify convergence.

The comparison plots use chi_abs = N(<m^2>-<|m|>^2)/T*. The signed-m chi is also
saved in all CSV tables. Below Tc, incomplete sampling of magnetization reversals
can strongly affect signed-m chi even when energy and |m| are stable.
