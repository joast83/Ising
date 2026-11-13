# Mean field, magnetic supercells, correlations and free energy

These separate tools extend the existing square-lattice Ising problem. None
changes the simulator or launches simulations implicitly. They use the same
analysis dependencies as before. From the repository directory:

```bash
git pull
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

If needed, create the virtual environment first with `python3 -m venv .venv`.
Keep outputs from different experiments in separate directories: derived files
are replaced on rerun. Raw simulation files are not changed.

## 1. First priority: mean-field curves and magnetic short-range order

Use your existing completed scan:

```bash
python analyze.py 'data/L16_T*_seed1.csv' --block-size 1000 --output results/L16_summary.csv
python compare_mean_field.py results/L16_summary.csv --output-dir results/L16_mean_field
```

For all sizes, change the input glob to `'data/L*_T*_seed1.csv'` and use distinct
summary and output names. Analyze only completed runs. Repeated seeds remain
separate points in the comparison; they are not silently combined.

Outputs:

- `comparison.png` and `.pdf`: MC and mean-field magnetization, energy, C1,
  and heat capacity versus reduced temperature.
- `sro.csv`: original run summaries plus C1, its standard error and MF C1.
- `mean_field.csv`: dense theoretical m, C1, energy, entropy, free energy,
  longitudinal susceptibility and heat capacity curves.

Mean field solves m=tanh(4m/T*) for the stable positive branch. Above T*=4,
m=0. At exactly 4, susceptibility diverges and heat capacity is undefined as
a single two-sided value. The theory table uses infinity and NaN respectively.
The scripts count each nearest-neighbour pair once, use unit spins and J>0,
and report energy per spin in J, heat capacity/entropy per spin in kB.

C1 = mean_<ij>(s_i s_j) = -<E>/(2NJ). Thus C1 needs no new sampling, and its
error is half the existing energy-per-spin error. MF uses C1=m_MF^2.
A paramagnet can have zero macroscopic magnetization but nonzero C1 and energy.
For this Hamiltonian C1 and energy contain the same information: their apparent
agreement is not an independent validation. Longer-range correlations are new
information.

Finite-cell mean |m| and the broken-symmetry MF order parameter are different
observables, particularly above Tc. The susceptibility table is MF response
within a chosen ordered branch, not finite-cell susceptibility including
reversals between signs. It is intentionally not overlaid on the finite-cell
signed-m susceptibility. Do not use MC mean(m)^2 as a substitute for the MF
correlation: symmetry can make mean(m) zero in an ordered finite system.

## 2. Compare paramagnetic reference constructions (cheap)

```bash
python paramagnetic_references.py --L 16 --samples 5000 --seed 1 \
  --output-dir results/references_L16
```

This generates independent random configurations and uniformly random exactly
compensated M=0 configurations (even L required). They have no temperature:
they are reference ensembles, not equilibrium MC samples at finite T.

Outputs are per-configuration `samples.csv`, an ensemble `summary.csv` with
standard errors, `C1_distribution.png`, and `representative.npy`. Expected C1
is 0 for independent spins and -1/(N-1) for exactly compensated random spins.
The latter follows from M^2=0 and permutation symmetry. Small cells make this
global-constraint bias more visible. Use L=4,8,16 to compare sizes cheaply.

`representative.npy` is the sampled compensated configuration with smallest
|C1|. This is a simple one-shell illustration, not an optimized multi-shell SQS.
Selection changes the distribution. A single representative configuration
cannot reproduce thermodynamic fluctuations or determine entropy. Its zero
C1 target differs from the finite-N expectation of the uniformly compensated
ensemble. Compare these references with MC C1 from `sro.csv`.

## 3. Spatial correlations and accessible wavelengths

Full spin configurations are required. Existing snapshots can be used; old
scalar CSVs cannot reconstruct spatial correlations. A modest new example is:

```bash
python ising.py --L 16 --T 2.3 --burn 10000 --sweeps 20000 --seed 7 \
  --snapshot-every 20 --output data/spatial_L16_T2.3.csv
python spatial_correlations.py data/spatial_L16_T2.3.snapshots.npz \
  --output-dir results/spatial_L16_T2.3
```

Only frames with total sweep greater than burn-in are used. Add
`--discard-sweeps 5000` to discard additional production sweeps or `--stride 2`
to use every second retained snapshot. Snapshot files are loaded into memory;
use sparse snapshots and modest lattices initially. Check equilibration and
sampling on the associated scalar CSV before interpreting spatial results.

The script computes raw C(dx,dy)=mean_i,frames(s_i s_(i+r)) and
S(q)=mean_frames(|sum_j s_j exp(i q.r_j)|^2)/N using FFTs. It saves:

- `correlations.npz`: full C and S arrays in unshifted FFT ordering and q values.
- `axial_C.csv`: C averaged over +/-x and +/-y directions (not radial shells).
- `spatial.png`: axial correlations, a displacement map and log10(1+S(q)).
- `notes.txt`: sample count, normalization and limitations.

Checks: C(0)=1, S(0)=N< m^2 > and sum_q S(q)=N. Wavevectors are in units of
1/a, with spacing 2*pi/L. C is not a connected correlation: it can contain an
ordered background. No automatic correlation-length fit is attempted. These
are exploratory means without error bars. Frames can remain correlated; extra
boundary snapshots can produce slightly unequal spacing and are weighted equally.

Repeat at L=8,16,32 and compare at the same T*. Local C1 may converge before
long-distance correlations or S(q). The q grids differ with size; compare
physical wavevectors or distances rather than array indices. Never infer an
infinite correlation length solely from a flat curve in one small cell.

## 4. Exploratory magnetic free energy and entropy

The script integrates d(beta*F/N)/d(beta)=U/N using b=J/(kB*T):

F/(NJ) = [-ln(2) + integral_0^b <E>/(NJ) db']/b,
S/(NkB) = b[<E>/(NJ)-F/(NJ)].

The infinite-temperature reference has entropy ln(2) per spin. The missing
interval from b=0 to 1/Tmax is approximated linearly between energy zero and
the highest-temperature MC energy. This is an approximation, not an exact
finite-temperature reference. Trapezoidal interpolation also introduces error,
especially across a coarse transition-region grid.

Start with one size, L=16. Add high-temperature samples separately:

```bash
for T in 6.0 8.0 10.0 15.0 20.0; do
  python ising.py --L 16 --T "$T" --burn 5000 --sweeps 50000 --seed 1 \
    --output "data/highT/L16_T${T}_seed1.csv"
done
python analyze.py 'data/L16_T*_seed1.csv' 'data/highT/L16_T*_seed1.csv' \
  --block-size 1000 --output results/L16_energy_scan.csv
python integrate_free_energy.py results/L16_energy_scan.csv \
  --output-dir results/L16_free_energy
```

Use the original named scan, not all CSVs in data/, to avoid accidentally
including convergence or snapshot runs at duplicate temperatures. The integration
script rejects multiple rows at the same L,T: select a consistent scan or
combine independent replicas explicitly before integration.

The script normally requires Tmax>=10 for each L. This is a screening threshold,
not a convergence guarantee. `--allow-coarse-reference` permits a demonstration
using the old Tmax=5 scan, with potentially substantial reference bias.
Check higher Tmax and denser temperature grids before interpreting energy versus
entropy compensation or small free-energy differences. Same-seed temperature
runs may have correlated statistical errors; no uncertainty propagation is
provided here.

Outputs are `free_energy.csv`, `free_energy.png` and `integration_notes.txt`.
The figure subtracts the trivial -T*ln(2) free-energy term for readability;
the CSV contains full F/(NJ). The entropy is not clipped to its physical bounds.
Values outside 0 to ln(2) flag numerical or sampling problems.

## Scope

These are controlled Ising-model studies, not quantitative predictions for
Fe-Cr or a replacement for DLM electronic-structure calculations. They separate
sampling, cell-size and mean-field effects within one Hamiltonian. A 3D vector-spin
Heisenberg model, realistic exchange shells and alloy disorder remain future
extensions. They are not added to the current simulator.

Run tests with `python -m unittest -v`. New checks cover MF self-consistency,
heat capacity derivatives, exact compensated-ensemble averages, FFT correlation
normalization and a known free-energy integration curve.
