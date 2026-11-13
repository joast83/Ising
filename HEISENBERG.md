# Scalar Ising and vector-spin Heisenberg simulations

The original `ising.py` remains unchanged: scalar +/-1 spins on an L x L square
lattice, with its existing analysis and animation scripts. The separate
`heisenberg.py` uses three-component **classical unit vectors**, on either a
square (`--dim 2`) or simple-cubic (`--dim 3`) lattice with periodic boundaries.
Lattice dimension and spin dimension are different choices. The square lattice
is an ideal 2D periodic layer, not a model of a surface coupled to a bulk substrate.

## Setup and quick examples

```bash
git pull
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

Create `.venv` first with `python3 -m venv .venv` if needed. No new dependencies
beyond the existing NumPy/Matplotlib analysis environment are required.

Your old workflow still works:

```bash
python ising.py --L 16 --T 2.3 --burn 10000 --sweeps 50000 \
  --seed 1 --output data/ising_L16_T2.3.csv
```

A small 3D vector-spin run (4 cubed = 64 spins):

```bash
python heisenberg.py --dim 3 --L 4 --T 1.45 --K 0 \
  --burn 1000 --sweeps 5000 --seed 1 \
  --output data/heisenberg/d3_L4_T1.45_K0_seed1.csv
```

A 2D layer with vector spins (16 squared = 256 spins):

```bash
python heisenberg.py --dim 2 --L 16 --T 1.0 --K 0 \
  --burn 1000 --sweeps 5000 --seed 1 \
  --output data/heisenberg/d2_L16_T1.0_K0_seed1.csv
```

These are exploratory run lengths, not convergence guarantees. Start with L=4
in 3D; L=64 in 3D would contain 262,144 spins. Runtime per sweep scales with
L**dim, and vector updates cost more than scalar flips. This is a readable
NumPy/Python implementation, not an optimized production spin-dynamics code.
There is no physical-time interpretation of sweeps.

## Hamiltonian, signs and proposals

With J>0, T*=kB*T/J and kappa=K/J:

H/J = -sum_<ij> e_i dot e_j - kappa sum_<ij> (e_i dot e_j)^2.

Each pair is counted once. Spins have fixed norm one: no longitudinal
moment-magnitude fluctuations, anisotropy, magnetic field, lattice motion or
quantum effects are included. `--K` supplies **K/J**, including its sign.
Positive K favours collinearity (either sign); negative K favours perpendicular
orientations when considered alone. Bilinear J competes with this tendency.
For K!=0, no K=0 critical-temperature reference is drawn in the plots.

Every sweep has N randomly selected sites with replacement. The local energy
change includes all 2*dim neighbours and both terms. Metropolis accepts with
min(1,exp(-DeltaE/T*)). The default fixed mixture proposes:

- 10% uniformly random directions, obtained by normalizing isotropic Gaussians;
- 10% exact inversion e -> -e;
- 80% normalize(e + step_size * isotropic Gaussian), default step_size=0.5.

All kernels are symmetric under interchange of old and new spins. The local
kernel depends only on their dot product, and the weights/step size never adapt
during a run. This permits the ordinary Metropolis rule without a proposal-ratio
correction. Global proposals maintain access to the full sphere; inversions help
sample sign changes when collinearity is favoured. Use `--proposal uniform` for
uniform-sphere proposals only. Check that observables agree between proposals
once converged. `--step-size` affects efficiency, not the target distribution.

`--start random` and `--start up` (all spins along +z), `--seed`, `--burn`,
`--sweeps`, `--every`, `--output` work as in the Ising workflow. Default burn and
production are 1000 and 5000. Existing output files are never overwritten.
Energy and correlations are evaluated directly at each measurement, avoiding
incremental floating-point energy drift. Norms are preserved by each proposal.

## A first temperature scan

Try L=4 first; add L=6 and then L=8 after estimating runtime:

```bash
for L in 4 6; do
  for T in 0.5 1.0 1.2 1.35 1.45 1.55 1.7 2.0 2.5 3.0; do
    python heisenberg.py --dim 3 --L "$L" --T "$T" --K 0 \
      --burn 1000 --sweeps 5000 --seed 1 \
      --output "data/heisenberg/d3_L${L}_T${T}_K0_seed1.csv"
  done
done
```

If you already ran the standalone L=4,T=1.45 example, that filename exists.
Skip that point or use a separate directory for the scan. The simulator refuses
to overwrite it. Repeat important points with more sweeps, different seeds and
both initial states. Near-transition samples can be strongly correlated.
The existing `run_convergence.py` is for scalar Ising only, not this new model.

Analyze and plot **vector data only**:

```bash
python analyze_heisenberg.py 'data/heisenberg/*.csv' \
  --block-size 100 --output results/heisenberg/summary.csv
python plot_heisenberg.py results/heisenberg/summary.csv
```

Figures are PNG and PDF, one set per lattice dimension and K/J. Different L
have separate markers; replicas remain individual points. Analysis rejects
incomplete runs, inconsistent observables, and scalar Ising files. Do not use
`analyze.py`, `plot_results.py`, `check_convergence.py` or the scalar free-energy
script on vector files. In particular, the spin integration measure and entropy
reference of continuous spins must be specified before extending free-energy work.

To check errors and initial-data sensitivity, rerun the vector analysis:

```bash
python analyze_heisenberg.py 'data/heisenberg/*.csv' \
  --block-size 250 --output results/heisenberg/summary_b250.csv
python analyze_heisenberg.py 'data/heisenberg/*.csv' \
  --block-size 100 --discard 1000 --output results/heisenberg/summary_discard1000.csv
```

Block sizes and discard are in **saved samples**, not always sweeps. Check error
stability over several block sizes while retaining at least roughly 20 blocks.
The block jackknife estimates standard errors; fewer than two blocks gives NaN.
All samples enter central values; only the error calculation drops the incomplete
final block. Independent runs and different starts remain essential checks.

## Definitions and interpretation

The simulator saves sweep, energy per spin/J, mx,my,mz, magnitude |m|, C1, B1,
nematic order and the acceptance fraction in the saved sweep. Here
m=(sum_i e_i)/N. C1 and B1 are nearest-neighbour averages of e_i dot e_j and
(e_i dot e_j)^2. The identity e/J = -dim*(C1+kappa*B1) holds in every sample.

The analysis reports these central values and `_err` columns:

| Quantity | Definition / interpretation |
|---|---|
| m_abs | mean magnitude of vector magnetization |
| chi_x_J, chi_y_J, chi_z_J | N/T* times the variance of that magnetization component |
| chi_iso_J | (chi_x_J+chi_y_J+chi_z_J)/3; response per Cartesian direction for an isotropic system |
| chi_magnitude_J | N/T* times [mean(m dot m)-mean(abs(m))^2]; magnitude-fluctuation diagnostic, not the Cartesian response |
| cv_over_kB | N/T*^2 times variance of energy per spin/J |
| binder | [5-3 mean(abs(m)^4)/mean(abs(m)^2)^2]/2; O(3) normalized Binder cumulant |
| C1, B1 | Mean bilinear and biquadratic nearest-neighbour correlations |
| P2 | (3 B1-1)/2; pair-axis correlation, not global nematic order |
| nematic | Mean largest eigenvalue of Q=(3/2N) sum_i(e_i outer e_i)-(1/2)I |
| acceptance | Mean of the recorded per-sweep acceptance fractions |

For the normalized vector Binder cumulant, the isotropic Gaussian disordered
limit is zero and the fixed-magnitude ordered limit is one. These are asymptotic
limits; finite-cell values need not be exactly zero/one and noisy estimates can
lie outside that interval. Curves for several L can help locate a transition,
but crossings need size and statistical convergence checks. No Tc is fitted
by these scripts.

Global magnetization can rotate at nearly constant magnitude. Cartesian means
and their susceptibility therefore require sufficient orientational sampling;
an apparently stable magnitude alone does not establish that. In an ordered
finite zero-field system global rotation can make the Cartesian susceptibility
large. The magnitude-fluctuation curve is a separately defined diagnostic useful
for studying finite-size peaks. It is not obtained by adding an arbitrary 1/3
factor to the old scalar susceptibility.

For independent isotropic vectors: C1=0, B1=1/3, and high-T energy per spin/J
tends to -dim*kappa/3, not necessarily zero. For independent random +/- spins
along one common axis: C1=0 but B1=1. Thus biquadratic exchange distinguishes
these two zero-magnetization references. For all aligned spins: C1=B1=1 and
energy per spin/J=-dim*(1+kappa).

Q detects collective axis alignment even when positive and negative directions
cancel in magnetization. Its largest eigenvalue is one for perfect collinearity;
it has a positive finite-size bias in random configurations. A large B1 alone
signals local collinearity, not necessarily a common global axis.

## What to expect for K=0

For the simple-cubic classical Heisenberg ferromagnet the established numerical
thermodynamic-limit Tc* is approximately **1.443**, whereas unit-vector mean field
gives z/3=2. These are not the Ising values 2.269 and 4. The plots show the 3D
numerical reference and the appropriate K=0 MF reference. MF magnetization uses
the Langevin function, m=coth(z*m/T*)-T*/(z*m), evaluated with its regular small-x
limit. The MF curve is a symmetry-broken order parameter, not finite-cell mean |m|.

A primary numerical reference is Chen, Ferrenberg and Landau, Phys. Rev. B 48,
3249 (1993), reporting inverse Tc*=0.693035(37) for the simple-cubic lattice:
https://doi.org/10.1103/PhysRevB.48.3249

The isotropic short-range **2D** Heisenberg model has no finite-T ferromagnetic
long-range order in the thermodynamic limit. Finite small cells can still look
strongly ordered. Their rounding and size dependence are therefore interesting
in their own right, not evidence for the 2D Ising transition. See Mermin and
Wagner: https://doi.org/10.1103/PhysRevLett.17.1133

The vector model supports small-angle excitations unavailable to Ising spins.
Consequently the low-T energy and heat-capacity behaviour differ as well as Tc.
Do not interpret the classical fixed-length model as including quantum or
longitudinal-moment effects.

## Explore biquadratic exchange after establishing the K=0 baseline

At one size and temperature, a first comparison is:

```bash
for K in -0.5 0.0 0.5; do
  python heisenberg.py --dim 3 --L 4 --T 1.45 --K "$K" \
    --burn 1000 --sweeps 5000 --seed 2 \
    --output "data/heisenberg/d3_L4_T1.45_K${K}_seed2.csv"
done
```

Then scan T for each K rather than interpreting a single-temperature comparison
as a phase diagram. Inspect magnetization, C1, B1, nematic order and heat capacity
together. Depending on couplings, different kinds of order may appear, so a
single magnetization peak does not identify every possible transition. Existing
K=0 numerical/MF references do not apply at nonzero K. No general K!=0 mean-field
or phase-diagram solver is included.

## Optional snapshots

Add `--snapshot-every 20` to save initial, equilibration-boundary, interval and
final configurations to `<output-stem>.vectors.npz`. Use `--snapshot-output` to
choose a path. The archive has `spins` of shape (frames,L,L,3) or
(frames,L,L,L,3), total `sweep`, and JSON metadata, with the vector model identifier.
Recording snapshots does not alter the trajectory. They accumulate in memory
until the run finishes and are not restart checkpoints. Interrupted files may be
incomplete. Existing scalar animation/spatial scripts do not support this format;
vector arrows or slice animations would be a separate next extension.

## Validation

`python -m unittest -v` tests neighbour/bond counting, ordered-state limits,
rotational invariance, local versus global energy differences for both signs of K,
uniform-sphere sampling, norms and snapshot invariance, susceptibility factors,
jackknife estimates and the K=0 mean-field self-consistency equation.
