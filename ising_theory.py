"""Analytic references for the square-lattice, zero-field Ising model (J>0)."""
import math
import numpy as np


def mean_field(T):
    """Positive equilibrium branch; energies per spin/J, entropy and c per kB.

    chi_J is longitudinal within a symmetry-broken branch below Tc, not the
    finite-cell susceptibility including reversals between +/- magnetization.
    At exactly Tc the susceptibility diverges and c has unequal one-sided limits.
    """
    if not math.isfinite(T) or T <= 0:
        raise ValueError('temperature must be finite and positive')
    m = 0.0
    if T < 4:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo+hi)/2
            if math.tanh(4*mid/T) > mid:
                lo = mid
            else:
                hi = mid
        m = (lo+hi)/2
    p = (1+m)/2
    entropy = -sum(x*math.log(x) for x in (p, 1-p) if x > 0)
    energy = -2*m*m
    d = 1-4*(1-m*m)/T
    chi = math.inf if T == 4 else (1-m*m)/(T*d)
    cv = math.nan if T == 4 else 16*m*m*(1-m*m)/(T*T*d)
    return dict(T=T, m=m, C1=m*m, e_over_J=energy, s_over_kB=entropy,
                f_over_J=energy-T*entropy, chi_J=chi, cv_over_kB=cv)


def integrate_energy(T, energy):
    """Trapezoidal beta integration with (beta,u)=(0,0) anchor.

    The unsampled high-T interval is approximated linearly. This is not an
    exact finite-T free-energy reference and convergence in Tmax is required.
    """
    T, energy = np.asarray(T, float), np.asarray(energy, float)
    if len(T) < 2 or len(T) != len(energy) or np.any(T <= 0):
        raise ValueError('need at least two temperatures and matching energies')
    if not np.all(np.isfinite(T)) or not np.all(np.isfinite(energy)):
        raise ValueError('nonfinite integration input')
    order = np.argsort(1/T)
    b, u = 1/T[order], energy[order]
    if np.any(np.diff(b) <= 0):
        raise ValueError('temperatures must be distinct')
    bb, uu = np.r_[0,b], np.r_[0,u]
    integral = np.cumsum(np.diff(bb)*(uu[1:]+uu[:-1])/2)
    f = (-math.log(2)+integral)/b
    return T[order], u, f, (u-f)*b
