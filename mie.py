"""
Mie-scattering particle sizing. Pure math, zero GUI imports — same decoupling
contract as measurement.py.

Premise (from the laser-sheet / MATLAB workflow this ports): the measured area of
a laser-sheet-lit particle blob, expressed in m², is taken to be that particle's
scattering cross-section C_sca [m²]. A forward Mie curve C_sca(a) is computed over
a radius sweep, then inverted per particle (area -> radius). The radii are binned
into a CDF and fit with a Rosin-Rammler model to report D10/D50/D90.

Ported 1:1 from Mie_scattering_MATLAB/b_mie_v2 (calculateMieScattering.m, mie_coeff.m,
sphbes.m, sphankel.m). scipy.special.jv/yv replace MATLAB besselj/bessely and accept
the complex argument z = m*x directly, so the port is numerically faithful.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import jv, yv
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


# --- physical-unit handling ------------------------------------------------

# Factor to convert a *length* in the given unit to metres. Area factor is this
# squared. Keys are lowercased; the micro sign and 'u' both map to micro.
_UNIT_TO_M = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "μm": 1e-6,
    "nm": 1e-9,
}


def area_unit_to_m2_factor(unit: str) -> float | None:
    """Multiplier taking an area in `unit²` to m². None if the unit is unknown.

    e.g. 'mm' -> 1e-6 (mm² -> m²). Matches the MATLAB `areas_mm2 * 1e-6`.
    """
    f = _UNIT_TO_M.get(unit.strip().lower())
    return None if f is None else f * f


# --- spherical Bessel / Hankel (MATLAB sphbes / sphankel) ------------------

def _sphbes(nu: int, x: np.ndarray) -> np.ndarray:
    """Spherical Bessel function of the first kind. Accepts real or complex x."""
    return np.sqrt(np.pi / (2.0 * x)) * jv(nu + 0.5, x)


def _sphankel(nu: int, x: np.ndarray) -> np.ndarray:
    """Spherical Hankel function of the first kind (real x in this workflow)."""
    return np.sqrt(np.pi / (2.0 * x)) * (jv(nu + 0.5, x) + 1j * yv(nu + 0.5, x))


def _mie_coeff(n: int, x: np.ndarray, z: np.ndarray, mu: float, m: complex):
    """Mie coefficients a_n, b_n. Faithful port of mie_coeff.m.

    x = k*a (real, size parameter), z = m*x (complex), m = n_sphere/n_medium.
    """
    jnx, jnx_1 = _sphbes(n, x), _sphbes(n - 1, x)
    hnx, hnx_1 = _sphankel(n, x), _sphankel(n - 1, x)
    jnz, jnz_1 = _sphbes(n, z), _sphbes(n - 1, z)

    # Recurrence for the derivative of x*j_n(x): d/dx[x j_n] = x j_{n-1} - n j_n
    x_jnxp = x * jnx_1 - n * jnx
    z_jnzp = z * jnz_1 - n * jnz
    x_hnxp = x * hnx_1 - n * hnx

    m2 = m * m
    an = (mu * m2 * jnz * x_jnxp - mu * jnx * z_jnzp) / (mu * m2 * jnz * x_hnxp - hnx * z_jnzp)
    bn = (mu * jnz * x_jnxp - mu * jnx * z_jnzp) / (mu * jnz * x_hnxp - hnx * z_jnzp)
    return an, bn


def csca_curve(a: np.ndarray, lam_nm: float, n_medium: float,
               n_sphere: complex, mu: float = 1.0) -> np.ndarray:
    """Forward Mie scattering cross-section C_sca(a) in m².

    a       : radius grid in metres (array)
    lam_nm  : vacuum wavelength in nm
    n_medium: real refractive index of the medium
    n_sphere: complex refractive index of the particle
    mu      : ratio of magnetic permeability (sphere/medium), usually 1
    """
    a = np.asarray(a, dtype=float)
    k = 2.0 * np.pi / (lam_nm * 1e-9) * n_medium
    x = k * a                       # size parameter, real
    m = n_sphere / n_medium         # relative index, complex
    z = m * x                       # complex argument

    x_max = float(np.max(x))
    n_max = int(round(2 + x_max + 4 * x_max ** (1.0 / 3.0)))

    csca = np.zeros_like(a, dtype=float)
    pref = 2.0 * np.pi / (k * k)
    # The Bessel recurrence produces NaN/inf at the near-zero grid edge and can
    # overflow at high order; those points are repaired by the fillmissing guard
    # below, so silence the expected warnings rather than spam the console.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for n in range(1, n_max + 1):
            an, bn = _mie_coeff(n, x, z, mu, m)
            csca += pref * (2 * n + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2)

    # MATLAB fillmissing(..., 'linear', 'extrap'): guard against NaN from the
    # Bessel recurrence at isolated grid points.
    if np.any(~np.isfinite(csca)):
        good = np.isfinite(csca)
        if good.sum() >= 2:
            csca = np.interp(a, a[good], csca[good])
    return csca


def invert_areas(areas_m2: np.ndarray, a: np.ndarray, csca: np.ndarray) -> np.ndarray:
    """Invert measured cross-sections (== areas in m²) to radii via the C_sca curve.

    Linear interpolation with extrapolation, on the sorted-unique C_sca values —
    exactly MATLAB `interp1(unique(Csca), a, areas, 'linear', 'extrap')`. Note the
    inversion is only well-posed where C_sca(a) is monotonic; in the Mie-ripple
    regime unique() keeps the first occurrence and the mapping is approximate.
    """
    csca_u, idx = np.unique(csca, return_index=True)   # sorted ascending, unique
    a_u = np.asarray(a)[idx]
    f = interp1d(csca_u, a_u, kind="linear", fill_value="extrapolate", bounds_error=False)
    return f(np.asarray(areas_m2, dtype=float))


def _rosin_rammler(x, b, c):
    return 1.0 - np.exp(-((x / b) ** c))


@dataclass
class MieResult:
    # inputs echoed for provenance
    lam_nm: float
    n_medium: float
    n_sphere: complex
    mu: float
    a_start: float
    a_stop: float
    # curves
    a: np.ndarray                 # radius grid (m)
    csca: np.ndarray              # C_sca(a) (m²)
    radii: np.ndarray             # per-particle inverted radii (m)
    # Rosin-Rammler fit
    rr_b: float
    rr_c: float
    r2: float
    D10: float
    D50: float
    D90: float
    cdf_bin_centers: np.ndarray = field(default_factory=lambda: np.array([]))
    cdf_values: np.ndarray = field(default_factory=lambda: np.array([]))


def rosin_rammler_fit(radii: np.ndarray, num_bins: int = 50):
    """Bin radii into an empirical CDF and fit the Rosin-Rammler model.

    Returns (b, c, r2, D10, D50, D90, bin_centers, cdf). Ported from
    calculateMieScattering.m (histogram -> cumulative -> fit -> D-values).
    """
    radii = np.asarray(radii, dtype=float)
    radii = radii[np.isfinite(radii)]
    counts, edges = np.histogram(radii, bins=num_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    cdf = np.cumsum(counts) / total if total else np.zeros_like(counts, dtype=float)

    # Prepend the (0, 0) anchor, as the MATLAB does.
    centers = np.concatenate([[0.0], centers])
    cdf = np.concatenate([[0.0], cdf])

    p0 = [max(np.mean(centers), 1e-30), 1.0]
    popt, _ = curve_fit(_rosin_rammler, centers, cdf, p0=p0,
                        bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=20000)
    b, c = float(popt[0]), float(popt[1])

    # R² of the fit on the binned CDF.
    resid = cdf - _rosin_rammler(centers, b, c)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((cdf - np.mean(cdf)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Invert the analytic RR CDF for D-values: D_p = b * (-ln(1-p))^(1/c).
    def d_of(p):
        return b * (-np.log(1.0 - p)) ** (1.0 / c)

    return b, c, r2, float(d_of(0.10)), float(d_of(0.50)), float(d_of(0.90)), centers, cdf


def analyze(areas_m2: np.ndarray, n_real: float, n_imag: float,
            a_start: float, a_stop: float, *, lam_nm: float = 527.0,
            n_medium: float = 1.0, mu: float = 1.0,
            num_a: int = 50000, num_bins: int = 50) -> MieResult:
    """End-to-end: areas (m²) -> radii -> Rosin-Rammler -> D10/D50/D90.

    Mirrors SubmitData.m defaults (527 nm, air, mu=1, 50000-point radius sweep).
    """
    if a_stop <= a_start:
        raise ValueError("a_stop must be greater than a_start")
    areas_m2 = np.asarray(areas_m2, dtype=float)
    if areas_m2.size == 0:
        raise ValueError("no particle areas supplied")

    n_sphere = complex(n_real, n_imag)
    a = np.linspace(a_start, a_stop, num_a)
    csca = csca_curve(a, lam_nm, n_medium, n_sphere, mu)
    radii = invert_areas(areas_m2, a, csca)
    b, c, r2, d10, d50, d90, centers, cdf = rosin_rammler_fit(radii, num_bins)

    return MieResult(
        lam_nm=lam_nm, n_medium=n_medium, n_sphere=n_sphere, mu=mu,
        a_start=a_start, a_stop=a_stop, a=a, csca=csca, radii=radii,
        rr_b=b, rr_c=c, r2=r2, D10=d10, D50=d50, D90=d90,
        cdf_bin_centers=centers, cdf_values=cdf,
    )
