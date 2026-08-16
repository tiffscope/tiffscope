"""
Mie-scattering particle sizing. Pure math, zero GUI imports — same decoupling
contract as measurement.py.

Premise (from the laser-sheet / MATLAB workflow this ports): the measured area of
a laser-sheet-lit particle blob, expressed in m², is taken to be that particle's
scattering cross-section C_sca [m²]. A forward Mie curve C_sca(a) is computed over
a radius sweep, then inverted per particle (area -> radius). Radii are doubled to
diameters, binned into a CDF, and fit with a Rosin-Rammler model to report
D10/D50/D90.

Ported 1:1 from Mie_scattering_MATLAB/b_mie_v2 (calculateMieScattering.m, mie_coeff.m,
sphbes.m, sphankel.m). scipy.special.jv/yv replace MATLAB besselj/bessely and accept
the complex argument z = m*x directly, so the port is numerically faithful.

D-VALUES ARE DIAMETERS. The .m files stop at radius: they interpolate D10/D50/D90
on the radius grid and plot them against an axis labelled "Radius [m]". In the
original workflow the doubling happened afterwards, in a separate Python step
that exported the MATLAB results and rebuilt the CDF on 2*r, so the published
numbers are diameters. That step lives here now. tiffscope's D-values are 2x
what the .m files alone produce and agree with the publication.

Everything upstream of the reporting boundary still works in radius: the Mie
size parameter is x = k*a, the sweep is a radius sweep, and `MieResult.radii`
is the raw inversion output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import jv, yv
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


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

class MieCancelled(Exception):
    """Raised when a progress_cb asks csca_curve to stop. Distinct from a real
    error so callers can treat a user cancel as a normal outcome."""


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
               n_sphere: complex, mu: float = 1.0,
               progress_cb=None) -> np.ndarray:
    """Forward Mie scattering cross-section C_sca(a) in m².

    a       : radius grid in metres (array)
    lam_nm  : vacuum wavelength in nm
    n_medium: real refractive index of the medium
    n_sphere: complex refractive index of the particle
    mu      : ratio of magnetic permeability (sphere/medium), usually 1
    progress_cb: optional ``f(order_done, order_total) -> bool``. Called once per
              Mie order — the only natural chunk boundary in this sum, and the
              loop dominates runtime, so it is where a determinate progress bar
              and a cooperative cancel both belong. Return False to abort; the
              partial sum is discarded and MieCancelled is raised. Deliberately a
              plain callable, not a Qt signal — mie.py stays GUI-free.
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
    # n_max is set by the *largest* radius. At small-radius grid points those
    # high orders (n >> x) are physically negligible, but the Bessel functions
    # of order n+1/2 overflow there (yv -> ±inf, giving inf-inf = NaN in the
    # coefficients), which would poison the whole point. Add only the finite
    # per-term contributions and drop the non-finite ones (their true magnitude
    # is ~0). Without this, a larger a_stop silently corrupts C_sca at small a
    # into a flat floor hundreds of times too high — see the fixed regression.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for n in range(1, n_max + 1):
            an, bn = _mie_coeff(n, x, z, mu, m)
            term = pref * (2 * n + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2)
            np.add(csca, term, out=csca, where=np.isfinite(term))
            if progress_cb is not None and progress_cb(n, n_max) is False:
                raise MieCancelled("Mie sweep cancelled")

    # Guard against any residual NaN (e.g. the near-zero grid edge). MATLAB used
    # fillmissing(..., 'linear', 'extrap') here.
    if np.any(~np.isfinite(csca)):
        good = np.isfinite(csca)
        if good.sum() >= 2:
            csca = np.interp(a, a[good], csca[good])
    return csca


def _monotone_trend(a: np.ndarray, csca: np.ndarray, ripple_da: float,
                    ripple_periods: float = 4.0) -> np.ndarray:
    """Smooth C_sca(a) to its monotone-increasing mean trend.

    C_sca(a) oscillates (Mie ripples) about a monotone trend, spacing Δa≈ripple_da
    in radius. Averaging over a few ripple periods removes the ripples; the window
    is defined in radius so it scales with grid density and the result is
    grid-independent. `maximum.accumulate` then guarantees strict monotonicity so
    the inversion is single-valued.
    """
    da = (a[-1] - a[0]) / (len(a) - 1)
    window = max(int(round(ripple_periods * ripple_da / da)), 1)
    # If the whole sweep spans less than one smoothing window, the range is
    # sub-ripple (e.g. Rayleigh regime) — no ripples to average, and smoothing
    # would flatten the curve to a constant. Only smooth when there is ripple
    # structure to remove; either way enforce monotonicity for the inversion.
    if window < len(csca):
        csca = uniform_filter1d(csca, size=window, mode="nearest")
    return np.maximum.accumulate(csca)


def invert_areas(areas_m2: np.ndarray, a: np.ndarray, csca: np.ndarray,
                 method: str = "trend", ripple_da: float | None = None) -> np.ndarray:
    """Invert measured cross-sections (== areas in m²) to radii via the C_sca curve.

    method='trend' (default): invert the smooth monotone trend of C_sca(a). The
    raw curve is non-monotonic in the Mie-ripple regime, where a cross-section maps
    to several radii; inverting the trend gives the unbiased, **grid-independent**
    mean-trend radius. Requires `ripple_da` (≈ λ / (2·n_medium)).

    method='legacy': the original MATLAB `interp1(unique(Csca), a, ...)`. Sorting by
    C_sca scrambles the a-order into a zigzag whose surviving points depend on the
    grid, so results are grid-sensitive. Kept only to reproduce prior MATLAB runs.
    """
    csca = np.asarray(csca, dtype=float)
    if method == "trend":
        if ripple_da is None:
            raise ValueError("method='trend' requires ripple_da")
        csca = _monotone_trend(np.asarray(a, dtype=float), csca, ripple_da)
    elif method != "legacy":
        raise ValueError(f"unknown inversion method: {method!r}")

    a = np.asarray(a, dtype=float)
    csca_u, idx = np.unique(csca, return_index=True)   # sorted ascending, unique
    a_u = a[idx]
    f = interp1d(csca_u, a_u, kind="linear", fill_value="extrapolate", bounds_error=False)
    radii = f(np.asarray(areas_m2, dtype=float))
    # Clamp to the physical sweep range. Cross-sections outside [csca.min, csca.max]
    # would otherwise linear-extrapolate off the ends — below the curve minimum this
    # produces *negative* (unphysical) radii. A radius pinned at a[0]/a[-1] signals
    # the sweep range does not cover the data (widen it / re-Auto-range).
    return np.clip(radii, a.min(), a.max())


def _rosin_rammler(x, b, c):
    # Clip the exponent: (x/b)**c overflows float64 for large c or tiny b during
    # curve_fit's search, producing inf -> NaN residuals that abort the fit.
    # exp(-700) is already 0 to machine precision, so clipping changes nothing.
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        z = np.clip((x / b) ** c, 0.0, 700.0)
        return 1.0 - np.exp(-z)


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
    diameters: np.ndarray         # per-particle diameters (m) == 2 * radii
    # Rosin-Rammler fit — fitted on DIAMETERS, so rr_b is a diameter scale.
    rr_b: float
    rr_c: float
    r2: float
    # D-values are DIAMETERS (m), the field convention. RR-fit when the fit
    # converged, else the empirical percentiles below.
    #
    # The .m files stop at radius (calculateMieScattering.m:
    # interp1(fit_curve, a_unique, 0.5)) and label the plot axis "Radius [m]".
    # The doubling was done afterwards, outside MATLAB, when the results were
    # exported and the CDF rebuilt on 2*r in Python. Folding that step in here
    # is what makes these agree with the published values, and it is also the
    # only reading that matches universal usage, where D means diameter.
    # `radii` keeps the raw inversion output, and the C_sca curve stays a
    # function of radius (the Mie size parameter x = k*a is defined on radius).
    D10: float
    D50: float
    D90: float
    # Empirical percentile D-values (diameters) — always finite for n >= 1.
    # Distribution-free, so they stay meaningful when the sample is too small
    # for a Rosin-Rammler fit.
    emp_D10: float = float("nan")
    emp_D50: float = float("nan")
    emp_D90: float = float("nan")
    rr_ok: bool = True   # False when the RR fit failed and D-values are empirical
    cdf_bin_centers: np.ndarray = field(default_factory=lambda: np.array([]))
    cdf_values: np.ndarray = field(default_factory=lambda: np.array([]))


def rosin_rammler_fit(sizes: np.ndarray, num_bins: int = 50):
    """Bin sizes into an empirical CDF and fit the Rosin-Rammler model.

    `sizes` is whatever quantity the D-values should be reported in. `analyze`
    passes DIAMETERS, so b and the returned D-values are diameters. The model
    is a scale family, so feeding radii instead returns exactly half of each
    (and the same c and R²).

    Returns (b, c, r2, D10, D50, D90, bin_centers, cdf). Structure ported from
    calculateMieScattering.m (histogram -> cumulative -> fit -> D-values).
    """
    sizes = np.asarray(sizes, dtype=float)
    sizes = sizes[np.isfinite(sizes)]
    counts, edges = np.histogram(sizes, bins=num_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    cdf = np.cumsum(counts) / total if total else np.zeros_like(counts, dtype=float)

    # Prepend the (0, 0) anchor, as the MATLAB does.
    centers = np.concatenate([[0.0], centers])
    cdf = np.concatenate([[0.0], cdf])

    # Seed b at the median radius (robust scale estimate) rather than the mean of
    # the bin centres — the latter is pulled toward empty high-radius bins for
    # small samples and gives curve_fit a poor start.
    med = float(np.median(sizes)) if sizes.size else 0.0
    p0 = [max(med, 1e-30), 1.0]
    try:
        popt, _ = curve_fit(_rosin_rammler, centers, cdf, p0=p0,
                            bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=20000)
        b, c = float(popt[0]), float(popt[1])
        resid = cdf - _rosin_rammler(centers, b, c)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((cdf - np.mean(cdf)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except (RuntimeError, ValueError, TypeError):
        # Fit did not converge (common when too few particles give a coarse,
        # jumpy CDF). Signal failure with NaNs; the caller falls back to
        # empirical percentiles so the user still gets D-values.
        b = c = r2 = float("nan")

    # Invert the analytic RR CDF for D-values: D_p = b * (-ln(1-p))^(1/c).
    def d_of(p):
        if not (np.isfinite(b) and np.isfinite(c) and b > 0 and c > 0):
            return float("nan")
        return b * (-np.log(1.0 - p)) ** (1.0 / c)

    return b, c, r2, float(d_of(0.10)), float(d_of(0.50)), float(d_of(0.90)), centers, cdf


def analyze(areas_m2: np.ndarray, n_real: float, n_imag: float,
            a_start: float, a_stop: float, *, lam_nm: float = 527.0,
            n_medium: float = 1.0, mu: float = 1.0,
            num_a: int = 50000, num_bins: int = 50,
            inversion: str = "trend", progress_cb=None) -> MieResult:
    """End-to-end: areas (m²) -> radii -> diameters -> Rosin-Rammler -> D10/D50/D90.

    D10/D50/D90 are DIAMETERS, which is what D means in particle sizing and what
    the published values are. The .m files stop at radius and were doubled in a
    separate post-processing step; that step is folded in here. See MieResult.

    Mirrors SubmitData.m defaults (527 nm, air, mu=1, 50000-point radius sweep).
    `inversion` selects the area->radius map: 'trend' (default, grid-stable) or
    'legacy' (original MATLAB unique(); grid-sensitive). See invert_areas.
    """
    if a_stop <= a_start:
        raise ValueError("a_stop must be greater than a_start")
    areas_m2 = np.asarray(areas_m2, dtype=float)
    if areas_m2.size == 0:
        raise ValueError("no particle areas supplied")

    n_sphere = complex(n_real, n_imag)
    a = np.linspace(a_start, a_stop, num_a)
    csca = csca_curve(a, lam_nm, n_medium, n_sphere, mu, progress_cb=progress_cb)
    ripple_da = lam_nm * 1e-9 / (2.0 * n_medium)   # Δa per Mie ripple
    radii = invert_areas(areas_m2, a, csca, method=inversion, ripple_da=ripple_da)

    # D-values are reported as DIAMETERS. The inversion is defined on radius
    # (the Mie size parameter is x = k*a), so the conversion happens here, once,
    # at the reporting boundary — the same place the original workflow did it,
    # just no longer by hand. See MieResult.
    diameters = 2.0 * radii
    b, c, r2, d10, d50, d90, centers, cdf = rosin_rammler_fit(diameters, num_bins)

    # Empirical percentile D-values — distribution-free, always finite for n >= 1.
    # For small samples that defeat the RR fit these are the usable result.
    df = diameters[np.isfinite(diameters) & (diameters > 0)]
    if df.size:
        e10, e50, e90 = (float(np.percentile(df, p)) for p in (10, 50, 90))
    else:
        e10 = e50 = e90 = float("nan")

    # Report the RR-fit D-values when the fit converged; otherwise fall back to
    # the empirical percentiles so the user always gets something meaningful.
    rr_ok = np.isfinite(d50)
    D10, D50, D90 = (d10, d50, d90) if rr_ok else (e10, e50, e90)

    return MieResult(
        lam_nm=lam_nm, n_medium=n_medium, n_sphere=n_sphere, mu=mu,
        a_start=a_start, a_stop=a_stop, a=a, csca=csca, radii=radii,
        diameters=diameters,
        rr_b=b, rr_c=c, r2=r2, D10=D10, D50=D50, D90=D90,
        emp_D10=e10, emp_D50=e50, emp_D90=e90, rr_ok=bool(rr_ok),
        cdf_bin_centers=centers, cdf_values=cdf,
    )
