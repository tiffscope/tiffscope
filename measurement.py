"""Facet thickness measurement geometry.

Pure numpy geometry for measuring thin-film coating thickness on slanted
substrate facets (e.g. an ITO film on KOH-textured silicon, where the
substrate surface is a sawtooth of (111) pyramid facets inclined ~54.74 deg
from horizontal).

Thickness is measured PERPENDICULAR TO THE LOCAL FACET, not vertically:
a vertical measurement overestimates the true film thickness by 1/cos(theta).

The workflow this module supports is two-phase and user-driven:

  Phase A  user clicks N >= 2 points along the film/substrate interface;
           a total-least-squares line is fitted through them.
  Phase B  user clicks M points along the outer film/vacuum surface; for each
           one the perpendicular is dropped onto the fitted interface line and
           the thickness is the length of that perpendicular.

All geometry here operates in IMAGE PIXEL coordinates. Conversion to physical
units happens once, at reporting/export time, via the session's stored scale.

No GUI imports: this module is importable and usable headless.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "fit_line_tls",
    "perpendicular_distance",
    "foot_of_perpendicular",
    "line_angle_deg",
    "line_segment_across_box",
    "facet_angle_hint",
    "FacetMeasurement",
    "FacetSession",
    "sessions_to_dict",
    "sessions_from_dict",
    "sessions_to_json",
    "sessions_from_json",
    "sessions_to_csv",
    "KOH_111_ANGLE_DEG",
]


# Angle between the (111) and (100) planes in cubic silicon: the facet angle
# produced by anisotropic KOH texturing of a (100) wafer.
KOH_111_ANGLE_DEG = math.degrees(math.atan(math.sqrt(2.0)))  # 54.7356...


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def _cross2(direction, vectors):
    """2D cross product direction x vectors.

    direction is (2,), vectors is (2,) or (N, 2). Returns a scalar or (N,).
    Equivalent to |d| |v| sin(angle); with a unit direction this is exactly the
    signed perpendicular offset of each vector from the direction axis.
    """
    v = np.asarray(vectors, dtype=np.float64)
    return direction[0] * v[..., 1] - direction[1] * v[..., 0]


def _canonical_direction(direction):
    """Flip the direction to a canonical half-plane so repeated fits agree.

    Without this the SVD is free to return either +v or -v, which would flip
    the sign of every perpendicular distance between two otherwise identical
    fits. Canonical form: dx > 0, or (dx == 0 and dy > 0).
    """
    if direction[0] < 0 or (direction[0] == 0.0 and direction[1] < 0):
        return -direction
    return direction


def fit_line_tls(points):
    """Total least squares (orthogonal regression) line fit via SVD.

    Minimises the sum of squared PERPENDICULAR distances from the points to
    the line. This is deliberately not numpy.polyfit / ordinary least squares:
    OLS minimises vertical residuals only, and degrades without bound as the
    line approaches vertical. Facet interfaces are steep (~55 deg) and images
    may be rotated, so the fit must be rotation-invariant.

    Parameters
    ----------
    points : array_like, shape (N, 2)
        Points in image pixel coordinates, N >= 2.

    Returns
    -------
    origin : ndarray, shape (2,)
        Centroid of the points; a point on the fitted line.
    direction : ndarray, shape (2,)
        Unit vector along the line, sign-canonicalised.
    rms_residual : float
        Root-mean-square PERPENDICULAR residual, in pixels. For clicked
        interface points this is the user's clicking noise, i.e. the honest
        error bar on the fit.

    Raises
    ------
    ValueError
        If fewer than 2 points are given, the shape is wrong, the points are
        not finite, or every point is identical (degenerate, no direction).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
    if pts.shape[0] < 2:
        raise ValueError("need at least 2 points to fit a line")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points contain non-finite values")

    origin = pts.mean(axis=0)
    centred = pts - origin

    # Right singular vector with the largest singular value = direction of
    # maximum variance = the TLS line direction. The smallest one is the
    # normal, and its singular value carries the orthogonal residual.
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)

    # A vanishing largest singular value means every point is the same point:
    # there is no direction to recover and the SVD's vt is arbitrary.
    scale = max(1.0, float(np.abs(pts).max()))
    if float(sv[0]) <= 1e-12 * scale:
        raise ValueError("degenerate point set: all points coincide")

    direction = np.asarray(vt[0], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("degenerate point set: no line direction")
    direction = _canonical_direction(direction / norm)

    residuals = _cross2(direction, centred)
    rms_residual = float(np.sqrt(np.mean(residuals ** 2)))

    return origin, direction, rms_residual


def perpendicular_distance(point, origin, direction):
    """Signed perpendicular distance from ``point`` to the line.

    Uses the 2D cross product with the unit direction. The SIGN is kept
    deliberately: it tells the caller which side of the interface the point
    lies on, so points clicked on the wrong side can be flagged rather than
    silently folded into the mean by an abs().

    Returns a float in pixels (negative on one side, positive on the other).
    """
    p = np.asarray(point, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    return float(_cross2(d, p - o))


def foot_of_perpendicular(point, origin, direction):
    """Orthogonal projection of ``point`` onto the line, as (x, y) in pixels.

    This is the physically meaningful position along the substrate for the
    measurement: thickness should be reported against the foot's x, not the x
    of the clicked outer point, which is displaced along the facet normal.
    """
    p = np.asarray(point, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    t = float(np.dot(p - o, d))
    return o + t * d


def line_angle_deg(direction):
    """Angle of the line to the horizontal image axis, in degrees.

    Returned in the closed range [0, 90]: a line is undirected, so 10 deg and
    170 deg are the same facet. 0 is horizontal, 90 is vertical.

    The range is closed rather than half-open so that a vertical line reports
    90.0 rather than aliasing onto 0.0 — the vertical case is exactly the one
    that exposes an accidental OLS implementation, and it must stay
    distinguishable from the horizontal case.
    """
    d = np.asarray(direction, dtype=np.float64)
    return float(math.degrees(math.atan2(abs(float(d[1])), abs(float(d[0])))))


def line_segment_across_box(origin, direction, x_min, x_max, y_min, y_max):
    """Endpoints of the fitted line, extended to span the given box.

    Projects the four box corners onto the line and takes the extreme
    parameters, so the returned segment always covers the box while staying
    bounded by its diagonal (an unbounded line would wreck view auto-ranging).

    Returns ((x0, y0), (x1, y1)).
    """
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    corners = np.array(
        [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]],
        dtype=np.float64,
    )
    t = (corners - o) @ d
    p0 = o + float(t.min()) * d
    p1 = o + float(t.max()) * d
    return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))


def facet_angle_hint(angle_deg, tol=2.0):
    """Soft informational hint about a measured facet angle.

    Returns (status, message) where status is "match" when the angle is within
    ``tol`` of the Si (111) plane angle, else "neutral". This is never a
    blocking warning: an off-nominal angle usually just means the cleave is
    tilted or off-centre, which is perfectly measurable.
    """
    delta = abs(float(angle_deg) - KOH_111_ANGLE_DEG)
    if delta <= tol:
        return "match", (
            f"✓ {angle_deg:.2f}° is within {tol:g}° of the Si (111) "
            f"facet angle ({KOH_111_ANGLE_DEG:.2f}°)."
        )
    return "neutral", (
        f"{angle_deg:.2f}° differs from the Si (111) facet angle "
        f"({KOH_111_ANGLE_DEG:.2f}°) by {delta:.2f}° — the cleave may be "
        f"tilted or off-centre. Measurements remain valid."
    )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class FacetMeasurement:
    """One outer-surface point and the perpendicular dropped from it.

    All geometry in image pixels; physical values are derived from the owning
    session's scale. Instances are computed on demand from the session's
    clicked points, so they always reflect the current fit — dragging an
    interface point re-fits the line and every measurement follows.
    """

    outer_x: float
    outer_y: float
    foot_x: float
    foot_y: float
    thickness_px: float
    signed_thickness_px: float
    sign_anomalous: bool = False

    def as_row(self, session):
        """Flat dict of every stored field, including physical units."""
        upp = session.unit_per_px
        row = {
            "session": session.label,
            "source_file": session.source_filename,
            "frame_idx": session.frame_idx,
            "outer_x_px": self.outer_x,
            "outer_y_px": self.outer_y,
            "foot_x_px": self.foot_x,
            "foot_y_px": self.foot_y,
            "thickness_px": self.thickness_px,
            "signed_thickness_px": self.signed_thickness_px,
            "sign_anomalous": self.sign_anomalous,
        }
        if upp is not None:
            row["thickness_scaled"] = self.thickness_px * upp
            row["foot_x_scaled"] = self.foot_x * upp
            row["unit"] = session.unit
        return row


@dataclass
class FacetSession:
    """A named set of clicks on one facet: interface points plus surface points.

    Only the clicked points are stored. The fitted line, the feet of the
    perpendiculars and the thicknesses are all derived, which keeps dragging a
    point and reloading from JSON on exactly the same code path.

    ``scale`` mirrors ``Pipeline.scale``: {"px": int, "value": float,
    "unit": str}, captured at measurement time so a session exported later
    still carries the calibration that was in force when it was measured.
    """

    label: str = "facet"
    interface_points: list = field(default_factory=list)
    surface_points: list = field(default_factory=list)
    source_filename: str = ""
    folder_path: str = ""
    frame_idx: int = 0
    scale: dict | None = None
    color_idx: int = 0
    visible: bool = True
    # 'interface' while clicking phase A, 'surface' once the interface is
    # finished. Persisted so a reloaded session resumes in the right phase.
    phase: str = "interface"
    # Opaque tag describing the display geometry (rotation + crop) in force
    # when the points were clicked. Points are stored in display coordinates,
    # so a later rotate or crop change means they no longer line up with what
    # is on screen; the caller compares tags and shows a warning.
    geometry_tag: str = ""

    # -- scale helpers ---------------------------------------------------

    @property
    def unit_per_px(self):
        """Physical units per pixel, or None when no scale is calibrated."""
        s = self.scale
        if not s:
            return None
        try:
            px = float(s["px"])
            value = float(s["value"])
        except (KeyError, TypeError, ValueError):
            return None
        if px <= 0:
            return None
        return value / px

    @property
    def unit(self):
        s = self.scale
        if not s:
            return "px"
        return str(s.get("unit", "px"))

    def to_units(self, value_px):
        """Convert a pixel length to physical units (identity when unscaled)."""
        upp = self.unit_per_px
        return value_px if upp is None else value_px * upp

    # -- geometry --------------------------------------------------------

    def fit(self):
        """(origin, direction, rms_px) for the interface line, or None.

        None when fewer than 2 interface points have been clicked, or when the
        points are degenerate (all coincident).
        """
        if len(self.interface_points) < 2:
            return None
        try:
            return fit_line_tls(np.asarray(self.interface_points, dtype=np.float64))
        except ValueError:
            return None

    def angle_deg(self):
        fit = self.fit()
        return None if fit is None else line_angle_deg(fit[1])

    def rms_residual_px(self):
        fit = self.fit()
        return None if fit is None else fit[2]

    def measurements(self):
        """Derive a FacetMeasurement for every clicked surface point.

        Empty when the interface line is not yet fitted. The sign-anomaly flag
        marks points on the minority side of the interface: with the film all
        on one side, a minority-sign point is almost always a misclick on the
        substrate side.
        """
        fit = self.fit()
        if fit is None or not self.surface_points:
            return []
        origin, direction, _ = fit

        pts = np.asarray(self.surface_points, dtype=np.float64)
        signed = _cross2(direction, pts - origin)

        # Majority side defines "expected"; ties fall back to the first point.
        n_pos = int(np.sum(signed > 0))
        n_neg = int(np.sum(signed < 0))
        if n_pos > n_neg:
            expected = 1.0
        elif n_neg > n_pos:
            expected = -1.0
        else:
            expected = 1.0 if signed[0] >= 0 else -1.0

        out = []
        for p, sd in zip(pts, signed):
            foot = foot_of_perpendicular(p, origin, direction)
            out.append(FacetMeasurement(
                outer_x=float(p[0]), outer_y=float(p[1]),
                foot_x=float(foot[0]), foot_y=float(foot[1]),
                thickness_px=abs(float(sd)),
                signed_thickness_px=float(sd),
                sign_anomalous=bool(sd != 0.0 and math.copysign(1.0, sd) != expected),
            ))
        return out

    def summary(self):
        """Per-session summary dict for the panel header and the CSV block.

        Thickness stats and the fit RMS are reported in physical units when a
        scale is set, in pixels otherwise. The RMS is the user's clicking
        noise on the interface — the honest error bar on every thickness in
        the session.
        """
        fit = self.fit()
        unit = self.unit
        summary = {
            "label": self.label,
            "source_file": self.source_filename,
            "frame_idx": self.frame_idx,
            "unit": unit,
            "scaled": self.unit_per_px is not None,
            "n_interface": len(self.interface_points),
            "n_surface": len(self.surface_points),
            "angle_deg": None,
            "rms_residual_px": None,
            "rms_residual": None,
            "n": 0,
            "mean": None, "median": None, "std": None, "min": None, "max": None,
            "n_anomalous": 0,
            "hint_status": None,
            "hint": None,
        }
        if fit is not None:
            angle = line_angle_deg(fit[1])
            summary["angle_deg"] = angle
            summary["rms_residual_px"] = fit[2]
            summary["rms_residual"] = self.to_units(fit[2])
            status, message = facet_angle_hint(angle)
            summary["hint_status"] = status
            summary["hint"] = message

        measurements = self.measurements()
        if measurements:
            t = np.array([m.thickness_px for m in measurements], dtype=np.float64)
            t = np.array([self.to_units(v) for v in t], dtype=np.float64)
            summary.update({
                "n": int(t.size),
                "mean": float(t.mean()),
                "median": float(np.median(t)),
                "std": float(t.std(ddof=0)),
                "min": float(t.min()),
                "max": float(t.max()),
                "n_anomalous": int(sum(1 for m in measurements if m.sign_anomalous)),
            })
        return summary

    # -- point editing ---------------------------------------------------

    def add_point(self, x, y):
        """Append a click to whichever phase is active. Returns the phase used."""
        if self.phase == "interface":
            self.interface_points.append([float(x), float(y)])
        else:
            self.surface_points.append([float(x), float(y)])
        return self.phase

    def finish_interface(self):
        """Advance from phase A to phase B. False if fewer than 2 points."""
        if len(self.interface_points) < 2:
            return False
        self.phase = "surface"
        return True

    def undo_last(self):
        """Remove the most recent point of the active phase. True if removed."""
        target = self.interface_points if self.phase == "interface" else self.surface_points
        if not target:
            return False
        target.pop()
        return True

    def move_point(self, kind, index, x, y):
        target = self.interface_points if kind == "interface" else self.surface_points
        if 0 <= index < len(target):
            target[index] = [float(x), float(y)]

    def remove_point(self, kind, index):
        target = self.interface_points if kind == "interface" else self.surface_points
        if 0 <= index < len(target):
            target.pop(index)
            return True
        return False

    def clear(self):
        self.interface_points.clear()
        self.surface_points.clear()
        self.phase = "interface"

    # -- serialization ---------------------------------------------------

    def to_dict(self):
        """JSON-serializable dict. Derived geometry is not stored."""
        return {
            "label": self.label,
            "interface_points": [[float(x), float(y)] for x, y in self.interface_points],
            "surface_points": [[float(x), float(y)] for x, y in self.surface_points],
            "source_filename": self.source_filename,
            "folder_path": self.folder_path,
            "frame_idx": int(self.frame_idx),
            "scale": dict(self.scale) if self.scale else None,
            "color_idx": int(self.color_idx),
            "visible": bool(self.visible),
            "phase": self.phase,
            "geometry_tag": self.geometry_tag,
        }

    @classmethod
    def from_dict(cls, d):
        scale = d.get("scale")
        if not isinstance(scale, dict):
            scale = None
        return cls(
            label=str(d.get("label", "facet")),
            interface_points=[[float(x), float(y)] for x, y in d.get("interface_points", [])],
            surface_points=[[float(x), float(y)] for x, y in d.get("surface_points", [])],
            source_filename=str(d.get("source_filename", "")),
            folder_path=str(d.get("folder_path", "")),
            frame_idx=int(d.get("frame_idx", 0)),
            scale=scale,
            color_idx=int(d.get("color_idx", 0)),
            visible=bool(d.get("visible", True)),
            phase=str(d.get("phase", "interface")),
            geometry_tag=str(d.get("geometry_tag", "")),
        )


# ---------------------------------------------------------------------------
# Session collection I/O
# ---------------------------------------------------------------------------

FACET_JSON_VERSION = 1


def sessions_to_dict(sessions):
    return {
        "type": "tiffscope_facet_sessions",
        "version": FACET_JSON_VERSION,
        "sessions": [s.to_dict() for s in sessions],
    }


def sessions_from_dict(d):
    if not isinstance(d, dict):
        raise ValueError("facet session file must contain a JSON object")
    raw = d.get("sessions")
    if not isinstance(raw, list):
        raise ValueError("missing 'sessions' list")
    return [FacetSession.from_dict(item) for item in raw]


def sessions_to_json(sessions):
    return json.dumps(sessions_to_dict(sessions), indent=2)


def sessions_from_json(s):
    return sessions_from_dict(json.loads(s))


def sessions_to_csv(path, sessions):
    """Write one row per measurement, with a commented per-session header block.

    The commented block carries each session's scale and summary so the CSV is
    self-describing; the rows below are plain CSV that any tool can read.
    """
    sessions = [s for s in sessions if s.measurements()]

    scaled = any(s.unit_per_px is not None for s in sessions)
    units = {s.unit for s in sessions if s.unit_per_px is not None}
    # Name the scaled columns after the unit when it is unambiguous.
    if scaled and len(units) == 1:
        unit_name = units.pop()
        thickness_col = f"thickness_{unit_name}"
        foot_col = f"foot_x_{unit_name}"
    else:
        thickness_col = "thickness_scaled"
        foot_col = "foot_x_scaled"

    columns = [
        "session", "source_file", "frame_idx",
        "outer_x_px", "outer_y_px", "foot_x_px", "foot_y_px",
        "thickness_px", "signed_thickness_px",
    ]
    if scaled:
        columns += [thickness_col, foot_col, "unit"]
    columns.append("sign_anomalous")

    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("# tiffscope facet thickness measurements\n")
        fh.write("# Thickness is measured perpendicular to the fitted facet interface line.\n")
        fh.write("# Position is reported at the foot of the perpendicular, not the clicked point.\n")
        for s in sessions:
            summary = s.summary()
            unit = summary["unit"]
            fh.write("#\n")
            fh.write(f"# session: {s.label}\n")
            fh.write(f"#   source: {s.source_filename or '(unknown)'}  frame_idx: {s.frame_idx}\n")
            if s.scale:
                fh.write(
                    f"#   scale: {s.scale.get('px')} px = {s.scale.get('value')} "
                    f"{s.scale.get('unit')}  ->  1 px = {s.unit_per_px:.6g} {unit}\n"
                )
            else:
                fh.write("#   scale: not calibrated - all values in pixels\n")
            if summary["angle_deg"] is not None:
                fh.write(
                    f"#   facet angle: {summary['angle_deg']:.3f} deg   "
                    f"interface fit RMS residual: {summary['rms_residual']:.4g} {unit} "
                    f"({summary['rms_residual_px']:.4g} px, n={summary['n_interface']} points)\n"
                )
                fh.write(f"#   {summary['hint']}\n")
            if summary["n"]:
                fh.write(
                    f"#   thickness ({unit}): n={summary['n']}  mean={summary['mean']:.6g}  "
                    f"median={summary['median']:.6g}  std={summary['std']:.6g}  "
                    f"min={summary['min']:.6g}  max={summary['max']:.6g}\n"
                )
            if summary["n_anomalous"]:
                fh.write(
                    f"#   NOTE: {summary['n_anomalous']} point(s) flagged on the "
                    f"minority side of the interface\n"
                )
        fh.write("#\n")

        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for s in sessions:
            for m in s.measurements():
                row = m.as_row(s)
                if scaled:
                    # Sessions without a scale leave the scaled columns blank.
                    row.setdefault("thickness_scaled", "")
                    row.setdefault("foot_x_scaled", "")
                    row.setdefault("unit", "px")
                    row[thickness_col] = row.pop("thickness_scaled")
                    row[foot_col] = row.pop("foot_x_scaled")
                writer.writerow(row)
    return len(sessions)
