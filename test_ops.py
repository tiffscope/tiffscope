"""
Regression tests for operations.py.
Run: .venv/bin/python3 test_ops.py
Tests use real TIFFs from test_imgs/ where available; synthetic arrays otherwise.
"""
import sys
import traceback
import numpy as np
import tifffile
import os

sys.path.insert(0, os.path.dirname(__file__))
from operations import (
    RotateOp, CropOp, BgSubtractOp, AdaptiveThresholdOp,
    CLAHEOp, GaussianBlurOp, SharpenOp, operation_from_dict, OPERATION_REGISTRY
)
from pipeline import Pipeline

PASS = 0
FAIL = 0

def ok(name):
    global PASS
    PASS += 1
    print(f"  PASS  {name}")

def fail(name, reason):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name}: {reason}")

def check(name, cond, reason=""):
    if cond:
        ok(name)
    else:
        fail(name, reason or "condition false")

def no_crash(name, fn):
    try:
        fn()
        ok(name)
    except Exception as e:
        fail(name, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_IMG_DIR = os.path.join(os.path.dirname(__file__), "test_imgs")

def load_real(idx=0) -> np.ndarray:
    files = sorted(f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".tif"))
    return tifffile.imread(os.path.join(TEST_IMG_DIR, files[idx]))

def synth_u16(h=128, w=160) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 4096, size=(h, w), dtype=np.uint16)

def synth_u8(h=128, w=160) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(h, w), dtype=np.uint8)

def rotated_crop(frame) -> np.ndarray:
    """Simulate RotateOp + CropOp output — non-contiguous array."""
    rotated = np.rot90(frame, k=-1)          # non-contiguous view
    h, w = rotated.shape
    sliced = rotated[10:h-10, 10:w-10]       # non-contiguous slice
    assert not sliced.flags['C_CONTIGUOUS'], "expected non-contiguous"
    return sliced


# ---------------------------------------------------------------------------
# Section 1: contiguity — the class of bugs that caused segfaults
# ---------------------------------------------------------------------------
print("\n=== Contiguity (segfault regression) ===")

def _test_gaussianblur_noncontiguous():
    frame = rotated_crop(synth_u16())
    op = GaussianBlurOp()
    result = op.apply(frame)
    assert result.shape == frame.shape
    assert result.dtype == frame.dtype

def _test_gaussianblur_noncontiguous_u8():
    frame = rotated_crop(synth_u8())
    op = GaussianBlurOp()
    result = op.apply(frame)
    assert result.shape == frame.shape

def _test_clahe_noncontiguous_u8():
    frame = rotated_crop(synth_u8())
    op = CLAHEOp()
    result = op.apply(frame)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8

def _test_clahe_noncontiguous_u16():
    frame = rotated_crop(synth_u16())
    op = CLAHEOp()
    result = op.apply(frame)
    assert result.shape == frame.shape

def _test_sharpen_noncontiguous():
    frame = rotated_crop(synth_u16())
    op = SharpenOp()
    result = op.apply(frame)
    assert result.shape == frame.shape

def _test_adaptive_noncontiguous():
    frame = rotated_crop(synth_u16())
    op = AdaptiveThresholdOp()
    result = op.apply(frame)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8

no_crash("GaussianBlurOp — non-contiguous uint16", _test_gaussianblur_noncontiguous)
no_crash("GaussianBlurOp — non-contiguous uint8", _test_gaussianblur_noncontiguous_u8)
no_crash("CLAHEOp — non-contiguous uint8", _test_clahe_noncontiguous_u8)
no_crash("CLAHEOp — non-contiguous uint16", _test_clahe_noncontiguous_u16)
no_crash("SharpenOp — non-contiguous uint16", _test_sharpen_noncontiguous)
no_crash("AdaptiveThresholdOp — non-contiguous uint16", _test_adaptive_noncontiguous)


# ---------------------------------------------------------------------------
# Section 2: full pipeline with real frames, simulating crash scenario
# ---------------------------------------------------------------------------
print("\n=== Full pipeline (Rotate→Crop→Sharpen→GaussianBlur→BgSubtract) ===")

def _test_full_pipeline_no_bg():
    """No background fitted — BgSubtractOp passthrough."""
    p = Pipeline()
    rotate = RotateOp(); rotate.params["k"] = 1
    crop = CropOp(); crop.params.update({"x": 10, "y": 10, "w": 100, "h": 80})
    sharpen = SharpenOp()
    blur = GaussianBlurOp()
    bgsub = BgSubtractOp()
    for op in (rotate, crop, sharpen, blur, bgsub):
        p.add_operation(op)
    frame = load_real(0)
    result = p.apply_to_frame(frame, 0)
    assert result is not None
    assert result.ndim == 2

def _test_full_pipeline_sharpen_disabled():
    """Sharpen disabled → GaussianBlur receives CropOp's non-contiguous slice."""
    p = Pipeline()
    rotate = RotateOp(); rotate.params["k"] = 1
    crop = CropOp(); crop.params.update({"x": 10, "y": 10, "w": 100, "h": 80})
    sharpen = SharpenOp()
    blur = GaussianBlurOp()
    for op in (rotate, crop, sharpen, blur):
        p.add_operation(op)
    p.set_enabled(2, False)  # disable SharpenOp (index 2)
    frame = load_real(0)
    result = p.apply_to_frame(frame, 0)
    assert result is not None

def _test_full_pipeline_clahe_after_crop():
    """CLAHE receives non-contiguous crop output."""
    p = Pipeline()
    crop = CropOp(); crop.params.update({"x": 5, "y": 5, "w": 120, "h": 90})
    clahe = CLAHEOp()
    for op in (crop, clahe):
        p.add_operation(op)
    frame = load_real(0)
    result = p.apply_to_frame(frame, 0)
    assert result is not None

no_crash("full pipeline — BgSubtract passthrough", _test_full_pipeline_no_bg)
no_crash("full pipeline — Sharpen disabled, GaussianBlur gets crop slice", _test_full_pipeline_sharpen_disabled)
no_crash("full pipeline — CLAHE after CropOp", _test_full_pipeline_clahe_after_crop)


# ---------------------------------------------------------------------------
# Section 3: basic correctness
# ---------------------------------------------------------------------------
print("\n=== Basic correctness ===")

def _rotate_k0():
    f = synth_u16()
    op = RotateOp(); op.params["k"] = 0
    assert op.apply(f) is f

def _rotate_roundtrip():
    f = synth_u16()
    op = RotateOp(); op.params["k"] = 1
    r = op.apply(f)
    op.params["k"] = 3
    back = op.apply(r)
    assert np.array_equal(f, back)

def _crop_clamps():
    f = synth_u16(64, 64)
    op = CropOp(); op.params.update({"x": 60, "y": 60, "w": 100, "h": 100})
    r = op.apply(f)
    assert r.shape == (4, 4)

def _bgsub_passthrough_when_no_bg():
    f = synth_u16()
    op = BgSubtractOp()
    assert op._background is None
    r = op.apply(f)
    assert np.array_equal(r, f)

def _bgsub_shape_mismatch_passthrough():
    f = synth_u16(64, 64)
    op = BgSubtractOp()
    op.set_background(np.zeros((32, 32), dtype=np.float32))
    r = op.apply(f)
    assert np.array_equal(r, f)

def _bgsub_subtracts():
    rng = np.random.default_rng(0)
    f = rng.integers(100, 200, (32, 32), dtype=np.uint16)
    bg = np.full((32, 32), 50, dtype=np.float32)
    op = BgSubtractOp()
    op.set_background(bg)
    r = op.apply(f)
    assert r.dtype == np.uint16
    assert r.min() >= 50  # all values were >= 100, minus 50 = >= 50

def _adaptive_output_dtype():
    f = synth_u16()
    op = AdaptiveThresholdOp()
    r = op.apply(f)
    assert r.dtype == np.uint8
    assert set(np.unique(r)).issubset({0, 255})

def _clahe_constant_passthrough():
    f = np.full((32, 32), 1000, dtype=np.uint16)
    op = CLAHEOp()
    r = op.apply(f)
    assert np.array_equal(r, f)

def _gaussianblur_dtype_preserve():
    for dtype in (np.uint8, np.uint16):
        f = synth_u16().astype(dtype)
        op = GaussianBlurOp()
        r = op.apply(f)
        assert r.dtype == dtype, f"dtype mismatch for {dtype}"

def _sharpen_dtype_preserve():
    for dtype in (np.uint8, np.uint16):
        f = synth_u16().astype(dtype)
        op = SharpenOp()
        r = op.apply(f)
        assert r.dtype == dtype

no_crash("RotateOp k=0 noop", _rotate_k0)
no_crash("RotateOp k=1 then k=3 roundtrip", _rotate_roundtrip)
no_crash("CropOp clamps to frame bounds", _crop_clamps)
no_crash("BgSubtractOp passthrough when no bg", _bgsub_passthrough_when_no_bg)
no_crash("BgSubtractOp passthrough on shape mismatch", _bgsub_shape_mismatch_passthrough)
no_crash("BgSubtractOp subtracts correctly", _bgsub_subtracts)
no_crash("AdaptiveThresholdOp output is uint8 binary", _adaptive_output_dtype)
no_crash("CLAHEOp constant frame passthrough", _clahe_constant_passthrough)
no_crash("GaussianBlurOp preserves dtype", _gaussianblur_dtype_preserve)
no_crash("SharpenOp preserves dtype", _sharpen_dtype_preserve)


# ---------------------------------------------------------------------------
# Section 4: serialization round-trip
# ---------------------------------------------------------------------------
print("\n=== Serialization ===")

def _roundtrip(cls, extra_params=None):
    op = cls()
    if extra_params:
        op.params.update(extra_params)
    d = op.to_dict()
    op2 = operation_from_dict(d)
    assert op2 is not None
    assert op2.params == op.params
    assert op2.enabled == op.enabled

no_crash("RotateOp roundtrip", lambda: _roundtrip(RotateOp, {"k": 2}))
no_crash("CropOp roundtrip", lambda: _roundtrip(CropOp, {"x": 5, "y": 10, "w": 50, "h": 60}))
no_crash("BgSubtractOp roundtrip", lambda: _roundtrip(BgSubtractOp, {"method": "mean", "sample_n": 20}))
no_crash("AdaptiveThresholdOp roundtrip", lambda: _roundtrip(AdaptiveThresholdOp))
no_crash("CLAHEOp roundtrip", lambda: _roundtrip(CLAHEOp, {"clip_limit": 4, "tile_size": 16}))
no_crash("GaussianBlurOp roundtrip", lambda: _roundtrip(GaussianBlurOp, {"kernel_size": 5, "sigma": 2}))
no_crash("SharpenOp roundtrip", lambda: _roundtrip(SharpenOp, {"amount": 3}))


# ---------------------------------------------------------------------------
# Section 5: pipeline cache invalidation
# ---------------------------------------------------------------------------
print("\n=== Pipeline cache invalidation ===")

def _cache_invalidated_on_param_change():
    p = Pipeline()
    op = GaussianBlurOp()
    p.add_operation(op)
    frame = synth_u16()
    p.apply_to_frame(frame, 0)
    p.update_params(0, {"kernel_size": 7})
    assert p._caches[0] == {}

def _downstream_cache_cleared():
    p = Pipeline()
    op1 = GaussianBlurOp()
    op2 = SharpenOp()
    p.add_operation(op1)
    p.add_operation(op2)
    frame = synth_u16()
    p.apply_to_frame(frame, 0)
    p._invalidate_from(0)
    assert p._caches[0] == {}
    assert p._caches[1] == {}

def _disabled_op_is_passthrough():
    p = Pipeline()
    op = GaussianBlurOp()
    p.add_operation(op)
    p.set_enabled(0, False)
    frame = synth_u16()
    result = p.apply_to_frame(frame, 0)
    assert np.array_equal(result, frame)

no_crash("cache cleared on param update", _cache_invalidated_on_param_change)
no_crash("downstream caches cleared on invalidate_from", _downstream_cache_cleared)
no_crash("disabled op is passthrough", _disabled_op_is_passthrough)


# ---------------------------------------------------------------------------
# Section 6: real frames through all ops
# ---------------------------------------------------------------------------
print("\n=== Real frames ===")

def _all_ops_on_real_frame():
    frame = load_real(0)
    ops = [
        RotateOp(),
        (lambda o: (o.params.update({"x": 0, "y": 0, "w": 200, "h": 200}), o)[1])(CropOp()),
        CLAHEOp(),
        GaussianBlurOp(),
        SharpenOp(),
        AdaptiveThresholdOp(),
    ]
    f = frame
    for op in ops:
        f = op.apply(f)
    assert f is not None

no_crash("all ops chained on real frame", _all_ops_on_real_frame)


# ---------------------------------------------------------------------------
# Section 7: facet thickness geometry (measurement.py)
#
# Pure synthetic geometry — no images needed. The whole point of this section
# is that the line fit is total least squares: every test below is run at a
# range of facet angles including 90 deg (vertical), which an accidental OLS
# implementation cannot survive.
# ---------------------------------------------------------------------------
print("\n=== Facet thickness geometry ===")

from measurement import (
    fit_line_tls, perpendicular_distance, foot_of_perpendicular,
    line_angle_deg, facet_angle_hint, FacetSession,
    sessions_to_json, sessions_from_json, KOH_111_ANGLE_DEG,
)

TEST_ANGLES = [0.0, 30.0, 54.74, 80.0, 90.0]


def _line_points(angle_deg, n=15, span=100.0, origin=(320.0, 240.0)):
    """n points evenly spaced along a line through origin at angle_deg."""
    th = np.radians(angle_deg)
    d = np.array([np.cos(th), np.sin(th)])
    t = np.linspace(-span / 2, span / 2, n)
    return np.asarray(origin, dtype=np.float64) + t[:, None] * d, d


def _normal(d):
    return np.array([-d[1], d[0]])


def _rotate(points, angle_deg, centre=(0.0, 0.0)):
    th = np.radians(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.asarray(centre, dtype=np.float64)
    return (np.asarray(points, dtype=np.float64) - c) @ R.T + c


# --- exact recovery of a known angle, with a near-zero residual -------------

def _angle_recovery():
    for angle in TEST_ANGLES:
        pts, _ = _line_points(angle)
        _, direction, rms = fit_line_tls(pts)
        got = line_angle_deg(direction)
        assert abs(got - angle) < 1e-6, f"angle {angle}: got {got}"
        assert rms < 1e-9, f"angle {angle}: rms {rms}"

no_crash("TLS recovers known angle at 0/30/54.74/80/90 deg", _angle_recovery)


def _vertical_line_is_90():
    """The case that catches an accidental OLS fit.

    An OLS (polyfit) fit of a vertical line has infinite slope: it either
    raises, returns a garbage near-horizontal line, or blows up numerically.
    A TLS fit returns exactly 90 deg with zero residual.
    """
    pts, _ = _line_points(90.0, n=12)
    origin, direction, rms = fit_line_tls(pts)
    assert abs(line_angle_deg(direction) - 90.0) < 1e-9
    assert rms < 1e-9
    # direction is (0, +-1): purely vertical
    assert abs(direction[0]) < 1e-12, f"dx={direction[0]}"
    assert abs(abs(direction[1]) - 1.0) < 1e-12
    # and a point offset horizontally is exactly that far away
    assert abs(abs(perpendicular_distance(pts[0] + np.array([13.0, 0.0]),
                                          origin, direction)) - 13.0) < 1e-9

no_crash("vertical line fits to 90 deg (catches OLS)", _vertical_line_is_90)


# --- thickness recovery at a known perpendicular offset --------------------

def _thickness_recovery():
    d_true = 7.25
    for angle in TEST_ANGLES:
        pts, d = _line_points(angle)
        origin, direction, _ = fit_line_tls(pts)
        outer = pts[::3] + d_true * _normal(d)
        for p in outer:
            got = abs(perpendicular_distance(p, origin, direction))
            assert abs(got - d_true) < 1e-8, f"angle {angle}: got {got}"

no_crash("thickness == known offset at every facet angle", _thickness_recovery)


def _thickness_is_not_vertical_distance():
    """Guard against reporting the vertical drop instead of the perpendicular.

    At 54.74 deg the vertical distance is d/cos(theta) ~ 1.73x the true
    thickness, so the two are unmistakably different.
    """
    d_true = 10.0
    angle = KOH_111_ANGLE_DEG
    pts, d = _line_points(angle)
    origin, direction, _ = fit_line_tls(pts)
    p = pts[5] + d_true * _normal(d)
    perp = abs(perpendicular_distance(p, origin, direction))
    vertical = d_true / np.cos(np.radians(angle))
    assert abs(perp - d_true) < 1e-8
    assert vertical > 1.7 * d_true          # the wrong answer, for contrast
    assert abs(perp - vertical) > 7.0

no_crash("perpendicular thickness != vertical drop at 54.74 deg", _thickness_is_not_vertical_distance)


# --- sign is preserved so wrong-side points can be flagged ------------------

def _sign_is_signed():
    pts, d = _line_points(35.0)
    origin, direction, _ = fit_line_tls(pts)
    n = _normal(d)
    above = perpendicular_distance(pts[4] + 5.0 * n, origin, direction)
    below = perpendicular_distance(pts[4] - 5.0 * n, origin, direction)
    assert above * below < 0, "opposite sides must have opposite signs"
    assert abs(abs(above) - 5.0) < 1e-9 and abs(abs(below) - 5.0) < 1e-9

no_crash("perpendicular_distance keeps its sign", _sign_is_signed)


# --- foot geometry ---------------------------------------------------------

def _foot_on_line_and_orthogonal():
    for angle in TEST_ANGLES:
        pts, d = _line_points(angle)
        origin, direction, _ = fit_line_tls(pts)
        for p in pts[::4] + 6.5 * _normal(d):
            foot = foot_of_perpendicular(p, origin, direction)
            # foot lies on the line: zero perpendicular distance
            assert abs(perpendicular_distance(foot, origin, direction)) < 1e-9
            # residual vector is orthogonal to the direction
            residual = np.asarray(p) - foot
            assert abs(float(np.dot(residual, direction))) < 1e-9
            # and its length is the thickness
            assert abs(np.linalg.norm(residual) - 6.5) < 1e-9

no_crash("foot lies on line, residual orthogonal to direction", _foot_on_line_and_orthogonal)


# --- rotation invariance ---------------------------------------------------

def _rotation_invariance():
    """Rotating the whole point set must leave every thickness unchanged.

    This is the property OLS does not have: its residuals are vertical, so
    rotating the scene changes the answer.
    """
    rng = np.random.default_rng(7)
    pts, d = _line_points(41.0, n=12)
    outer = pts[::2] + 4.75 * _normal(d) + rng.normal(0, 0.3, size=(6, 2))

    origin, direction, rms0 = fit_line_tls(pts)
    base = np.array([perpendicular_distance(p, origin, direction) for p in outer])

    for rot in (13.0, 47.0, 90.0, 137.0, -66.0):
        rp = _rotate(pts, rot, centre=(320.0, 240.0))
        ro = _rotate(outer, rot, centre=(320.0, 240.0))
        o2, d2, rms2 = fit_line_tls(rp)
        got = np.array([perpendicular_distance(p, o2, d2) for p in ro])
        # sign may flip with the canonical direction; magnitudes must not move
        assert np.allclose(np.abs(got), np.abs(base), atol=1e-8), f"rot {rot}"
        assert abs(rms2 - rms0) < 1e-9

no_crash("thicknesses unchanged by whole-scene rotation", _rotation_invariance)


# --- noise: angle within tolerance, rms tracks the injected sigma -----------

def _noise_behaviour():
    rng = np.random.default_rng(1234)
    sigma = 1.5
    for angle in TEST_ANGLES:
        pts, d = _line_points(angle, n=400, span=600.0)
        noisy = pts + rng.normal(0, sigma, size=pts.shape[0])[:, None] * _normal(d)
        _, direction, rms = fit_line_tls(noisy)
        got = line_angle_deg(direction)
        # 90 and 0 are the same facet seen from either end of the wrap
        delta = min(abs(got - angle), abs(abs(got - angle) - 90.0)) if angle in (0.0, 90.0) \
            else abs(got - angle)
        assert delta < 1.0, f"angle {angle}: got {got}"
        assert abs(rms - sigma) < 0.25 * sigma, f"angle {angle}: rms {rms} vs sigma {sigma}"

no_crash("noisy fit recovers angle and rms ~ sigma", _noise_behaviour)


# --- session-level behaviour ------------------------------------------------

def _session_measurements():
    pts, d = _line_points(54.7356, n=10)
    outer = pts[::2] + 8.0 * _normal(d)
    s = FacetSession(label="upleg", scale={"px": 10, "value": 2.0, "unit": "µm"})
    for p in pts:
        s.add_point(*p)
    assert s.finish_interface()
    assert s.phase == "surface"
    for p in outer:
        s.add_point(*p)

    measurements = s.measurements()
    assert len(measurements) == len(outer)
    for m in measurements:
        assert abs(m.thickness_px - 8.0) < 1e-8
        assert not m.sign_anomalous
    # thickness is reported against the FOOT x, not the clicked outer x
    for m, p in zip(measurements, outer):
        assert abs(m.foot_x - p[0]) > 1e-6

    summary = s.summary()
    # scale is 2 µm per 10 px = 0.2 µm/px, so 8 px -> 1.6 µm
    assert abs(summary["mean"] - 1.6) < 1e-8, summary["mean"]
    assert abs(summary["angle_deg"] - 54.7356) < 1e-6
    assert summary["hint_status"] == "match"
    assert summary["n"] == len(outer)

no_crash("session measurements, foot-x reporting and unit conversion", _session_measurements)


def _sign_anomaly_flag():
    pts, d = _line_points(50.0, n=8)
    s = FacetSession(label="mixed")
    for p in pts:
        s.add_point(*p)
    s.finish_interface()
    n = _normal(d)
    for p in pts[:4]:
        s.add_point(*(p + 6.0 * n))
    s.add_point(*(pts[4] - 6.0 * n))     # deliberate wrong-side click

    measurements = s.measurements()
    flags = [m.sign_anomalous for m in measurements]
    assert flags == [False, False, False, False, True], flags
    assert s.summary()["n_anomalous"] == 1

no_crash("minority-side point is flagged, not dropped", _sign_anomaly_flag)


def _hint_is_soft():
    status, msg = facet_angle_hint(54.0)
    assert status == "match" and msg
    status, msg = facet_angle_hint(38.0)
    assert status == "neutral" and "valid" in msg

no_crash("facet angle hint is informational at both outcomes", _hint_is_soft)


def _no_scale_falls_back_to_px():
    pts, d = _line_points(60.0, n=6)
    s = FacetSession(label="unscaled")
    for p in pts:
        s.add_point(*p)
    s.finish_interface()
    s.add_point(*(pts[2] + 3.0 * _normal(d)))
    assert s.unit == "px"
    assert s.unit_per_px is None
    assert abs(s.summary()["mean"] - 3.0) < 1e-8

no_crash("no scale set falls back to pixels", _no_scale_falls_back_to_px)


def _json_round_trip():
    pts, d = _line_points(54.7356, n=9)
    s = FacetSession(
        label="downleg",
        source_filename="cross_section_042.tif",
        folder_path="/data/ito",
        frame_idx=41,
        scale={"px": 25, "value": 1.0, "unit": "µm"},
        color_idx=2,
        visible=False,
        geometry_tag="rot1|crop10,20,300,400",
    )
    for p in pts:
        s.add_point(*p)
    s.finish_interface()
    for p in pts[::3] + 9.5 * _normal(d):
        s.add_point(*p)

    restored = sessions_from_json(sessions_to_json([s]))
    assert len(restored) == 1
    r = restored[0]
    for field in ("label", "source_filename", "folder_path", "frame_idx",
                  "scale", "color_idx", "visible", "phase", "geometry_tag"):
        assert getattr(r, field) == getattr(s, field), field
    assert np.allclose(r.interface_points, s.interface_points)
    assert np.allclose(r.surface_points, s.surface_points)
    # derived geometry survives because it is recomputed from the same points
    assert r.to_dict() == s.to_dict()
    before = [m.thickness_px for m in s.measurements()]
    after = [m.thickness_px for m in r.measurements()]
    assert np.allclose(before, after)
    assert abs(r.summary()["angle_deg"] - s.summary()["angle_deg"]) < 1e-12

no_crash("JSON round-trip preserves every field", _json_round_trip)


def _fit_rejects_degenerate_input():
    try:
        fit_line_tls(np.array([[1.0, 1.0]]))
        raise AssertionError("expected ValueError for a single point")
    except ValueError:
        pass
    try:
        fit_line_tls(np.array([[5.0, 5.0]] * 6))
        raise AssertionError("expected ValueError for coincident points")
    except ValueError:
        pass
    s = FacetSession(label="empty")
    assert s.fit() is None
    assert s.measurements() == []
    assert not s.finish_interface()

no_crash("degenerate point sets rejected cleanly", _fit_rejects_degenerate_input)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*45}")
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
