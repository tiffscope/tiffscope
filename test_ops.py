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
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*45}")
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
