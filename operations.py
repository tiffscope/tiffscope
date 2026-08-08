import hashlib
import json
from abc import ABC, abstractmethod

import cv2
import numpy as np


class Operation(ABC):
    name: str = ""
    temporal_radius: int = 0
    supports_batch_cache: bool = False
    params_schema: list = []

    def __init__(self):
        self.params: dict = {p["key"]: p["default"] for p in self.params_schema}
        self.enabled: bool = True
        self._fit_hash: str = ""

    # --- subclasses override ---

    def fit(self, sequence) -> None:
        """Learn state from the whole sequence. Default: no-op."""
        pass

    def fit_with_progress(self, pipeline, op_idx, sequence, progress_callback=None) -> None:
        """Override to compute fit state with progress reporting."""
        pass

    @abstractmethod
    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        pass

    # --- type contract ---

    @property
    def input_requirements(self) -> dict:
        return {"ndim": 2, "dtypes": ["uint8", "uint16", "float32"]}

    @property
    def output_dtype(self) -> str:
        return "preserve"

    def validate_input(self, frame: np.ndarray) -> list:
        """Returns list of (severity, message) strings. Empty = ok."""
        issues = []
        req = self.input_requirements
        if req.get("ndim") and frame.ndim != req["ndim"]:
            issues.append(("error", f"{self.name}: expected {req['ndim']}D array, got {frame.ndim}D"))
        allowed = req.get("dtypes", [])
        if allowed and str(frame.dtype) not in allowed:
            issues.append(("warning", f"{self.name}: unexpected dtype {frame.dtype}, expected one of {allowed}"))
        return issues

    # --- serialization ---

    def to_dict(self) -> dict:
        return {"name": self.name, "params": dict(self.params), "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d: dict) -> "Operation":
        op = cls()
        op.enabled = d.get("enabled", True)
        for k, v in d.get("params", {}).items():
            if k in op.params:
                op.params[k] = v
        return op

    # --- cache invalidation support ---

    def fingerprint(self) -> str:
        blob = json.dumps({"name": self.name, "params": self.params, "fit_hash": self._fit_hash}, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Concrete operations
# ---------------------------------------------------------------------------

class RotateOp(Operation):
    name = "RotateOp"
    params_schema = [
        {"key": "k", "type": "int", "default": 1, "range": [0, 3], "label": "Turns CW", "widget": "spinbox"},
    ]

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        k = self.params["k"]
        if k == 0:
            return frame
        # np.rot90 returns a view with negative strides for k != 0. Downstream
        # ops that do frame.astype(...) on a negative-strided array get an
        # F-contiguous result (astype default order='K' preserves layout) which
        # segfaults when passed to OpenCV on macOS. Force C-contiguous here so
        # every downstream op sees a clean buffer.
        return np.ascontiguousarray(np.rot90(frame, k=-k))


class CropOp(Operation):
    name = "CropOp"
    params_schema = [
        {"key": "x", "type": "int", "default": 0, "range": [0, 99999], "label": "X", "widget": "spinbox"},
        {"key": "y", "type": "int", "default": 0, "range": [0, 99999], "label": "Y", "widget": "spinbox"},
        {"key": "w", "type": "int", "default": 100, "range": [1, 99999], "label": "W", "widget": "spinbox"},
        {"key": "h", "type": "int", "default": 100, "range": [1, 99999], "label": "H", "widget": "spinbox"},
    ]

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        x, y, w, h = self.params["x"], self.params["y"], self.params["w"], self.params["h"]
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))
        # The slice is a view; if `frame` already has negative strides (e.g.
        # post-RotateOp), the slice inherits them. Force C-contiguous so the
        # contract "every Operation.apply() returns a C-contiguous array" holds
        # regardless of upstream ops.
        return np.ascontiguousarray(frame[y:y + h, x:x + w])


class BgSubtractOp(Operation):
    name = "BgSubtractOp"
    params_schema = [
        {"key": "method", "type": "str", "default": "median", "choices": ["median", "mean"],
         "label": "Method", "widget": "combo"},
        {"key": "sample_n", "type": "int", "default": 50, "range": [2, 9999],
         "label": "Frames to sample", "widget": "spinbox"},
    ]

    def __init__(self):
        super().__init__()
        self._background: np.ndarray | None = None

    @property
    def input_requirements(self) -> dict:
        return {"ndim": 2, "dtypes": ["uint8", "uint16", "float32"]}

    def fit_with_progress(self, pipeline, op_idx: int, sequence, progress_callback=None) -> None:
        import os
        import tifffile

        file_paths = [os.path.join(sequence.folder_path, f) for f in sequence.files]
        sample_n = self.params["sample_n"]
        n = min(sample_n, len(file_paths))
        indices = np.unique(np.round(np.linspace(0, len(file_paths) - 1, n)).astype(int))
        n_loaded = len(indices)

        upstream_ops = [op for op in pipeline.operations[:op_idx] if op.enabled]

        stack = None
        for i, idx in enumerate(indices):
            raw = tifffile.imread(file_paths[idx])
            frame = raw
            for op in upstream_ops:
                frame = op.apply(frame)
            if stack is None:
                stack = np.empty((n_loaded, *frame.shape), dtype=np.float32)
            stack[i] = frame.astype(np.float32)
            if progress_callback:
                progress_callback(i + 1, n_loaded)

        if stack is None:
            return

        if self.params["method"] == 'mean':
            self._background = stack.mean(axis=0).astype(np.float32)
            del stack
        else:
            # Sequential chunked median — no inner thread pool to avoid segfaults
            # when called from a QThread with OpenCV ops in the upstream pipeline.
            H, W = stack.shape[1], stack.shape[2]
            target_bytes = 128 * 1024 * 1024
            chunk_rows = max(1, int(target_bytes / (n_loaded * W * 4)))
            n_chunks = max(1, (H + chunk_rows - 1) // chunk_rows)
            result = np.empty((H, W), dtype=np.float32)
            completed = 0
            for r0 in range(0, H, chunk_rows):
                r1 = min(r0 + chunk_rows, H)
                result[r0:r1] = np.median(stack[:, r0:r1, :], axis=0)
                completed += 1
                if progress_callback:
                    progress_callback(n_loaded + completed, n_loaded + n_chunks)
            del stack
            self._background = result

        self._fit_hash = hashlib.md5(self._background.tobytes()).hexdigest()[:8]

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        if self._background is None:
            return frame
        bg = self._background
        if bg.shape != frame.shape:
            return frame
        out = frame.astype(np.float32)
        out -= bg
        np.clip(out, 0, None, out=out)
        return out.astype(frame.dtype)

    def get_background(self) -> np.ndarray | None:
        return self._background

    def set_background(self, bg: np.ndarray) -> None:
        self._background = bg
        if bg is not None:
            self._fit_hash = hashlib.md5(bg.tobytes()).hexdigest()[:8]
        else:
            self._fit_hash = ""

    def to_dict(self) -> dict:
        d = super().to_dict()
        # fit state (background array) not serialized — re-fit on load
        return d


class AdaptiveThresholdOp(Operation):
    name = "AdaptiveThresholdOp"
    supports_batch_cache = True
    params_schema = [
        {"key": "method", "type": "str", "default": "gaussian",
         "choices": ["gaussian", "mean"], "label": "Method", "widget": "combo"},
        {"key": "block_size", "type": "int", "default": 11, "range": [3, 501],
         "label": "Block size (odd)", "widget": "spinbox"},
        {"key": "C", "type": "int", "default": 2, "range": [-50, 50],
         "label": "C", "widget": "spinbox"},
        {"key": "blur_size", "type": "int", "default": 1, "range": [1, 51],
         "label": "Blur kernel (1=off)", "widget": "spinbox"},
    ]

    def __init__(self):
        super().__init__()
        self._batch_was_run: bool = False

    @property
    def output_dtype(self) -> str:
        return "uint8"

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        method_str = self.params["method"]
        cv_method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method_str == "gaussian"
                     else cv2.ADAPTIVE_THRESH_MEAN_C)
        block_size = self.params["block_size"]
        if block_size % 2 == 0:
            block_size += 1
        C = self.params["C"]
        blur_size = self.params["blur_size"]

        # Defensive: if `frame` arrives with negative strides (e.g. an un-fixed
        # upstream op), frame.astype(...) returns F-contiguous, which segfaults
        # in cv2 on macOS. ascontiguousarray is a no-op when already C-contig.
        frame = np.ascontiguousarray(frame)

        d = frame.astype(np.float32)
        lo, hi = d.min(), d.max()
        if hi > lo:
            d -= lo
            d *= 255.0 / (hi - lo)
            u8 = d.astype(np.uint8)
        else:
            u8 = np.zeros(frame.shape, dtype=np.uint8)

        if blur_size > 1:
            k = blur_size if blur_size % 2 == 1 else blur_size + 1
            u8 = cv2.GaussianBlur(u8, (k, k), 0)

        mask = cv2.adaptiveThreshold(u8, 255, cv_method, cv2.THRESH_BINARY, block_size, C)
        return mask


# ---------------------------------------------------------------------------
# Preprocessing ops
# ---------------------------------------------------------------------------

class RollingBallBgOp(Operation):
    """Per-frame *spatial* background subtraction (ImageJ-style rolling ball).

    Unlike BgSubtractOp, this estimates the background from each frame on its
    own — no temporal sampling. Use it when the sequence is too short for a
    clean temporal median/mean (e.g. 5 frames), where the temporal estimate
    leaves ghost artifacts of the moving particles.

    Two methods:
      - ``opening`` (default): grayscale morphological opening with a ball-sized
        ellipse kernel — the classic fast rolling-ball approximation. cv2 only,
        fast enough for live scrubbing.
      - ``rolling_ball``: scikit-image's true rolling-ball surface. More faithful
        to a curved ball but O(seconds) on large frames — use for a few frames.

    Optional pre-smooth (``smooth`` kernel, odd, 1=off): the background surface
    is estimated on a Gaussian-smoothed copy, then subtracted from the *original*
    frame. Mirrors ImageJ — stops per-pixel noise from poking through the ball.

    Assumes bright features on a dark background (standard PIV/PTV). Dtype-preserving.
    """

    name = "RollingBallBgOp"
    params_schema = [
        {"key": "method", "type": "str", "default": "opening",
         "choices": ["opening", "rolling_ball"], "label": "Method", "widget": "combo"},
        {"key": "radius", "type": "int", "default": 50, "range": [1, 500],
         "label": "Ball radius (px)", "widget": "spinbox"},
        {"key": "smooth", "type": "int", "default": 1, "range": [1, 51],
         "label": "Pre-smooth kernel (1=off)", "widget": "spinbox", "step": 2},
    ]

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        radius = max(1, int(self.params["radius"]))
        method = self.params["method"]
        smooth = int(self.params["smooth"])

        src = np.ascontiguousarray(frame)
        work = src.astype(np.float32)

        # Estimate the background surface on a smoothed copy (ImageJ behaviour):
        # noise on the raw image would let single hot pixels poke through the ball.
        est = work
        if smooth > 1:
            k = smooth if smooth % 2 == 1 else smooth + 1
            est = cv2.GaussianBlur(work, (k, k), 0)

        if method == "rolling_ball":
            try:
                from skimage.restoration import rolling_ball
            except ImportError:
                # scikit-image missing — fall back to the opening approximation.
                method = "opening"
            else:
                bg = rolling_ball(est, radius=radius).astype(np.float32)

        if method == "opening":
            k = 2 * radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bg = cv2.morphologyEx(est, cv2.MORPH_OPEN, kernel)

        out = work - bg
        np.clip(out, 0, None, out=out)

        if frame.dtype == np.uint8:
            return np.clip(out, 0, 255).astype(np.uint8)
        if frame.dtype == np.uint16:
            return np.clip(out, 0, 65535).astype(np.uint16)
        return out.astype(frame.dtype)


class CLAHEOp(Operation):
    name = "CLAHEOp"
    params_schema = [
        {"key": "clip_limit", "type": "int", "default": 2, "range": [1, 40],
         "label": "Clip limit", "widget": "spinbox"},
        {"key": "tile_size", "type": "int", "default": 8, "range": [2, 64],
         "label": "Tile size (px)", "widget": "spinbox"},
    ]

    def __init__(self):
        super().__init__()
        self._clahe = None
        self._clahe_key = None

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        clip = self.params["clip_limit"]
        tile = self.params["tile_size"]
        key = (clip, tile)
        if self._clahe is None or self._clahe_key != key:
            self._clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
            self._clahe_key = key
        src = np.ascontiguousarray(frame)
        if src.dtype == np.uint8:
            return self._clahe.apply(src)
        # uint16 or float32 — normalize to uint16, apply, scale back; in-place to minimize allocs
        lo, hi = float(src.min()), float(src.max())
        if hi == lo:
            return frame
        f32 = src.astype(np.float32)
        f32 -= lo
        f32 *= 65535.0 / (hi - lo)
        u16 = f32.astype(np.uint16)
        result = self._clahe.apply(u16)
        r = result.astype(np.float32)
        r *= (hi - lo) / 65535.0
        r += lo
        return r.astype(frame.dtype)


class GaussianBlurOp(Operation):
    name = "GaussianBlurOp"
    params_schema = [
        {"key": "kernel_size", "type": "int", "default": 3, "range": [3, 51],
         "label": "Kernel size (odd)", "widget": "spinbox", "step": 2},
        {"key": "sigma", "type": "int", "default": 0, "range": [0, 20],
         "label": "Sigma (0=auto)", "widget": "spinbox"},
    ]

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        k = self.params["kernel_size"]
        if k % 2 == 0:
            k += 1
        sigma = self.params["sigma"]
        src = np.ascontiguousarray(frame)
        return cv2.GaussianBlur(src, (k, k), sigma).astype(frame.dtype)


class SharpenOp(Operation):
    name = "SharpenOp"
    params_schema = [
        {"key": "amount", "type": "int", "default": 1, "range": [1, 10],
         "label": "Amount (×0.5)", "widget": "spinbox"},
        {"key": "kernel_size", "type": "int", "default": 3, "range": [3, 15],
         "label": "Blur kernel (odd)", "widget": "spinbox", "step": 2},
    ]

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        k = self.params["kernel_size"]
        if k % 2 == 0:
            k += 1
        strength = self.params["amount"] * 0.5
        # Defensive: frame.astype on a negative-strided input returns
        # F-contiguous, which segfaults in cv2.GaussianBlur on macOS.
        frame = np.ascontiguousarray(frame)
        blurred = cv2.GaussianBlur(frame.astype(np.float32), (k, k), 0)
        sharpened = frame.astype(np.float32) + strength * (frame.astype(np.float32) - blurred)
        if frame.dtype == np.uint8:
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        if frame.dtype == np.uint16:
            return np.clip(sharpened, 0, 65535).astype(np.uint16)
        return sharpened.astype(frame.dtype)


class LowPassOp(Operation):
    name = "LowPassOp"
    params_schema = [
        {"key": "kernel_size", "type": "int", "default": 15, "range": [3, 151],
         "label": "Kernel size (odd)", "widget": "spinbox", "step": 2},
    ]

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        k = self.params["kernel_size"]
        if k % 2 == 0:
            k += 1
        src = np.ascontiguousarray(frame)
        return cv2.blur(src, (k, k)).astype(frame.dtype)


class HighPassOp(Operation):
    name = "HighPassOp"
    params_schema = [
        {"key": "kernel_size", "type": "int", "default": 15, "range": [3, 151],
         "label": "Kernel size (odd)", "widget": "spinbox", "step": 2},
    ]

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        k = self.params["kernel_size"]
        if k % 2 == 0:
            k += 1
        src = np.ascontiguousarray(frame)
        blurred = cv2.blur(src, (k, k)).astype(np.float32)
        result = np.clip(src.astype(np.float32) - blurred, 0, None)
        if frame.dtype == np.uint8:
            return np.clip(result, 0, 255).astype(np.uint8)
        if frame.dtype == np.uint16:
            return np.clip(result, 0, 65535).astype(np.uint16)
        return result.astype(frame.dtype)


# ---------------------------------------------------------------------------
# Binary mask ops — run downstream of AdaptiveThresholdOp on 0/255 uint8 masks
# ---------------------------------------------------------------------------

_MORPH_SHAPE_MAP = {
    "rect":    cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross":   cv2.MORPH_CROSS,
}

_MORPH_OP_MAP = {
    "open":     cv2.MORPH_OPEN,
    "close":    cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat":   cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}


class MorphologyOp(Operation):
    name = "MorphologyOp"
    is_binary_mask_op = True
    params_schema = [
        {"key": "operation", "type": "str", "default": "open",
         "choices": ["erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"],
         "label": "Operation", "widget": "combo"},
        {"key": "kernel_size", "type": "int", "default": 3, "range": [1, 51],
         "label": "Kernel size (odd)", "widget": "spinbox", "step": 2},
        {"key": "kernel_shape", "type": "str", "default": "rect",
         "choices": ["rect", "ellipse", "cross"],
         "label": "Kernel shape", "widget": "combo"},
        {"key": "iterations", "type": "int", "default": 1, "range": [1, 20],
         "label": "Iterations", "widget": "spinbox"},
    ]

    @property
    def input_requirements(self) -> dict:
        return {"ndim": 2, "dtypes": ["uint8"]}

    @property
    def output_dtype(self) -> str:
        return "uint8"

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        frame = np.ascontiguousarray(frame)
        k = self.params["kernel_size"]
        if k % 2 == 0:
            k += 1
        shape_const = _MORPH_SHAPE_MAP[self.params["kernel_shape"]]
        kernel = cv2.getStructuringElement(shape_const, (k, k))
        op_str = self.params["operation"]
        n = self.params["iterations"]
        if op_str == "erode":
            return cv2.erode(frame, kernel, iterations=n)
        if op_str == "dilate":
            return cv2.dilate(frame, kernel, iterations=n)
        return cv2.morphologyEx(frame, _MORPH_OP_MAP[op_str], kernel, iterations=n)


class BinarySmoothOp(Operation):
    name = "BinarySmoothOp"
    is_binary_mask_op = True
    params_schema = [
        {"key": "kernel_size", "type": "int", "default": 3, "range": [3, 15],
         "label": "Kernel size (odd)", "widget": "spinbox", "step": 2},
    ]

    @property
    def input_requirements(self) -> dict:
        return {"ndim": 2, "dtypes": ["uint8"]}

    @property
    def output_dtype(self) -> str:
        return "uint8"

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        frame = np.ascontiguousarray(frame)
        k = self.params["kernel_size"]
        if k % 2 == 0:
            k += 1
        return cv2.medianBlur(frame, k)


class WatershedSplitOp(Operation):
    name = "WatershedSplitOp"
    is_binary_mask_op = True
    params_schema = [
        {"key": "min_peak_distance", "type": "int", "default": 5, "range": [1, 50],
         "label": "Min peak distance (px)", "widget": "spinbox"},
        {"key": "min_peak_value", "type": "int", "default": 2, "range": [1, 30],
         "label": "Min peak value (px radius)", "widget": "spinbox"},
    ]

    @property
    def input_requirements(self) -> dict:
        return {"ndim": 2, "dtypes": ["uint8"]}

    @property
    def output_dtype(self) -> str:
        return "uint8"

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        frame = np.ascontiguousarray(frame)
        dist = cv2.distanceTransform(frame, cv2.DIST_L2, 5)

        # Local maxima via dilation-equality.
        min_dist = max(1, int(self.params["min_peak_distance"]))
        min_val = float(self.params["min_peak_value"])
        k = 2 * min_dist + 1
        dilated = cv2.dilate(dist, np.ones((k, k), np.uint8))
        peaks = (dist == dilated) & (dist >= min_val)

        # Every blob needs at least one marker, otherwise watershed absorbs
        # peakless blobs into the background label.
        n_blobs, blob_labels = cv2.connectedComponents(frame)
        for bid in range(1, n_blobs):
            comp = blob_labels == bid
            if not (peaks & comp).any():
                cd = np.where(comp, dist, -1.0)
                y, x = np.unravel_index(int(np.argmax(cd)), dist.shape)
                peaks[y, x] = True

        peaks_u8 = peaks.astype(np.uint8) * 255
        n_pl, peak_labels = cv2.connectedComponents(peaks_u8)
        if n_pl <= 2:
            return frame

        markers = np.zeros(frame.shape, np.int32)
        markers[frame == 0] = 1
        fg = peak_labels > 0
        markers[fg] = peak_labels[fg] + 1

        dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dist_bgr = cv2.cvtColor(dist_u8, cv2.COLOR_GRAY2BGR)
        cv2.watershed(dist_bgr, markers)

        return np.where(markers > 1, 255, 0).astype(np.uint8)


class IntensityWatershedSplitOp(Operation):
    """Intensity-guided circular-particle deduplication for PIV/PTV images.

    For each connected blob in the binary mask:
      - 1 intensity peak found → copy original blob pixels unchanged (no shrinkage ever)
      - N > 1 peaks found    → draw N filled circles, one per peak.
                                Radius = min(area-preserving r, half-nearest-peak-dist − 0.5)
                                guarantees adjacent circles never overlap at radius_scale=100%.

    This approach is designed for roughly circular fluorescent/Mie-scattering
    particles.  Single particles are *never* modified.  Only blobs whose
    intensity topology reveals multiple peaks are split, and the reconstruction
    always produces circles of the correct per-particle area.

    Context requirement: the binary-chain walker in update_frame_display (and
    save_threshold_masks) must pass ``context={'intensity_frame': ndarray}``
    so the op can access the upstream grayscale image.  Without context the op
    returns the input mask unchanged (graceful no-op).

    ``requires_intensity_context = True`` signals the walker to populate context.
    """

    name = "IntensityWatershedSplitOp"
    is_binary_mask_op = True
    requires_intensity_context = True

    # Set by apply(); the GUI overlay uses this to colour reconstructed circles
    # differently from preserved blobs.  None until first apply().
    #   _last_split_pixels — pixels placed by circle reconstruction (N≥2 peaks) → cyan
    #   all other mask pixels (small blobs + large single-particle blobs)        → orange
    _last_split_pixels: "np.ndarray | None" = None

    params_schema = [
        # intensity_source_idx: -1 = pre-threshold frame (display_frame fed into
        # AdaptiveThresholdOp); N = output of pipeline op at index N.
        # Rendered as a dynamic combo by IntensityWatershedParamsWidget; spinbox fallback.
        {"key": "intensity_source_idx", "type": "int", "default": -1, "range": [-1, 999],
         "label": "Intensity source stage", "widget": "spinbox"},
        {"key": "min_peak_distance", "type": "int", "default": 5, "range": [1, 50],
         "label": "Min peak distance (px)", "widget": "spinbox"},
        # threshold_abs is in 0-255 normalised intensity space
        {"key": "threshold_abs", "type": "int", "default": 10, "range": [0, 254],
         "label": "Min peak intensity (0-255)", "widget": "spinbox"},
        # radius_scale: percentage of area-derived radius to draw; 100 = exact area-preserving
        {"key": "radius_scale", "type": "int", "default": 100, "range": [50, 200],
         "label": "Radius scale (%)", "widget": "spinbox"},
        # min_blob_area: blobs smaller than this are copied unchanged (no peak detection).
        # Primary performance lever — skips skimage.peak_local_max for the many
        # small single-particle blobs that can never be merged particles.
        {"key": "min_blob_area", "type": "int", "default": 50, "range": [1, 100000],
         "label": "Min blob area for splitting (px²)", "widget": "spinbox"},
    ]

    @property
    def input_requirements(self) -> dict:
        return {"ndim": 2, "dtypes": ["uint8"]}

    @property
    def output_dtype(self) -> str:
        return "uint8"

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def apply(self, frame: np.ndarray, context=None) -> np.ndarray:
        frame = np.ascontiguousarray(frame)

        if not (frame > 0).any():
            return frame  # empty mask — nothing to do

        # ---- resolve intensity image -----------------------------------
        intensity_raw = None
        if context is not None:
            intensity_raw = context.get("intensity_frame")

        if intensity_raw is None:
            # No intensity context: return mask unchanged.
            return frame

        intensity_raw = np.ascontiguousarray(intensity_raw.astype(np.float32))

        min_dist = max(1, int(self.params["min_peak_distance"]))
        threshold = float(self.params["threshold_abs"])
        radius_scale = float(self.params["radius_scale"]) / 100.0
        min_blob_area = max(1, int(self.params.get("min_blob_area", 50)))

        # ---- scikit-image peak finder (required) -----------------------
        try:
            from skimage.feature import peak_local_max
        except ImportError:
            return frame

        # Connected components of the binary mask — one label per particle blob
        n_blobs, blob_labels = cv2.connectedComponents(frame)

        output = np.zeros_like(frame)
        # split_pixels: tracks pixels placed by circle reconstruction (N≥2 peaks).
        # Lazily allocated — avoids full-frame alloc when no blobs are split.
        # All other output pixels (small blobs + large single-particle) → orange.
        split_pixels: "np.ndarray | None" = None

        for bid in range(1, n_blobs):
            comp = blob_labels == bid          # bool mask for this blob
            blob_area = int(comp.sum())

            # Small blobs: skip peak detection entirely (performance + they can't
            # be merged particles).  Copy to output; leave tracking arrays clear
            # so overlay colours them orange.
            if blob_area < min_blob_area:
                output[comp] = 255
                continue

            # --- Per-blob intensity normalisation -----------------------
            # Map each blob's own intensity range to [0, 255] independently.
            # Global frame normalisation squashes dim blobs when brighter
            # particles dominate the range — per-blob normalisation gives every
            # blob full dynamic range for peak detection.
            blob_vals = intensity_raw[comp]
            lo_b, hi_b = float(blob_vals.min()), float(blob_vals.max())
            if hi_b <= lo_b:
                # Flat blob — cannot distinguish peaks; keep as-is.
                output[comp] = 255
                continue

            # Normalised image: blob pixels in [0, 255], background = 0.
            intensity_blob = np.zeros(frame.shape, dtype=np.float32)
            intensity_blob[comp] = (blob_vals - lo_b) / (hi_b - lo_b) * 255.0

            # --- NO internal smoothing ---
            # Blurring on a zero-background causes edge-bleed that darkens blob
            # perimeters and forces the maximum to the geometric centre — it
            # destroys the twin-peak saddle in merged particles.  The forced
            # min sigma=1 is also too aggressive for small particles (≤10 px wide).
            # Any desired smoothing should come from the upstream "Intensity source"
            # selection (e.g. GaussianBlurOp) which operates on the full frame
            # without a hard zero boundary at the blob edge.
            intensity_for_peaks = intensity_blob

            # Find local maxima within blob interior.
            # ``labels`` restricts neighbour comparison to within this blob so
            # edge pixels are not promoted by the 0.0 background outside.
            coords = peak_local_max(
                intensity_for_peaks,
                min_distance=min_dist,
                threshold_abs=threshold,
                labels=comp.astype(np.uint8),
            )
            # coords: (N, 2) array of (row, col), empty shape (0, 2) when no peaks

            n_peaks = len(coords)

            if n_peaks <= 1:
                # Zero or one peak → single large particle; preserve exactly.
                output[comp] = 255
                continue

            # ---- N ≥ 2 peaks: Voronoi partition → per-region circle ----
            # Assign each blob pixel to the nearest detected peak (Voronoi cell).
            #
            # Circle centre = geometric centroid of the cell, NOT the intensity
            # peak.  This corrects centroid shift when the PSF peak lies off-centre
            # of the physical particle (asymmetric illumination, clipping, etc.).
            #
            # Circle radius = sqrt(cell_area / π) × radius_scale.
            # Derived from the actual area of each sub-region, so unequal-sized
            # particles get correctly-sized circles without any global formula.
            # No separation term needed: Voronoi regions are non-overlapping by
            # definition, so circles only overlap when particles are physically
            # very close — which is accurate.
            comp_rows, comp_cols = np.where(comp)
            pixel_coords = np.stack(
                [comp_rows.astype(np.float32), comp_cols.astype(np.float32)], axis=1
            )                                               # (M, 2)
            peak_f = coords.astype(np.float32)              # (N, 2)
            # Squared Euclidean distance from each blob pixel to each peak (M×N)
            dists_sq = np.sum(
                (pixel_coords[:, None, :] - peak_f[None, :, :]) ** 2, axis=2
            )
            nearest = np.argmin(dists_sq, axis=1)           # (M,) Voronoi label

            if split_pixels is None:
                split_pixels = np.zeros_like(frame)

            for k in range(n_peaks):
                region = nearest == k
                if not region.any():
                    continue
                rr = comp_rows[region]
                cc = comp_cols[region]
                cy = int(round(float(rr.mean())))
                cx = int(round(float(cc.mean())))
                # Subtract 1px so adjacent circles (2r == d) leave a gap rather
                # than touching — 8-connectivity treats touching as connected.
                r  = max(1, int(round(np.sqrt(region.sum() / np.pi) * radius_scale)) - 1)
                cv2.circle(output,       (cx, cy), r, 255, thickness=-1)
                cv2.circle(split_pixels, (cx, cy), r, 255, thickness=-1)

        self._last_split_pixels = split_pixels
        return output


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OPERATION_REGISTRY: dict[str, type[Operation]] = {
    cls.name: cls
    for cls in [RotateOp, CropOp, BgSubtractOp, RollingBallBgOp, AdaptiveThresholdOp,
                CLAHEOp, GaussianBlurOp, SharpenOp, LowPassOp, HighPassOp,
                MorphologyOp, BinarySmoothOp, WatershedSplitOp,
                IntensityWatershedSplitOp]
}


def operation_from_dict(d: dict) -> Operation | None:
    cls = OPERATION_REGISTRY.get(d.get("name", ""))
    if cls is None:
        return None
    return cls.from_dict(d)
