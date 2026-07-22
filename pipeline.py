import json
import time
from typing import TYPE_CHECKING

import numpy as np

from operations import Operation, operation_from_dict

if TYPE_CHECKING:
    from image_engine import LazyTiffSequence


class Pipeline:
    def __init__(self):
        self.operations: list[Operation] = []
        # batch caches: op index → {frame_idx: ndarray}
        self._caches: list[dict] = []
        # Pixel-to-physical-unit scale, set via Tools → Set Scale…
        # None or {"px": int, "value": float, "unit": str}
        self.scale: dict | None = None

    # ------------------------------------------------------------------
    # Mutation API — all call _invalidate_from(idx) as needed
    # ------------------------------------------------------------------

    def add_operation(self, op: Operation, at_idx: int = -1) -> int:
        if at_idx == -1 or at_idx >= len(self.operations):
            self.operations.append(op)
            self._caches.append({})
            return len(self.operations) - 1
        else:
            self.operations.insert(at_idx, op)
            self._caches.insert(at_idx, {})
            self._invalidate_from(at_idx)
            return at_idx

    def remove_operation(self, idx: int) -> None:
        if 0 <= idx < len(self.operations):
            self.operations.pop(idx)
            self._caches.pop(idx)
            self._invalidate_from(idx)

    def move_operation(self, from_idx: int, to_idx: int) -> None:
        if from_idx == to_idx:
            return
        op = self.operations.pop(from_idx)
        cache = self._caches.pop(from_idx)
        self.operations.insert(to_idx, op)
        self._caches.insert(to_idx, cache)
        self._invalidate_from(min(from_idx, to_idx))

    def set_enabled(self, idx: int, enabled: bool) -> None:
        if 0 <= idx < len(self.operations):
            self.operations[idx].enabled = enabled
            self._invalidate_from(idx)

    def update_params(self, idx: int, params: dict) -> None:
        if 0 <= idx < len(self.operations):
            self.operations[idx].params.update(params)
            self._invalidate_from(idx)

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def _invalidate_from(self, stage_idx: int) -> None:
        for i in range(stage_idx, len(self._caches)):
            self._caches[i].clear()

    def store_batch_cache(self, idx: int, frame_idx: int, result: np.ndarray) -> None:
        if 0 <= idx < len(self._caches):
            self._caches[idx][frame_idx] = result

    def get_batch_cache(self, idx: int, frame_idx: int) -> np.ndarray | None:
        if 0 <= idx < len(self._caches):
            return self._caches[idx].get(frame_idx)
        return None

    def clear_batch_cache(self, idx: int) -> None:
        if 0 <= idx < len(self._caches):
            self._caches[idx].clear()

    def has_full_batch_cache(self, idx: int, total_frames: int) -> bool:
        if 0 <= idx < len(self._caches):
            return len(self._caches[idx]) == total_frames
        return False

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply_to_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        target_stage_idx: int = -1,
    ) -> np.ndarray:
        """
        Walk enabled operations up to and including target_stage_idx (-1 = all).
        -2 or any value < -1 = run zero ops (return frame unchanged).
        Uses batch cache for ops that support it when available.
        """
        if target_stage_idx is not None and target_stage_idx < -1:
            return frame
        result = frame
        limit = len(self.operations) if target_stage_idx == -1 else target_stage_idx + 1
        for i, op in enumerate(self.operations[:limit]):
            if not op.enabled:
                continue
            # use batch cache if available
            cached = self.get_batch_cache(i, frame_idx)
            if cached is not None:
                result = cached
                continue
            result = op.apply(result)
        return result

    def apply_with_snapshots(
        self,
        frame: np.ndarray,
        frame_idx: int,
        snapshot_indices: set,
        final_target: int = -1,
        timings: dict | None = None,
    ) -> tuple:
        """Walk the pipeline once and capture intermediate frames.

        snapshot_indices: inclusive op indices to record after that op runs.
                          Use -1 to capture the input frame before any op.
        final_target: same sentinel as apply_to_frame (-2=none, -1=all, n>=0=up to n inclusive).
        timings:       optional dict; when provided, per-op wall-clock (ms) is recorded with
                       keys like "  [3] CLAHEOp" (cached ops get suffix "(cached)").
                       Disabled ops are not recorded.
        Returns (final_frame, {idx: ndarray}).
        Disabled ops are passthrough; their snapshot (if requested) is the current result.
        """
        snapshots = {}
        if -1 in snapshot_indices:
            snapshots[-1] = frame

        if final_target < -1:
            return frame, snapshots

        result = frame
        limit = len(self.operations) if final_target == -1 else final_target + 1
        for i, op in enumerate(self.operations[:limit]):
            if not op.enabled:
                if i in snapshot_indices:
                    snapshots[i] = result
                continue
            cached = self.get_batch_cache(i, frame_idx)
            if cached is not None:
                if timings is not None:
                    timings[f"  [{i}] {op.name} (cached)"] = 0.0
                result = cached
            else:
                if timings is not None:
                    _t0 = time.perf_counter()
                    result = op.apply(result)
                    timings[f"  [{i}] {op.name}"] = (time.perf_counter() - _t0) * 1000.0
                else:
                    result = op.apply(result)
            if i in snapshot_indices:
                snapshots[i] = result
        return result, snapshots

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_chain(self) -> list[tuple[int, str, str]]:
        """Returns list of (stage_idx, severity, message)."""
        issues = []
        prev_dtype = None
        for i, op in enumerate(self.operations):
            if not op.enabled:
                continue
            req = op.input_requirements
            if prev_dtype is not None:
                allowed = req.get("dtypes", [])
                if allowed and prev_dtype not in allowed:
                    issues.append((i, "error",
                                   f"{op.name}: requires {allowed}, previous op outputs {prev_dtype}"))
            out = op.output_dtype
            if out != "preserve":
                prev_dtype = out
        return issues

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "version": 1,
            "operations": [op.to_dict() for op in self.operations],
        }
        if self.scale is not None:
            d["scale"] = self.scale
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict, warn_unknown=True) -> "Pipeline":
        p = cls()
        for entry in d.get("operations", []):
            op = operation_from_dict(entry)
            if op is None:
                if warn_unknown:
                    print(f"[Pipeline] Unknown operation '{entry.get('name', '?')}' — skipped.")
                continue
            p.add_operation(op)
        scale = d.get("scale")
        if isinstance(scale, dict) and {"px", "value", "unit"} <= set(scale.keys()):
            try:
                p.scale = {
                    "px": int(scale["px"]),
                    "value": float(scale["value"]),
                    "unit": str(scale["unit"]),
                }
            except (TypeError, ValueError):
                if warn_unknown:
                    print("[Pipeline] Invalid 'scale' field — ignored.")
        return p

    @classmethod
    def from_json(cls, s: str, warn_unknown=True) -> "Pipeline":
        return cls.from_dict(json.loads(s), warn_unknown=warn_unknown)
