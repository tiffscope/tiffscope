import sys
import os
import time
import faulthandler
faulthandler.enable()
import cv2
import tifffile
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QFormLayout,
    QWidget, QLabel, QFileDialog, QSlider, QMessageBox, QDialog, QComboBox,
    QSpinBox, QDialogButtonBox, QProgressDialog, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QCheckBox, QGroupBox, QScrollArea,
    QSizePolicy, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, QRectF, QThread, pyqtSignal, QEventLoop
from PyQt6.QtGui import QShortcut, QKeySequence, QAction, QActionGroup, QColor, QIntValidator, QDoubleValidator

from image_engine import LazyTiffSequence, scale_16bit_to_8bit, build_display_lut
from operations import (
    Operation, RotateOp, CropOp, BgSubtractOp, RollingBallBgOp, AdaptiveThresholdOp,
    CLAHEOp, GaussianBlurOp, SharpenOp, LowPassOp, HighPassOp,
    MorphologyOp, BinarySmoothOp, WatershedSplitOp, IntensityWatershedSplitOp,
    OPERATION_REGISTRY,
)
from pipeline import Pipeline
from measurement import (
    FacetSession, line_segment_across_box,
    sessions_to_json, sessions_from_json, sessions_to_csv,
)

pg.setConfigOptions(antialias=True)


def _make_separator():
    from PyQt6.QtWidgets import QFrame
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


def _fmt_num(v: float) -> str:
    """Compact float repr: drops trailing zeros; integer values render without decimals."""
    if float(v).is_integer():
        return str(int(v))
    return f"{v:g}"


# ---------------------------------------------------------------------------
# Worker threads (unchanged from original)
# ---------------------------------------------------------------------------

class BgComputeWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, op: BgSubtractOp, op_idx: int, pipeline: Pipeline, sequence):
        super().__init__()
        self.op = op
        self.op_idx = op_idx
        self.pipeline = pipeline
        self.sequence = sequence

    def run(self):
        try:
            self.op.fit_with_progress(
                self.pipeline,
                self.op_idx,
                self.sequence,
                progress_callback=lambda c, t: self.progress.emit(c, t),
            )
            self.finished.emit(self.op.get_background())
        except Exception as e:
            self.error.emit(str(e))


class ThresholdBatchWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, op: AdaptiveThresholdOp, pipeline: Pipeline, op_idx: int,
                 file_paths: list):
        super().__init__()
        self.op = op
        self.pipeline = pipeline
        self.op_idx = op_idx
        self.file_paths = file_paths

    def run(self):
        try:
            masks = {}
            n = len(self.file_paths)
            for i, fp in enumerate(self.file_paths):
                raw = tifffile.imread(fp)
                # apply pipeline up to (but not including) this op
                frame = raw
                for j, step in enumerate(self.pipeline.operations[:self.op_idx]):
                    if step.enabled:
                        frame = step.apply(frame)
                mask = self.op.apply(frame)
                masks[i] = mask
                self.progress.emit(i + 1, n)
            self.finished.emit(masks)
        except Exception as e:
            self.error.emit(str(e))


class BlobSizeAnalysisWorker(QThread):
    """Background worker: collects blob areas from a sample of frames.

    Runs AdaptiveThresholdOp (batch cache when available) + binary chain ops
    (excluding IntensityWatershedSplitOp) on each sampled frame, then uses
    cv2.connectedComponentsWithStats to collect blob areas.  Results are
    emitted as a flat list of int pixel areas across all sampled frames.
    """
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, pipeline, sequence, thresh_idx: int,
                 sample_indices: list, binary_chain_ops: list):
        super().__init__()
        self.pipeline = pipeline
        self.sequence = sequence
        self.thresh_idx = thresh_idx
        self.sample_indices = sample_indices
        self.binary_chain_ops = binary_chain_ops
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            areas = []
            file_paths = [
                os.path.join(self.sequence.folder_path, f)
                for f in self.sequence.files
            ]
            thresh_op = self.pipeline.operations[self.thresh_idx]
            upstream_ops = [
                op for op in self.pipeline.operations[:self.thresh_idx]
                if op.enabled
            ]
            n = len(self.sample_indices)

            for i, frame_idx in enumerate(self.sample_indices):
                if self._cancelled:
                    return

                # Use batch cache when available — fast path
                cached = self.pipeline.get_batch_cache(self.thresh_idx, frame_idx)
                if cached is not None:
                    mask = cached.copy()
                else:
                    raw = tifffile.imread(file_paths[frame_idx])
                    frame = raw
                    for op in upstream_ops:
                        frame = op.apply(frame)
                    mask = thresh_op.apply(frame)

                for op in self.binary_chain_ops:
                    mask = op.apply(mask, context=None)

                # connectedComponentsWithStats gives area for each label directly
                n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
                for lbl in range(1, n_labels):
                    areas.append(int(stats[lbl, cv2.CC_STAT_AREA]))

                self.progress.emit(i + 1, n)

            self.finished.emit(areas)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# PerfTracker + PerformanceToolWindow — live per-stage timing of the
# update_frame_display hot path. Tracker is gated by `enabled`; when False all
# `span()` calls are near-free (one attribute load, no perf_counter call).
# ---------------------------------------------------------------------------

class PerfTracker:
    HISTORY = 30  # rolling window length (frames)

    def __init__(self):
        self.enabled = False
        self.stages: dict[str, list[float]] = {}
        self._frame_t0 = 0.0
        self.last_frame_ms = 0.0
        self.frame_count = 0

    class _Span:
        __slots__ = ("tracker", "name", "t0")
        def __init__(self, tracker, name):
            self.tracker = tracker
            self.name = name
            self.t0 = 0.0
        def __enter__(self):
            if self.tracker.enabled:
                self.t0 = time.perf_counter()
            return self
        def __exit__(self, exc_type, exc, tb):
            if self.tracker.enabled:
                self.tracker._record(self.name, (time.perf_counter() - self.t0) * 1000.0)
            return False

    def span(self, name: str):
        return PerfTracker._Span(self, name)

    def _record(self, name: str, ms: float):
        buf = self.stages.get(name)
        if buf is None:
            buf = []
            self.stages[name] = buf
        buf.append(ms)
        if len(buf) > self.HISTORY:
            del buf[0]

    def merge(self, timings: dict):
        for k, v in timings.items():
            self._record(k, v)

    def begin_frame(self):
        if self.enabled:
            self._frame_t0 = time.perf_counter()

    def end_frame(self):
        if self.enabled:
            self.last_frame_ms = (time.perf_counter() - self._frame_t0) * 1000.0
            self._record("__frame_total__", self.last_frame_ms)
            self.frame_count += 1

    def fps(self) -> float:
        buf = self.stages.get("__frame_total__")
        if not buf:
            return 0.0
        avg = sum(buf) / len(buf)
        return 1000.0 / avg if avg > 0 else 0.0

    def snapshot(self) -> list[tuple]:
        """Return [(name, last_ms, avg_ms, max_ms, pct_of_frame_avg)] sorted by avg desc."""
        frame_buf = self.stages.get("__frame_total__")
        total_avg = (sum(frame_buf) / len(frame_buf)) if frame_buf else 0.0
        rows = []
        for name, buf in self.stages.items():
            if name == "__frame_total__" or not buf:
                continue
            last = buf[-1]
            avg = sum(buf) / len(buf)
            mx = max(buf)
            pct = (avg / total_avg * 100.0) if total_avg > 0 else 0.0
            rows.append((name, last, avg, mx, pct))
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows

    def reset(self):
        self.stages.clear()
        self.last_frame_ms = 0.0
        self.frame_count = 0


class PerformanceToolWindow(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.setWindowTitle("Performance Monitor")
        self.setWindowFlags(Qt.WindowType.Tool)
        self.resize(600, 460)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.header_label = QLabel("Open a sequence and scrub to populate.")
        self.header_label.setStyleSheet("font-family: monospace; font-size: 13px;")
        layout.addWidget(self.header_label)

        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Stage", "Last (ms)", "Avg30 (ms)", "Max30 (ms)", "% frame"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 5):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(reset_btn)
        copy_btn = QPushButton("Copy CSV")
        copy_btn.clicked.connect(self._copy_csv)
        btn_row.addWidget(copy_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        hint = QLabel(
            "Numbers are wall-clock per-stage timings. Pipeline ops are nested under 'pipeline_walk'."
            " Scrub frames to fill the rolling 30-frame window."
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def _on_reset(self):
        self.main_app.perf.reset()
        self.refresh()

    def _copy_csv(self):
        rows = self.main_app.perf.snapshot()
        lines = ["stage,last_ms,avg_ms,max_ms,pct_total"]
        for name, last, avg, mx, pct in rows:
            lines.append(f"{name.strip()},{last:.3f},{avg:.3f},{mx:.3f},{pct:.2f}")
        QApplication.clipboard().setText("\n".join(lines))

    def refresh(self):
        if not self.isVisible():
            return
        rows = self.main_app.perf.snapshot()
        fps = self.main_app.perf.fps()
        last_total = self.main_app.perf.last_frame_ms
        frame_buf = self.main_app.perf.stages.get("__frame_total__")
        avg_total = (sum(frame_buf) / len(frame_buf)) if frame_buf else 0.0
        self.header_label.setText(
            f"FPS: {fps:5.1f}   |   Frame total — last: {last_total:6.2f} ms   "
            f"avg30: {avg_total:6.2f} ms   |   Frames: {self.main_app.perf.frame_count}"
        )

        from PyQt6.QtWidgets import QTableWidgetItem
        self.table.setRowCount(len(rows))
        for r, (name, last, avg, mx, pct) in enumerate(rows):
            items = [
                QTableWidgetItem(name),
                QTableWidgetItem(f"{last:7.2f}"),
                QTableWidgetItem(f"{avg:7.2f}"),
                QTableWidgetItem(f"{mx:7.2f}"),
                QTableWidgetItem(f"{pct:5.1f}%"),
            ]
            for c, item in enumerate(items):
                if c > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)

    def showEvent(self, ev):
        self.main_app.perf.enabled = True
        super().showEvent(ev)

    def hideEvent(self, ev):
        self.main_app.perf.enabled = False
        super().hideEvent(ev)


# ---------------------------------------------------------------------------
# BlobSizeAnalysisWindow
# ---------------------------------------------------------------------------

class BlobSizeAnalysisWindow(QWidget):
    """Histogram of connected-blob areas across a sample of frames.

    Helps the user pick a good ``min_blob_area`` threshold for
    IntensityWatershedSplitOp: the tail of the distribution identifies
    large merged blobs that are candidates for splitting.

    Analysis runs in BlobSizeAnalysisWorker (background QThread).
    When IntensityWatershedSplitOp is in the pipeline a dashed yellow
    vertical line shows the current min_blob_area value.
    """

    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.setWindowTitle("Blob Size Analysis")
        self.setWindowFlags(Qt.WindowType.Tool)
        self.resize(660, 520)
        self._worker: BlobSizeAnalysisWorker | None = None
        self._areas: list[int] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # --- Controls row ---
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Sample frames:"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 99999)
        self.spin_n.setValue(200)
        ctrl.addWidget(self.spin_n)
        self.all_check = QCheckBox("All frames")
        self.all_check.toggled.connect(lambda checked: self.spin_n.setEnabled(not checked))
        ctrl.addWidget(self.all_check)
        ctrl.addStretch()
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self._run)
        ctrl.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel)
        ctrl.addWidget(self.cancel_btn)
        layout.addLayout(ctrl)

        # --- Progress bar ---
        from PyQt6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # --- Histogram plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Blob Area (px²)')
        self.plot_widget.setLabel('left', 'Count')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._bar_item: pg.BarGraphItem | None = None
        # Dashed yellow vertical line = current min_blob_area threshold
        self._vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color=(255, 215, 0), width=2,
                         style=Qt.PenStyle.DashLine),
            label="min_blob_area={value:.0f}",
            labelOpts={"position": 0.85, "color": (255, 215, 0),
                       "fill": (0, 0, 0, 120)},
        )
        self._vline.setVisible(False)
        self.plot_widget.addItem(self._vline)
        layout.addWidget(self.plot_widget, stretch=1)

        # --- Options row ---
        opt = QHBoxLayout()
        self.log_y_check = QCheckBox("Log Y axis")
        self.log_y_check.toggled.connect(self._redraw)
        opt.addWidget(self.log_y_check)
        opt.addStretch()
        layout.addLayout(opt)

        # --- Stats label ---
        self.stats_label = QLabel("Run analysis to see blob size statistics.")
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

    # ------------------------------------------------------------------

    def _run(self):
        app = self.main_app
        if app.sequence_manager is None:
            self.stats_label.setText("No sequence loaded.")
            return

        thresh_idx, thresh_op = app._find_op(AdaptiveThresholdOp)
        if thresh_op is None or not thresh_op.enabled:
            self.stats_label.setText("No enabled AdaptiveThresholdOp in pipeline.")
            return

        # Binary chain ops after thresh, excluding IntensityWatershedSplitOp
        binary_ops = [
            op for op in app.pipeline.operations[thresh_idx + 1:]
            if op.enabled
            and getattr(op, "is_binary_mask_op", False)
            and not isinstance(op, IntensityWatershedSplitOp)
        ]

        n_frames = app.sequence_manager.num_frames
        if self.all_check.isChecked():
            sample_indices = list(range(n_frames))
        else:
            n = min(self.spin_n.value(), n_frames)
            sample_indices = list(np.unique(
                np.round(np.linspace(0, n_frames - 1, n)).astype(int)
            ))

        self._worker = BlobSizeAnalysisWorker(
            app.pipeline, app.sequence_manager,
            thresh_idx, sample_indices, binary_ops,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self.run_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setMaximum(len(sample_indices))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._areas = []
        self.stats_label.setText(f"Analysing {len(sample_indices)} frames…")
        self._worker.start()

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait()
            self._worker = None
        self._set_idle()
        self.stats_label.setText("Cancelled.")

    def _on_progress(self, done: int, total: int):
        self.progress_bar.setValue(done)

    def _on_finished(self, areas: list):
        self._areas = areas
        self._worker = None
        self._set_idle()
        self._redraw()

    def _on_error(self, msg: str):
        self._worker = None
        self._set_idle()
        self.stats_label.setText(f"Error: {msg}")

    def _set_idle(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)

    def _redraw(self):
        areas = self._areas
        if not areas:
            return

        arr = np.array(areas, dtype=np.float64)
        # Bin count: sqrt rule, clamped to [30, 300]
        n_bins = int(np.clip(np.sqrt(len(arr)), 30, 300))
        counts, edges = np.histogram(arr, bins=n_bins)

        use_log = self.log_y_check.isChecked()
        y = np.log10(counts.astype(np.float64) + 1) if use_log else counts.astype(np.float64)

        if self._bar_item is not None:
            self.plot_widget.removeItem(self._bar_item)
        widths = np.diff(edges)
        self._bar_item = pg.BarGraphItem(
            x=edges[:-1], height=y, width=widths * 0.9,
            brush=(100, 180, 255, 180), pen=pg.mkPen(None),
        )
        self.plot_widget.addItem(self._bar_item)
        self.plot_widget.setLabel('left', 'log₁₀(count+1)' if use_log else 'Count')

        # Update min_blob_area line
        _, iw_op = self.main_app._find_op(IntensityWatershedSplitOp)
        if iw_op is not None and iw_op.enabled:
            mba = iw_op.params.get("min_blob_area", 50)
            self._vline.setValue(mba)
            self._vline.setVisible(True)
        else:
            self._vline.setVisible(False)

        # Stats
        median = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        stats_text = (
            f"Blobs: {len(arr):,}   |   "
            f"Median: {median:.0f} px²   |   "
            f"95th pct: {p95:.0f} px²   |   "
            f"99th pct: {p99:.0f} px²   |   "
            f"Max: {arr.max():.0f} px²"
        )
        if iw_op is not None and iw_op.enabled:
            mba = iw_op.params.get("min_blob_area", 50)
            n_above = int((arr >= mba).sum())
            stats_text += f"   |   ≥ min_blob_area ({mba} px²): {n_above:,} blobs"
        self.stats_label.setText(stats_text)

    def showEvent(self, event):
        super().showEvent(event)
        # Refresh vline whenever window becomes visible in case params changed
        if self._areas:
            self._redraw()


# ---------------------------------------------------------------------------
# RegionPropsWorker / RegionPropsWindow
# ---------------------------------------------------------------------------

class RegionPropsWorker(QThread):
    """Background worker: collects per-blob regionprops across sampled frames.

    Applies AdaptiveThresholdOp + selected binary chain ops, then uses
    skimage.measure.regionprops to collect area, equivalent_diameter,
    major_axis_length, minor_axis_length, eccentricity per blob.

    Emits finished(dict) with arrays keyed by stat name plus 'frame_idx' and
    'label' for CSV export.
    """
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, pipeline, sequence, thresh_idx: int,
                 sample_indices: list, binary_chain_ops: list):
        super().__init__()
        self.pipeline = pipeline
        self.sequence = sequence
        self.thresh_idx = thresh_idx
        self.sample_indices = sample_indices
        self.binary_chain_ops = binary_chain_ops
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from skimage.measure import label as sk_label, regionprops
            results = {
                'frame_idx': [], 'label': [],
                'area': [], 'equiv_diameter': [],
                'major_axis': [], 'minor_axis': [], 'eccentricity': [],
            }
            file_paths = [
                os.path.join(self.sequence.folder_path, f)
                for f in self.sequence.files
            ]
            thresh_op = self.pipeline.operations[self.thresh_idx]
            upstream_ops = [
                op for op in self.pipeline.operations[:self.thresh_idx]
                if op.enabled
            ]
            n = len(self.sample_indices)

            for i, frame_idx in enumerate(self.sample_indices):
                if self._cancelled:
                    return

                # Use batch cache when available
                cached = self.pipeline.get_batch_cache(self.thresh_idx, frame_idx)
                if cached is not None:
                    mask = cached.copy()
                    raw = None
                else:
                    raw = tifffile.imread(file_paths[frame_idx])
                    frame = raw
                    for op in upstream_ops:
                        frame = op.apply(frame)
                    mask = thresh_op.apply(frame)

                for op in self.binary_chain_ops:
                    if getattr(op, 'requires_intensity_context', False):
                        src_idx = op.params.get('intensity_source_idx', -1)
                        if raw is None:
                            raw = tifffile.imread(file_paths[frame_idx])
                        if src_idx == -1:
                            intensity_frame = self.pipeline.apply_to_frame(
                                raw, frame_idx, self.thresh_idx - 1)
                        else:
                            intensity_frame = self.pipeline.apply_to_frame(
                                raw, frame_idx, src_idx)
                        mask = op.apply(mask, context={'intensity_frame': intensity_frame})
                    else:
                        mask = op.apply(mask, context=None)

                labeled = sk_label(mask > 0)
                for region in regionprops(labeled):
                    results['frame_idx'].append(frame_idx)
                    results['label'].append(region.label)
                    results['area'].append(float(region.area))
                    results['equiv_diameter'].append(float(region.equivalent_diameter))
                    results['major_axis'].append(float(region.major_axis_length))
                    results['minor_axis'].append(float(region.minor_axis_length))
                    results['eccentricity'].append(float(region.eccentricity))

                self.progress.emit(i + 1, n)

            self.finished.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class RegionPropsWindow(QWidget):
    """Histogram + summary stats of per-blob regionprops across sampled frames.

    User selects which binary stage to analyze (AdaptiveThresholdOp output or
    after any subsequent enabled is_binary_mask_op op). Stats shown in px and
    physical units when a pixel scale is set.
    """

    _STAT_KEYS = [
        ("area",           "Area",              "px²",   2),
        ("equiv_diameter", "Equiv. Diameter",   "px",    1),
        ("major_axis",     "Major Axis",        "px",    1),
        ("minor_axis",     "Minor Axis",        "px",    1),
        ("eccentricity",   "Eccentricity",      "",      0),
    ]

    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.setWindowTitle("Region Props Analysis")
        self.setWindowFlags(Qt.WindowType.Tool)
        self.resize(720, 580)
        self._worker: RegionPropsWorker | None = None
        self._results: dict = {}
        self._binary_stage_ops: list = []  # ops to apply up to selected stage

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # --- Controls row 1: stage + sample ---
        ctrl1 = QHBoxLayout()
        ctrl1.addWidget(QLabel("Binary stage:"))
        self.stage_combo = QComboBox()
        self.stage_combo.setMinimumWidth(220)
        ctrl1.addWidget(self.stage_combo)
        ctrl1.addSpacing(12)
        ctrl1.addWidget(QLabel("Sample frames:"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 99999)
        self.spin_n.setValue(200)
        ctrl1.addWidget(self.spin_n)
        self.all_check = QCheckBox("All frames")
        self.all_check.toggled.connect(lambda c: self.spin_n.setEnabled(not c))
        ctrl1.addWidget(self.all_check)
        ctrl1.addStretch()
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self._run)
        ctrl1.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel)
        ctrl1.addWidget(self.cancel_btn)
        layout.addLayout(ctrl1)

        # --- Progress bar ---
        from PyQt6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # --- Stat selector row ---
        stat_row = QHBoxLayout()
        stat_row.addWidget(QLabel("Show stat:"))
        self.stat_combo = QComboBox()
        for _, label, _, _ in self._STAT_KEYS:
            self.stat_combo.addItem(label)
        self.stat_combo.currentIndexChanged.connect(self._redraw)
        stat_row.addWidget(self.stat_combo)
        stat_row.addStretch()
        self.log_y_check = QCheckBox("Log Y axis")
        self.log_y_check.toggled.connect(self._redraw)
        stat_row.addWidget(self.log_y_check)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_csv)
        stat_row.addWidget(self.export_btn)
        layout.addLayout(stat_row)

        # --- Histogram plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._bar_item: pg.BarGraphItem | None = None
        layout.addWidget(self.plot_widget, stretch=1)

        # --- Stats label ---
        self.stats_label = QLabel("Run analysis to see region property statistics.")
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

    # ------------------------------------------------------------------

    def _rebuild_stage_combo(self):
        """Populate stage combo from current pipeline binary chain."""
        app = self.main_app
        self.stage_combo.blockSignals(True)
        prev_text = self.stage_combo.currentText()
        self.stage_combo.clear()

        thresh_idx, thresh_op = app._find_op(AdaptiveThresholdOp)
        if thresh_op is None:
            self.stage_combo.addItem("(No AdaptiveThresholdOp)")
            self.stage_combo.blockSignals(False)
            return

        self.stage_combo.addItem("AdaptiveThresholdOp output")
        for op in app.pipeline.operations[thresh_idx + 1:]:
            if op.enabled and getattr(op, 'is_binary_mask_op', False):
                self.stage_combo.addItem(f"After {op.name}")

        idx = self.stage_combo.findText(prev_text)
        if idx >= 0:
            self.stage_combo.setCurrentIndex(idx)
        self.stage_combo.blockSignals(False)

    def _binary_ops_for_selected_stage(self):
        """Return binary chain ops to apply up to and including selected stage."""
        app = self.main_app
        thresh_idx, thresh_op = app._find_op(AdaptiveThresholdOp)
        if thresh_op is None:
            return []
        stage_text = self.stage_combo.currentText()
        if stage_text == "AdaptiveThresholdOp output":
            return []
        ops = []
        for op in app.pipeline.operations[thresh_idx + 1:]:
            if op.enabled and getattr(op, 'is_binary_mask_op', False):
                ops.append(op)
                if stage_text == f"After {op.name}":
                    break
        return ops

    def _run(self):
        app = self.main_app
        if app.sequence_manager is None:
            self.stats_label.setText("No sequence loaded.")
            return

        thresh_idx, thresh_op = app._find_op(AdaptiveThresholdOp)
        if thresh_op is None or not thresh_op.enabled:
            self.stats_label.setText("No enabled AdaptiveThresholdOp in pipeline.")
            return

        binary_ops = self._binary_ops_for_selected_stage()
        n_frames = app.sequence_manager.num_frames
        if self.all_check.isChecked():
            sample_indices = list(range(n_frames))
        else:
            n = min(self.spin_n.value(), n_frames)
            sample_indices = list(np.unique(
                np.round(np.linspace(0, n_frames - 1, n)).astype(int)
            ))

        self._worker = RegionPropsWorker(
            app.pipeline, app.sequence_manager,
            thresh_idx, sample_indices, binary_ops,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self.run_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setMaximum(len(sample_indices))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._results = {}
        self.export_btn.setEnabled(False)
        stage_label = self.stage_combo.currentText()
        self.stats_label.setText(
            f"Analysing {len(sample_indices)} frames "
            f"({stage_label})…"
        )
        self._worker.start()

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait()
            self._worker = None
        self._set_idle()
        self.stats_label.setText("Cancelled.")

    def _on_progress(self, done: int, total: int):
        self.progress_bar.setValue(done)

    def _on_finished(self, results: dict):
        self._results = {k: np.array(v) for k, v in results.items()}
        self._worker = None
        self._set_idle()
        self.export_btn.setEnabled(bool(self._results.get('area', []).__len__()))
        self._redraw()

    def _on_error(self, msg: str):
        self._worker = None
        self._set_idle()
        self.stats_label.setText(f"Error: {msg}")

    def _set_idle(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)

    def _scale_factor(self, dim_power: int):
        """Return (factor, unit_label) for converting px^dim_power → physical."""
        scale = self.main_app.pipeline.scale
        if scale is None or dim_power == 0:
            return None, None
        unit_per_px = scale["value"] / scale["px"]
        factor = unit_per_px ** dim_power
        unit = scale["unit"]
        if dim_power == 2:
            unit_label = f"{unit}²"
        else:
            unit_label = unit
        return factor, unit_label

    def _redraw(self):
        results = self._results
        if not results or 'area' not in results or len(results['area']) == 0:
            return

        stat_idx = self.stat_combo.currentIndex()
        key, label, px_unit, dim_power = self._STAT_KEYS[stat_idx]
        arr = results[key]

        # Histogram
        n_bins = int(np.clip(np.sqrt(len(arr)), 30, 300))
        counts, edges = np.histogram(arr, bins=n_bins)
        use_log = self.log_y_check.isChecked()
        y = np.log10(counts.astype(np.float64) + 1) if use_log else counts.astype(np.float64)

        if self._bar_item is not None:
            self.plot_widget.removeItem(self._bar_item)
        widths = np.diff(edges)
        self._bar_item = pg.BarGraphItem(
            x=edges[:-1], height=y, width=widths * 0.9,
            brush=(100, 180, 255, 180), pen=pg.mkPen(None),
        )
        self.plot_widget.addItem(self._bar_item)

        x_label = f"{label} ({px_unit})" if px_unit else label
        self.plot_widget.setLabel('bottom', x_label)
        self.plot_widget.setLabel('left', 'log₁₀(count+1)' if use_log else 'Count')

        # Stats
        factor, unit_label = self._scale_factor(dim_power)
        n_blobs = len(arr)
        mean_v = float(np.mean(arr))
        median_v = float(np.median(arr))
        std_v = float(np.std(arr))
        p95_v = float(np.percentile(arr, 95))
        p99_v = float(np.percentile(arr, 99))
        max_v = float(np.max(arr))

        def _fmtv(v):
            return f"{v:.3g}"

        if dim_power == 0:
            # Dimensionless (eccentricity) — single-unit display
            stats_text = (
                f"Blobs: {n_blobs:,}   |   "
                f"Mean: {_fmtv(mean_v)}   |   "
                f"Median: {_fmtv(median_v)}   |   "
                f"Std: {_fmtv(std_v)}   |   "
                f"P95: {_fmtv(p95_v)}   |   "
                f"P99: {_fmtv(p99_v)}   |   "
                f"Max: {_fmtv(max_v)}"
            )
        elif factor is not None:
            def _fmtu(v):
                return f"{v * factor:.3g}"
            stats_text = (
                f"Blobs: {n_blobs:,}   |   "
                f"Mean: {_fmtv(mean_v)} {px_unit} ({_fmtu(mean_v)} {unit_label})   |   "
                f"Median: {_fmtv(median_v)} {px_unit} ({_fmtu(median_v)} {unit_label})   |   "
                f"Std: {_fmtv(std_v)} {px_unit} ({_fmtu(std_v)} {unit_label})   |   "
                f"P95: {_fmtv(p95_v)} {px_unit} ({_fmtu(p95_v)} {unit_label})   |   "
                f"P99: {_fmtv(p99_v)} {px_unit} ({_fmtu(p99_v)} {unit_label})   |   "
                f"Max: {_fmtv(max_v)} {px_unit} ({_fmtu(max_v)} {unit_label})"
            )
            scale = self.main_app.pipeline.scale
            stats_text += (
                f"\nScale: {scale['px']} px = {_fmt_num(scale['value'])} {scale['unit']}   "
                f"→  1 px = {_fmt_num(scale['value'] / scale['px'])} {scale['unit']}"
            )
        else:
            stats_text = (
                f"Blobs: {n_blobs:,}   |   "
                f"Mean: {_fmtv(mean_v)} {px_unit}   |   "
                f"Median: {_fmtv(median_v)} {px_unit}   |   "
                f"Std: {_fmtv(std_v)} {px_unit}   |   "
                f"P95: {_fmtv(p95_v)} {px_unit}   |   "
                f"P99: {_fmtv(p99_v)} {px_unit}   |   "
                f"Max: {_fmtv(max_v)} {px_unit}"
            )
            stats_text += "\n(Set scale via Tools → Set Scale… to see physical units)"

        self.stats_label.setText(stats_text)

    def _export_csv(self):
        results = self._results
        if not results or 'area' not in results:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Region Props CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        scale = self.main_app.pipeline.scale
        if scale is not None:
            upp = scale["value"] / scale["px"]  # units per pixel
            unit = scale["unit"]
        else:
            upp = None
            unit = None

        n = len(results['area'])
        lines = []
        if upp is not None:
            lines.append(
                "frame_idx,label,"
                f"area_px2,area_{unit}2,"
                f"equiv_diam_px,equiv_diam_{unit},"
                f"major_axis_px,major_axis_{unit},"
                f"minor_axis_px,minor_axis_{unit},"
                "eccentricity"
            )
        else:
            lines.append(
                "frame_idx,label,"
                "area_px2,equiv_diam_px,major_axis_px,minor_axis_px,eccentricity"
            )

        for i in range(n):
            fi = int(results['frame_idx'][i])
            lbl = int(results['label'][i])
            area = results['area'][i]
            ed = results['equiv_diameter'][i]
            maj = results['major_axis'][i]
            mn = results['minor_axis'][i]
            ecc = results['eccentricity'][i]
            if upp is not None:
                lines.append(
                    f"{fi},{lbl},"
                    f"{area:.4g},{area * upp**2:.4g},"
                    f"{ed:.4g},{ed * upp:.4g},"
                    f"{maj:.4g},{maj * upp:.4g},"
                    f"{mn:.4g},{mn * upp:.4g},"
                    f"{ecc:.6f}"
                )
            else:
                lines.append(
                    f"{fi},{lbl},{area:.4g},{ed:.4g},{maj:.4g},{mn:.4g},{ecc:.6f}"
                )

        try:
            with open(path, 'w') as f:
                f.write("\n".join(lines))
            self.stats_label.setText(
                self.stats_label.text() + f"\nExported {n} rows → {path}"
            )
        except OSError as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def showEvent(self, event):
        super().showEvent(event)
        self._rebuild_stage_combo()
        if self._results:
            self._redraw()


# ---------------------------------------------------------------------------
# MieAnalysisWindow — particle sizing from scattering cross-section
# ---------------------------------------------------------------------------

class MieComputeWorker(QThread):
    """Runs mie.analyze() off the GUI thread. The forward Mie sweep is O(num_a ·
    n_max) and can take tens of seconds for large particles / fine grids, so it
    must never block the UI."""
    finished = pyqtSignal(object)   # mie.MieResult
    error = pyqtSignal(str)

    def __init__(self, areas_m2, n_real, n_imag, a_start, a_stop,
                 lam_nm, n_medium, mu, num_a, inversion):
        super().__init__()
        self._args = (areas_m2, n_real, n_imag, a_start, a_stop)
        self._kw = dict(lam_nm=lam_nm, n_medium=n_medium, mu=mu, num_a=num_a,
                        inversion=inversion)

    def run(self):
        try:
            import mie
            res = mie.analyze(*self._args, **self._kw)
            self.finished.emit(res)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")



class MieAnalysisWindow(QWidget):
    """Mie-scattering particle sizing.

    Treats each measured blob area (in m²) as that particle's scattering
    cross-section C_sca, inverts a forward Mie curve to per-particle radii, and
    fits a Rosin-Rammler CDF for D10/D50/D90. See mie.py.

    Method from:
      Dasgupta, Raut, Vadukut, Bose, "Design and Development of a Pneumatic
      Atomizer for Seeding Tracers in PIV Experiments", J. Flow Visualization
      and Image Processing 32(3), 2025.

    Areas come from either the in-memory Region Props path (px² → m² via the
    Ctrl+M pixel scale) or a loaded CSV whose areas are already physical.
    """

    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.setWindowTitle("Mie Particle Sizing")
        self.setWindowFlags(Qt.WindowType.Tool)
        self.resize(760, 680)
        self._worker: RegionPropsWorker | None = None
        self._mie_worker: MieComputeWorker | None = None
        self._areas_m2: np.ndarray | None = None   # particle areas (== C_sca), m²
        self._areas_source: str = ""               # human-readable provenance
        self._result = None                         # mie.MieResult

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # --- Area source row ---
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Areas from:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Region Props (in-memory)", "CSV file"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        src_row.addWidget(self.source_combo)
        src_row.addSpacing(8)
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 99999)
        self.spin_n.setValue(200)
        self.sample_lbl = QLabel("Sample frames:")
        src_row.addWidget(self.sample_lbl)
        src_row.addWidget(self.spin_n)
        self.all_check = QCheckBox("All frames")
        self.all_check.toggled.connect(lambda c: self.spin_n.setEnabled(not c))
        src_row.addWidget(self.all_check)
        self.load_csv_btn = QPushButton("Load CSV…")
        self.load_csv_btn.setVisible(False)
        self.load_csv_btn.clicked.connect(self._load_csv)
        src_row.addWidget(self.load_csv_btn)
        self.csv_unit_combo = QComboBox()
        self.csv_unit_combo.addItems(["mm", "µm", "m", "cm", "nm"])
        self.csv_unit_combo.setVisible(False)
        self.csv_unit_lbl = QLabel("CSV area unit:")
        self.csv_unit_lbl.setVisible(False)
        src_row.addWidget(self.csv_unit_lbl)
        src_row.addWidget(self.csv_unit_combo)
        src_row.addStretch()
        layout.addLayout(src_row)

        # --- Optical parameters row ---
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("n (real):"))
        self.n_real = QLineEdit("1.5")
        self.n_real.setValidator(QDoubleValidator())
        self.n_real.setMaximumWidth(70)
        opt_row.addWidget(self.n_real)
        opt_row.addWidget(QLabel("n (imag):"))
        self.n_imag = QLineEdit("0.0")
        self.n_imag.setValidator(QDoubleValidator())
        self.n_imag.setMaximumWidth(70)
        opt_row.addWidget(self.n_imag)
        opt_row.addWidget(QLabel("λ (nm):"))
        self.lam = QLineEdit("527")
        self.lam.setValidator(QDoubleValidator(1.0, 1e5, 3))
        self.lam.setMaximumWidth(70)
        opt_row.addWidget(self.lam)
        opt_row.addWidget(QLabel("n medium:"))
        self.n_medium = QLineEdit("1.0")
        self.n_medium.setValidator(QDoubleValidator(1e-3, 10.0, 4))
        self.n_medium.setMaximumWidth(70)
        opt_row.addWidget(self.n_medium)
        opt_row.addWidget(QLabel("μ:"))
        self.mu = QLineEdit("1.0")
        self.mu.setValidator(QDoubleValidator())
        self.mu.setMaximumWidth(60)
        opt_row.addWidget(self.mu)
        opt_row.addSpacing(8)
        opt_row.addWidget(QLabel("inversion:"))
        self.inversion_combo = QComboBox()
        self.inversion_combo.addItem("Smoothed trend", "trend")
        self.inversion_combo.addItem("Legacy (MATLAB unique)", "legacy")
        self.inversion_combo.setToolTip(
            "Smoothed trend: grid-stable, inverts the monotone mean of C_sca(a).\n"
            "Legacy: original MATLAB unique() — grid-sensitive in the ripple regime,\n"
            "kept only to reproduce prior runs."
        )
        opt_row.addWidget(self.inversion_combo)
        opt_row.addStretch()
        layout.addLayout(opt_row)

        # --- Radius sweep row ---
        rad_row = QHBoxLayout()
        rad_row.addWidget(QLabel("Radius start (m):"))
        self.a_start = QLineEdit("")
        self.a_start.setValidator(QDoubleValidator(0.0, 1.0, 12))
        self.a_start.setMaximumWidth(110)
        rad_row.addWidget(self.a_start)
        rad_row.addWidget(QLabel("stop (m):"))
        self.a_stop = QLineEdit("")
        self.a_stop.setValidator(QDoubleValidator(0.0, 1.0, 12))
        self.a_stop.setMaximumWidth(110)
        rad_row.addWidget(self.a_stop)
        self.auto_btn = QPushButton("Auto range")
        self.auto_btn.setToolTip("Seed radius range from the measured areas (geometric estimate).")
        self.auto_btn.clicked.connect(self._auto_range)
        rad_row.addWidget(self.auto_btn)
        rad_row.addSpacing(8)
        rad_row.addWidget(QLabel("grid:"))
        self.num_a = QLineEdit("50000")
        self.num_a.setValidator(QIntValidator(100, 500000))
        self.num_a.setMaximumWidth(70)
        self.num_a.setToolTip(
            "Radius sweep resolution. 50000 matches the original MATLAB (accurate "
            "but slow); lower it to trade fidelity for speed."
        )
        rad_row.addWidget(self.num_a)
        rad_row.addStretch()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        rad_row.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel)
        rad_row.addWidget(self.cancel_btn)
        layout.addLayout(rad_row)

        from PyQt6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # --- Plots ---
        self.csca_plot = pg.PlotWidget()
        self.csca_plot.showGrid(x=True, y=True, alpha=0.3)
        self.csca_plot.setLabel('bottom', 'radius a (m)')
        self.csca_plot.setLabel('left', 'C_sca (m²)')
        self.csca_plot.setLogMode(x=True, y=True)
        layout.addWidget(self.csca_plot, stretch=1)

        self.cdf_plot = pg.PlotWidget()
        self.cdf_plot.showGrid(x=True, y=True, alpha=0.3)
        self.cdf_plot.setLabel('bottom', 'radius a (m)')
        self.cdf_plot.setLabel('left', 'cumulative fraction')
        layout.addWidget(self.cdf_plot, stretch=1)

        self.export_btn = QPushButton("Export radii CSV…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_csv)
        layout.addWidget(self.export_btn)

        self.stats_label = QLabel(
            "Load areas and Run. Each blob area (m²) is treated as its scattering "
            "cross-section C_sca. In-memory areas need a pixel scale (Ctrl+M)."
        )
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        cite = QLabel(
            "Method: Dasgupta et al., J. Flow Visualization and Image Processing "
            "32(3), 2025."
        )
        cite.setStyleSheet("color: gray; font-size: 10px;")
        cite.setWordWrap(True)
        layout.addWidget(cite)

        self._on_source_changed()

    # ------------------------------------------------------------------

    def _on_source_changed(self):
        csv = self.source_combo.currentText() == "CSV file"
        self.load_csv_btn.setVisible(csv)
        self.csv_unit_combo.setVisible(csv)
        self.csv_unit_lbl.setVisible(csv)
        self.sample_lbl.setVisible(not csv)
        self.spin_n.setVisible(not csv)
        self.all_check.setVisible(not csv)

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load areas CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            areas = self._read_area_column(path)
        except Exception as e:
            QMessageBox.critical(self, "CSV Load Failed", str(e))
            return
        import mie
        factor = mie.area_unit_to_m2_factor(self.csv_unit_combo.currentText())
        self._areas_m2 = areas * factor
        self._areas_source = f"{len(areas)} areas from {os.path.basename(path)} " \
                             f"({self.csv_unit_combo.currentText()}²)"
        self.stats_label.setText(f"Loaded {self._areas_source}. Set params and Run.")
        self._auto_range()

    @staticmethod
    def _read_area_column(path: str) -> np.ndarray:
        """Read an 'Area' column from a CSV. Case-insensitive header match; falls
        back to the first numeric column if no 'area' header exists."""
        import csv as _csv
        with open(path, newline="") as f:
            reader = _csv.reader(f)
            rows = [r for r in reader if r]
        if not rows:
            raise ValueError("empty CSV")
        header = [h.strip().lower() for h in rows[0]]
        col = None
        for i, h in enumerate(header):
            if h == "area" or h.startswith("area"):
                col = i
                break
        data_rows = rows[1:] if col is not None else rows
        if col is None:
            col = 0  # no header match — assume first column is area
        vals = []
        for r in data_rows:
            if col >= len(r):
                continue
            try:
                vals.append(float(r[col]))
            except ValueError:
                continue
        if not vals:
            raise ValueError("no numeric area values found")
        return np.asarray(vals, dtype=float)

    def _auto_range(self):
        """Seed radius sweep from measured areas via geometric estimate a≈√(area/π)."""
        if self._areas_m2 is None or self._areas_m2.size == 0:
            return
        a_geom = np.sqrt(np.clip(self._areas_m2, 0, None) / np.pi)
        a_geom = a_geom[a_geom > 0]
        if a_geom.size == 0:
            return
        # Geometric radius already slightly over-estimates the true radius (in the
        # optics limit C_sca→2πa², so a≈√(area/2π) < √(area/π)); a modest bracket
        # is enough and keeps the size parameter — hence n_max and runtime — small.
        a_start = max(a_geom.min() * 0.5, 1e-10)
        a_stop = a_geom.max() * 1.5
        self.a_start.setText(f"{a_start:.4e}")
        self.a_stop.setText(f"{a_stop:.4e}")

    # --- in-memory area collection (reuses RegionPropsWorker) ----------

    def _collect_inmemory(self):
        app = self.main_app
        if app.sequence_manager is None:
            self.stats_label.setText("No sequence loaded.")
            return False
        if app.pipeline.scale is None:
            self.stats_label.setText(
                "No pixel scale set. Set one via Ctrl+M (Tools → Set Scale…) so "
                "areas can be converted to m², or switch to a CSV source."
            )
            return False
        thresh_idx, thresh_op = app._find_op(AdaptiveThresholdOp)
        if thresh_op is None or not thresh_op.enabled:
            self.stats_label.setText("No enabled AdaptiveThresholdOp in pipeline.")
            return False

        binary_ops = [
            op for op in app.pipeline.operations[thresh_idx + 1:]
            if op.enabled and getattr(op, 'is_binary_mask_op', False)
        ]
        n_frames = app.sequence_manager.num_frames
        if self.all_check.isChecked():
            sample_indices = list(range(n_frames))
        else:
            n = min(self.spin_n.value(), n_frames)
            sample_indices = list(np.unique(
                np.round(np.linspace(0, n_frames - 1, n)).astype(int)
            ))

        self._worker = RegionPropsWorker(
            app.pipeline, app.sequence_manager, thresh_idx, sample_indices, binary_ops,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_areas_ready)
        self._worker.error.connect(self._on_error)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setMaximum(len(sample_indices))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.stats_label.setText(f"Collecting areas from {len(sample_indices)} frames…")
        self._worker.start()
        return True

    def _on_areas_ready(self, results: dict):
        self._worker = None
        self._set_idle()
        areas_px2 = np.asarray(results.get('area', []), dtype=float)
        if areas_px2.size == 0:
            self.stats_label.setText("No blobs found in sampled frames.")
            return
        import mie
        scale = self.main_app.pipeline.scale
        unit_per_px = scale["value"] / scale["px"]
        unit_factor = mie.area_unit_to_m2_factor(scale["unit"])
        if unit_factor is None:
            self.stats_label.setText(
                f"Scale unit '{scale['unit']}' not recognised (need m/cm/mm/µm/nm)."
            )
            return
        # px² → (scale unit)² → m²
        self._areas_m2 = areas_px2 * (unit_per_px ** 2) * unit_factor
        self._areas_source = f"{areas_px2.size} blobs (in-memory, scale {scale['unit']})"
        if not self.a_start.text() or not self.a_stop.text():
            self._auto_range()
        self._compute()

    # --- run orchestration ---------------------------------------------

    def _run(self):
        if self.source_combo.currentText() == "CSV file":
            if self._areas_m2 is None:
                self.stats_label.setText("Load a CSV first.")
                return
            self._compute()
        else:
            self._collect_inmemory()

    def _compute(self):
        if self._areas_m2 is None or self._areas_m2.size == 0:
            self.stats_label.setText("No areas to analyse.")
            return
        try:
            a_start = float(self.a_start.text())
            a_stop = float(self.a_stop.text())
            params = dict(
                n_real=float(self.n_real.text()), n_imag=float(self.n_imag.text()),
                lam_nm=float(self.lam.text()), n_medium=float(self.n_medium.text()),
                mu=float(self.mu.text()), num_a=int(self.num_a.text()),
            )
        except ValueError as e:
            self.stats_label.setText(f"Invalid parameter: {e}")
            return

        # Forward Mie is heavy — run it off the GUI thread.
        self._mie_worker = MieComputeWorker(
            self._areas_m2, params["n_real"], params["n_imag"], a_start, a_stop,
            params["lam_nm"], params["n_medium"], params["mu"], params["num_a"],
            self.inversion_combo.currentData(),
        )
        self._mie_worker.finished.connect(self._on_mie_finished)
        self._mie_worker.error.connect(self._on_error)
        self.run_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)   # indeterminate — compute isn't chunked
        self.progress_bar.setVisible(True)
        self.stats_label.setText(
            f"Computing Mie inversion for {self._areas_m2.size} particles "
            f"(grid {params['num_a']})…"
        )
        self._mie_worker.start()

    def _on_mie_finished(self, res):
        self._mie_worker = None
        self.progress_bar.setRange(0, 100)
        self._set_idle()
        self._result = res
        self.export_btn.setEnabled(True)
        self._redraw()

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait()
            self._worker = None
        self._set_idle()
        self.stats_label.setText("Cancelled.")

    def _on_progress(self, done: int, total: int):
        self.progress_bar.setValue(done)

    def _on_error(self, msg: str):
        self._worker = None
        self._mie_worker = None
        self.progress_bar.setRange(0, 100)
        self._set_idle()
        self.stats_label.setText(f"Error: {msg}")

    def _set_idle(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)

    def _redraw(self):
        res = self._result
        if res is None:
            return
        self.csca_plot.clear()
        self.csca_plot.plot(res.a, res.csca, pen=pg.mkPen((0, 114, 189), width=2))

        self.cdf_plot.clear()
        self.cdf_plot.plot(
            res.cdf_bin_centers, res.cdf_values,
            pen=None, symbol='o', symbolSize=4,
            symbolBrush=(150, 150, 150), name='empirical',
        )
        a = res.a
        rr = 1.0 - np.exp(-((a / res.rr_b) ** res.rr_c))
        self.cdf_plot.plot(a, rr, pen=pg.mkPen((217, 83, 25), width=2), name='Rosin-Rammler')
        for d, col in ((res.D10, (0, 0, 255)), (res.D50, (0, 160, 0)), (res.D90, (255, 0, 0))):
            self.cdf_plot.addItem(pg.InfiniteLine(pos=d, angle=90, pen=pg.mkPen(col, style=Qt.PenStyle.DashLine)))

        def um(v):
            return f"{v * 1e6:.3g}"
        stats = (
            f"{self._areas_source}   |   n = {complex(float(self.n_real.text()), float(self.n_imag.text()))}\n"
            f"D10 = {res.D10:.3e} m ({um(res.D10)} µm)   |   "
            f"D50 = {res.D50:.3e} m ({um(res.D50)} µm)   |   "
            f"D90 = {res.D90:.3e} m ({um(res.D90)} µm)\n"
            f"Rosin-Rammler:  b = {res.rr_b:.3e}   c = {res.rr_c:.3g}   R² = {res.r2:.4f}   "
            f"(n particles = {res.radii.size})"
        )
        self.stats_label.setText(stats)

    def _export_csv(self):
        res = self._result
        if res is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export radii CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return
        lines = [
            f"# Mie particle sizing — Dasgupta et al., J. Flow Vis. Image Proc. 32(3), 2025",
            f"# n_sphere={res.n_sphere}, lambda_nm={res.lam_nm}, n_medium={res.n_medium}, mu={res.mu}",
            f"# D10_m={res.D10:.6e}, D50_m={res.D50:.6e}, D90_m={res.D90:.6e}, "
            f"RR_b={res.rr_b:.6e}, RR_c={res.rr_c:.6e}, R2={res.r2:.6f}",
            "particle_idx,area_m2,radius_m,radius_um",
        ]
        for i, (area, r) in enumerate(zip(self._areas_m2, res.radii)):
            lines.append(f"{i},{area:.6e},{r:.6e},{r * 1e6:.6g}")
        try:
            with open(path, 'w') as f:
                f.write("\n".join(lines))
            self.stats_label.setText(
                self.stats_label.text() + f"\nExported {res.radii.size} radii → {path}"
            )
        except OSError as e:
            QMessageBox.critical(self, "Export Failed", str(e))


# ---------------------------------------------------------------------------
# ContrastToolWindow (unchanged)
# ---------------------------------------------------------------------------

class ContrastToolWindow(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.setWindowTitle("Contrast & Histogram Tools")
        self.setGeometry(1150, 100, 450, 650)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        self.hist_plot = pg.PlotWidget(title="Raw Intensity Histogram")
        self.hist_plot.setLabel('left', "Pixel Count")
        self.hist_plot.setLabel('bottom', "Raw Intensity")
        self.hist_plot.setBackground((0, 0, 0, 0))
        self.hist_plot.showGrid(x=True, y=True, alpha=0.2)
        self.hist_plot.setMouseEnabled(x=False, y=False)
        self.hist_plot.hideButtons()
        layout.addWidget(self.hist_plot)

        self.hist_curve = self.hist_plot.plot(
            pen=(100, 200, 255), stepMode=True, fillLevel=0, brush=(100, 200, 255, 80)
        )

        self.region = pg.LinearRegionItem(
            [self.main_app.vmin, self.main_app.vmax],
            bounds=[0, self.main_app.bit_depth_max],
            brush=(0, 0, 0, 0),
            pen=pg.mkPen(color=(0, 200, 255), width=2),
            hoverPen=pg.mkPen(color=(255, 255, 255), width=4),
        )
        self.region.sigRegionChanged.connect(self.on_region_changed)
        self.hist_plot.addItem(self.region)

        self.curve_plot = pg.PlotWidget(title="8-Bit Mapping Preview")
        self.curve_plot.setLabel('left', "Output (0-255)")
        self.curve_plot.setLabel('bottom', "Input Intensity")
        self.curve_plot.setBackground((0, 0, 0, 0))
        self.curve_plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve_plot.setMouseEnabled(x=False, y=False)
        self.curve_plot.hideButtons()
        self.curve_plot.setYRange(-10, 265, padding=0)
        layout.addWidget(self.curve_plot)

        self.curve_line = self.curve_plot.plot(pen=pg.mkPen(color=(255, 200, 100), width=3))

        gamma_layout = QVBoxLayout()
        self.gamma_label = QLabel("Gamma Curve: 1.00")
        self.gamma_label.setStyleSheet("padding-top: 10px; padding-bottom: 5px;")
        gamma_layout.addWidget(self.gamma_label)
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(1, 50)
        self.gamma_slider.setValue(10)
        self.gamma_slider.valueChanged.connect(self.on_gamma_changed)
        gamma_layout.addWidget(self.gamma_slider)
        layout.addLayout(gamma_layout)
        self._updating = False

    def on_region_changed(self):
        if self._updating:
            return
        vmin, vmax = self.region.getRegion()
        self.main_app.vmin = vmin
        self.main_app.vmax = vmax
        self.update_curve_preview()
        self.main_app.update_frame_display()

    def on_gamma_changed(self, val):
        gamma = val / 10.0
        self.gamma_label.setText(f"Gamma Curve: {gamma:.2f}")
        self.main_app.gamma = gamma
        self.update_curve_preview()
        self.main_app.update_frame_display()

    def sync_from_main(self):
        self._updating = True
        self.region.setBounds([0, self.main_app.bit_depth_max])
        self.region.setRegion([self.main_app.vmin, self.main_app.vmax])
        self.hist_plot.setXRange(0, self.main_app.bit_depth_max, padding=0.05)
        self.update_curve_preview()
        self._updating = False

    def update_histogram(self, raw_frame):
        # 4× subsample for large frames — negligible accuracy loss for display histogram
        sample = raw_frame[::2, ::2] if raw_frame.size > 262144 else raw_frame
        y, x = np.histogram(sample, bins=250, range=(0, self.main_app.bit_depth_max))
        self.hist_curve.setData(x, y)

    def update_curve_preview(self):
        vmin, vmax = self.main_app.vmin, self.main_app.vmax
        gamma = self.main_app.gamma
        if vmax <= vmin:
            vmax = vmin + 1e-8
        x = np.linspace(0, self.main_app.bit_depth_max, 300)
        clipped = np.clip(x, vmin, vmax)
        normalized = (clipped - vmin) / (vmax - vmin)
        y = np.power(normalized, gamma) * 255.0
        self.curve_line.setData(x, y)
        self.curve_plot.setXRange(0, self.main_app.bit_depth_max, padding=0.05)


# ---------------------------------------------------------------------------
# CropParamsWidget — 2×2 grid embedded in PipelinePanel params area
# ---------------------------------------------------------------------------

class CropParamsWidget(QWidget):
    def __init__(self, main_app, op, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.op = op
        self._updating = False

        grid = QVBoxLayout(self)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(4)

        from PyQt6.QtWidgets import QGridLayout
        self._grid = QGridLayout()
        self._grid.setSpacing(4)

        self.spin_x = QSpinBox(); self.spin_x.setRange(0, 99999)
        self.spin_y = QSpinBox(); self.spin_y.setRange(0, 99999)
        self.spin_w = QSpinBox(); self.spin_w.setRange(1, 99999)
        self.spin_h = QSpinBox(); self.spin_h.setRange(1, 99999)

        for spin, label, val in [
            (self.spin_x, "X", op.params["x"]),
            (self.spin_y, "Y", op.params["y"]),
            (self.spin_w, "W", op.params["w"]),
            (self.spin_h, "H", op.params["h"]),
        ]:
            spin.setValue(val)

        lbl_x = QLabel("X:"); lbl_y = QLabel("Y:")
        lbl_w = QLabel("W:"); lbl_h = QLabel("H:")

        self._grid.addWidget(lbl_x,       0, 0)
        self._grid.addWidget(self.spin_x, 0, 1)
        self._grid.addWidget(lbl_y,       0, 2)
        self._grid.addWidget(self.spin_y, 0, 3)
        self._grid.addWidget(lbl_w,       1, 0)
        self._grid.addWidget(self.spin_w, 1, 1)
        self._grid.addWidget(lbl_h,       1, 2)
        self._grid.addWidget(self.spin_h, 1, 3)

        grid.addLayout(self._grid)

        for spin in (self.spin_x, self.spin_y, self.spin_w, self.spin_h):
            spin.editingFinished.connect(self._on_edited)

    def _on_edited(self):
        if self._updating:
            return
        self.main_app._apply_crop_from_panel(
            self.spin_x.value(), self.spin_y.value(),
            self.spin_w.value(), self.spin_h.value(),
        )

    def sync(self, x, y, w, h):
        self._updating = True
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)
        self.spin_w.setValue(w)
        self.spin_h.setValue(h)
        self._updating = False

    def update_bounds(self, img_w, img_h):
        self.spin_x.setRange(0, max(0, img_w - 1))
        self.spin_y.setRange(0, max(0, img_h - 1))
        self.spin_w.setRange(1, img_w)
        self.spin_h.setRange(1, img_h)


# ---------------------------------------------------------------------------
# IntensityWatershedParamsWidget — custom params panel for IntensityWatershedSplitOp
# ---------------------------------------------------------------------------

class IntensityWatershedParamsWidget(QWidget):
    """Params editor for IntensityWatershedSplitOp.

    Replaces the generic ParamsWidget so that the ``intensity_source_idx``
    parameter is shown as a human-readable dynamic combo whose choices are
    derived from the current pipeline state (stages upstream of
    AdaptiveThresholdOp), rather than a raw integer spinbox.

    The other two params (min_peak_distance, threshold_abs) are regular
    spinboxes.

    Design notes:
    - ``_updating`` flag prevents re-entrant signal → _emit loops.
    - Calls ``on_change()`` which is ``PipelinePanel._on_param_changed``;
      that is safe to call from here (uses update_stale_colors, not rebuild_list).
    - The combo is populated once at widget-creation time (when _show_params
      is called).  If the pipeline changes structurally, _show_params recreates
      this widget with fresh choices.
    """

    def __init__(self, main_app, op: IntensityWatershedSplitOp, on_change, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.op = op
        self.on_change = on_change
        self._updating = False
        # _source_choices: list of (display_label, intensity_source_idx) tuples
        self._source_choices: list[tuple[str, int]] = []

        layout = QFormLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- dynamic intensity-source combo ---
        self.source_combo = QComboBox()
        self.source_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.source_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._populate_source_combo()
        layout.addRow("Intensity source:", self.source_combo)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

        # --- min_peak_distance ---
        self.spin_min_dist = QSpinBox()
        self.spin_min_dist.setRange(1, 50)
        self.spin_min_dist.setValue(op.params["min_peak_distance"])
        self.spin_min_dist.valueChanged.connect(
            lambda v: self._emit("min_peak_distance", v)
        )
        layout.addRow("Min peak distance (px):", self.spin_min_dist)

        # --- threshold_abs ---
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 254)
        self.spin_threshold.setValue(op.params["threshold_abs"])
        self.spin_threshold.valueChanged.connect(
            lambda v: self._emit("threshold_abs", v)
        )
        layout.addRow("Min peak intensity (0-255):", self.spin_threshold)

        # --- radius_scale ---
        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(50, 200)
        self.spin_radius.setSuffix(" %")
        self.spin_radius.setValue(op.params.get("radius_scale", 100))
        self.spin_radius.setToolTip(
            "Scale factor applied to the area-derived circle radius.\n"
            "100 % = area-preserving (r = sqrt(blob_area / N·π)).\n"
            "Increase if circles appear too small; decrease if they bleed into neighbours."
        )
        self.spin_radius.valueChanged.connect(
            lambda v: self._emit("radius_scale", v)
        )
        layout.addRow("Radius scale (%):", self.spin_radius)

        # --- min_blob_area ---
        self.spin_min_blob_area = QSpinBox()
        self.spin_min_blob_area.setRange(1, 100000)
        self.spin_min_blob_area.setSuffix(" px²")
        self.spin_min_blob_area.setValue(op.params.get("min_blob_area", 50))
        self.spin_min_blob_area.setToolTip(
            "Blobs smaller than this area are copied unchanged (orange).\n"
            "Only blobs at or above this size are analysed for merged particles.\n"
            "Large blobs with one peak → green.  Split blobs → cyan.\n"
            "Increase to skip more small particles and improve performance."
        )
        self.spin_min_blob_area.valueChanged.connect(
            lambda v: self._emit("min_blob_area", v)
        )
        layout.addRow("Min blob area for splitting:", self.spin_min_blob_area)

    # ------------------------------------------------------------------

    def _populate_source_combo(self):
        """Build combo choices from shape-compatible stages upstream of AdaptiveThresholdOp.

        Only ops whose output has the same H×W as the binary mask are listed.
        That means ops at indices strictly after the last RotateOp / CropOp in the
        pipeline (those are the shape-changing ops).  The sentinel value −1 means
        "pre-threshold frame" — the exact frame fed into AdaptiveThresholdOp —
        which is always shape-compatible by definition.
        """
        self._updating = True
        self.source_combo.clear()
        # -1 = pre-threshold frame (display_frame fed to AdaptiveThresholdOp).
        # Always shape-compatible; best default for most use-cases.
        self._source_choices = [("Pre-threshold frame (default)", -1)]

        pipeline = self.main_app.pipeline
        thresh_idx = -1
        for i, op in enumerate(pipeline.operations):
            if isinstance(op, AdaptiveThresholdOp):
                thresh_idx = i
                break

        limit = thresh_idx if thresh_idx >= 0 else len(pipeline.operations)

        # Find the last enabled shape-changing op before AdaptiveThresholdOp.
        # Ops at index ≤ last_shape_idx may change H or W; skip them.
        _SHAPE_OPS = (RotateOp, CropOp)
        last_shape_idx = -1
        for i, op in enumerate(pipeline.operations[:limit]):
            if isinstance(op, _SHAPE_OPS) and op.enabled:
                last_shape_idx = i

        # Only list ops after the last shape-changing op.
        for i, op in enumerate(pipeline.operations[:limit]):
            if i > last_shape_idx:
                self._source_choices.append((f"{i}: {op.name}", i))

        for label, _ in self._source_choices:
            self.source_combo.addItem(label)

        # Restore current selection; reset to -1 if stored index is no longer valid
        current = self.op.params.get("intensity_source_idx", -1)
        matched = False
        for j, (_, src_idx) in enumerate(self._source_choices):
            if src_idx == current:
                self.source_combo.setCurrentIndex(j)
                matched = True
                break
        if not matched:
            self.source_combo.setCurrentIndex(0)
            self.op.params["intensity_source_idx"] = -1

        self._updating = False

    def _on_source_changed(self, combo_idx: int):
        if self._updating:
            return
        if 0 <= combo_idx < len(self._source_choices):
            _, src_idx = self._source_choices[combo_idx]
            self.op.params["intensity_source_idx"] = src_idx
            self.on_change()

    def _emit(self, key: str, val):
        if self._updating:
            return
        self.op.params[key] = val
        self.on_change()


# ---------------------------------------------------------------------------
# PipelinePanel — new floating tool window
# ---------------------------------------------------------------------------

class ParamsWidget(QWidget):
    """Auto-generated param editor for a single Operation."""

    def __init__(self, op, on_change, parent=None):
        super().__init__(parent)
        self.op = op
        self.on_change = on_change
        self._updating = False
        self._widgets = {}

        layout = QFormLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        for spec in op.params_schema:
            key = spec["key"]
            widget_hint = spec.get("widget", "spinbox")
            current = op.params[key]

            if widget_hint == "spinbox":
                w = QSpinBox()
                lo, hi = spec.get("range", [0, 99999])
                w.setRange(lo, hi)
                step = spec.get("step", 1)
                w.setSingleStep(step)
                w.setValue(current)
                w.valueChanged.connect(lambda val, k=key: self._emit(k, val))
            elif widget_hint == "combo":
                w = QComboBox()
                w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
                w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                choices = spec.get("choices", [])
                w.addItems(choices)
                if current in choices:
                    w.setCurrentIndex(choices.index(current))
                w.currentIndexChanged.connect(
                    lambda idx, k=key, c=choices: self._emit(k, c[idx])
                )
            else:
                w = QLabel(str(current))

            self._widgets[key] = w
            layout.addRow(spec.get("label", key) + ":", w)

    def _emit(self, key, val):
        if self._updating:
            return
        self.op.params[key] = val
        self.on_change()

    def refresh(self):
        self._updating = True
        for spec in self.op.params_schema:
            key = spec["key"]
            w = self._widgets.get(key)
            val = self.op.params[key]
            if isinstance(w, QSpinBox):
                w.setValue(val)
            elif isinstance(w, QComboBox):
                choices = spec.get("choices", [])
                if val in choices:
                    w.setCurrentIndex(choices.index(val))
        self._updating = False


class PipelinePanel(QWidget):
    """Floating tool window showing the ordered operation list."""

    def __init__(self, main_app):
        super().__init__(main_app, Qt.WindowType.Tool)
        self.main_app = main_app
        self.setWindowTitle("Pipeline")
        self.setGeometry(1150, 100, 340, 520)
        self._selected_idx = -1

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # --- stage list ---
        list_label = QLabel("Stages (drag to reorder):")
        outer.addWidget(list_label)

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list_widget.setMaximumHeight(160)
        self.list_widget.currentRowChanged.connect(self._on_row_selected)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        outer.addWidget(self.list_widget)

        # --- add / remove buttons ---
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("+  Add Stage")
        self.remove_btn = QPushButton("−  Remove")
        self.remove_btn.setEnabled(False)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        outer.addLayout(btn_row)
        self.add_btn.clicked.connect(self._on_add)
        self.remove_btn.clicked.connect(self._on_remove)

        # --- display selector ---
        disp_box = QGroupBox("Display")
        disp_layout = QVBoxLayout(disp_box)
        disp_layout.setSpacing(4)
        self.base_label = QLabel("Base stage: (all)")
        disp_layout.addWidget(self.base_label)
        self.overlay_label = QLabel("Overlay stage: (none)")
        disp_layout.addWidget(self.overlay_label)
        outer.addWidget(disp_box)

        # --- params area ---
        self.params_group = QGroupBox("Stage Params")
        self.params_group.setMinimumHeight(120)
        params_outer = QVBoxLayout(self.params_group)
        params_outer.setContentsMargins(4, 4, 4, 4)

        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        params_outer.addWidget(self.params_scroll)
        self._show_params(None)  # set initial placeholder

        outer.addWidget(self.params_group)

        # --- fit / apply-all buttons ---
        action_row = QHBoxLayout()
        self.fit_btn = QPushButton("Fit (learn from sequence)")
        self.fit_btn.setEnabled(False)
        self.apply_all_btn = QPushButton("Apply to All Frames")
        self.apply_all_btn.setEnabled(False)
        action_row.addWidget(self.fit_btn)
        action_row.addWidget(self.apply_all_btn)
        outer.addLayout(action_row)
        self.fit_btn.clicked.connect(self._on_fit)
        self.apply_all_btn.clicked.connect(self._on_apply_all)

        # save masks button (visible for AdaptiveThresholdOp)
        self.save_masks_btn = QPushButton("Save Masks...")
        self.save_masks_btn.setEnabled(False)
        outer.addWidget(self.save_masks_btn)
        self.save_masks_btn.clicked.connect(main_app.save_threshold_masks)

        # pipeline serialization
        io_row = QHBoxLayout()
        save_pl_btn = QPushButton("Save Pipeline…")
        load_pl_btn = QPushButton("Load Pipeline…")
        io_row.addWidget(save_pl_btn)
        io_row.addWidget(load_pl_btn)
        outer.addLayout(io_row)
        save_pl_btn.clicked.connect(main_app.save_pipeline_dialog)
        load_pl_btn.clicked.connect(main_app.load_pipeline_dialog)

    # ------------------------------------------------------------------

    def _is_stale(self, idx, op) -> bool:
        pipeline = self.main_app.pipeline
        if isinstance(op, BgSubtractOp):
            return op.enabled and op._background is None
        if isinstance(op, AdaptiveThresholdOp):
            cache = pipeline._caches[idx] if 0 <= idx < len(pipeline._caches) else {}
            return op._batch_was_run and not bool(cache)
        return False

    def rebuild_list(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for i, op in enumerate(self.main_app.pipeline.operations):
            item = QListWidgetItem(op.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if op.enabled else Qt.CheckState.Unchecked)
            if self._is_stale(i, op):
                item.setForeground(QColor(255, 150, 0))
                item.setToolTip("Stale — re-run Fit or Apply to All")
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

        if 0 <= self._selected_idx < len(self.main_app.pipeline.operations):
            self.list_widget.setCurrentRow(self._selected_idx)
        else:
            self._selected_idx = -1
            self._show_params(None)

        self._update_action_buttons()
        self.main_app._sync_filter_menu_actions()

    def update_stale_colors(self):
        """Update stale indicators on existing list items without rebuilding.
        Safe to call from inside a ParamsWidget callback (no widget destruction)."""
        ops = self.main_app.pipeline.operations
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item is None or i >= len(ops):
                continue
            op = ops[i]
            if self._is_stale(i, op):
                item.setForeground(QColor(255, 150, 0))
                item.setToolTip("Stale — re-run Fit or Apply to All")
            else:
                item.setForeground(QColor())
                item.setToolTip("")

    def _on_row_selected(self, row):
        self._selected_idx = row
        ops = self.main_app.pipeline.operations
        op = ops[row] if 0 <= row < len(ops) else None
        self._show_params(op)
        self._update_action_buttons()
        self.remove_btn.setEnabled(op is not None)
        self.main_app._update_block_grid()

    def _on_item_changed(self, item):
        row = self.list_widget.row(item)
        ops = self.main_app.pipeline.operations
        if not (0 <= row < len(ops)):
            return
        op = ops[row]
        enabled = item.checkState() == Qt.CheckState.Checked
        if op.enabled != enabled:
            op.enabled = enabled
            self.main_app.pipeline._invalidate_from(row)
            # toggling an op upstream of BgSubtractOp makes stored background stale
            bg_idx, bg_op = self.main_app._find_op(BgSubtractOp)
            if bg_op is not None and row < bg_idx:
                bg_op.set_background(None)
                self.rebuild_list()
            self.main_app.update_frame_display()
            # toggling CropOp changes displayed image size — autoRange so user sees the change
            if isinstance(op, CropOp):
                self.main_app.view_box.autoRange()
            self._show_validation_warnings()

    def _on_rows_moved(self, parent, start, end, dest_parent, dest_row):
        # QListWidget handles visual reorder; sync to Pipeline
        ops = self.main_app.pipeline.operations
        new_order = []
        for i in range(self.list_widget.count()):
            name = self.list_widget.item(i).text()
            # find first op with that name not yet claimed
            for op in ops:
                if op.name == name and op not in new_order:
                    new_order.append(op)
                    break
        self.main_app.pipeline.operations = new_order
        self.main_app.pipeline._caches = [
            self.main_app.pipeline._caches[ops.index(op)] if op in ops else {}
            for op in new_order
        ]
        self.main_app.pipeline._invalidate_from(0)
        self.main_app._clear_bg_unconditionally()
        self.main_app._invalidate_threshold_caches()
        self.main_app.update_frame_display()
        self.rebuild_list()
        self._show_validation_warnings()

    def _show_validation_warnings(self):
        issues = self.main_app.pipeline.validate_chain()
        errors = [msg for _, sev, msg in issues if sev == "error"]
        warnings = [msg for _, sev, msg in issues if sev == "warning"]
        if errors:
            self.main_app.status_label.setText("Pipeline error: " + "; ".join(errors))
        elif warnings:
            self.main_app.status_label.setText("Pipeline warning: " + "; ".join(warnings))
        else:
            current = self.main_app.status_label.text()
            if current.startswith("Pipeline error:") or current.startswith("Pipeline warning:"):
                self.main_app.update_frame_display()

    def _show_params(self, op):
        if op is None:
            lbl = QLabel("Select a stage to edit its params.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.params_scroll.setWidget(lbl)
            return
        if isinstance(op, CropOp):
            pw = CropParamsWidget(self.main_app, op)
        elif isinstance(op, IntensityWatershedSplitOp):
            pw = IntensityWatershedParamsWidget(self.main_app, op, self._on_param_changed)
        else:
            pw = ParamsWidget(op, self._on_param_changed)
        self.params_scroll.setWidget(pw)

    def _on_param_changed(self):
        idx = self._selected_idx
        self.main_app.pipeline._invalidate_from(idx)
        self.main_app._clear_bg_if_upstream_changed(idx)
        self.main_app.update_frame_display()
        self.main_app._update_block_grid()

    def _update_action_buttons(self):
        idx = self._selected_idx
        ops = self.main_app.pipeline.operations
        op = ops[idx] if 0 <= idx < len(ops) else None
        has_fit = op is not None and type(op).fit_with_progress is not Operation.fit_with_progress
        self.fit_btn.setEnabled(has_fit and self.main_app.sequence_manager is not None)
        self.apply_all_btn.setEnabled(
            op is not None and op.supports_batch_cache and
            self.main_app.sequence_manager is not None
        )
        # Save Masks enabled when selected op is AdaptiveThresholdOp or any
        # downstream is_binary_mask_op, AND AdaptiveThresholdOp has a batch cache.
        # save_threshold_masks always finds AdaptiveThresholdOp via _find_op and
        # walks the full binary chain, so all binary-chain rows should offer save.
        self.save_masks_btn.setEnabled(self._thresh_cache_ready_for_selected())

    def _on_add(self):
        # Simple palette dialog
        dlg = QDialog(self.main_app)
        dlg.setWindowTitle("Add Stage")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Choose operation:"))
        combo = QComboBox()
        combo.addItems(list(OPERATION_REGISTRY.keys()))
        layout.addWidget(combo)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cls = OPERATION_REGISTRY[combo.currentText()]
        op = cls()
        insert_at = self._selected_idx + 1 if self._selected_idx >= 0 else -1
        self.main_app.pipeline.add_operation(op, at_idx=insert_at)
        new_idx = self.main_app.pipeline.operations.index(op)
        self.main_app._clear_bg_if_upstream_changed(new_idx)
        self.main_app._invalidate_threshold_caches()
        self.rebuild_list()
        self.main_app.update_frame_display()

    def _on_remove(self):
        idx = self._selected_idx
        if idx < 0:
            return
        bg_idx, _ = self.main_app._find_op(BgSubtractOp)
        self.main_app.pipeline.remove_operation(idx)
        if bg_idx >= 0 and idx < bg_idx:
            _, bg_op = self.main_app._find_op(BgSubtractOp)
            if bg_op is not None:
                bg_op.set_background(None)
        self.main_app._invalidate_threshold_caches()
        self._selected_idx = max(0, idx - 1) if self.main_app.pipeline.operations else -1
        self.rebuild_list()
        self.main_app.update_frame_display()

    def _on_fit(self):
        idx = self._selected_idx
        ops = self.main_app.pipeline.operations
        if not (0 <= idx < len(ops)):
            return
        op = ops[idx]
        if not isinstance(op, BgSubtractOp):
            return
        self.main_app._run_bg_fit(op, idx)

    def _on_apply_all(self):
        idx = self._selected_idx
        ops = self.main_app.pipeline.operations
        if not (0 <= idx < len(ops)):
            return
        op = ops[idx]
        if not isinstance(op, AdaptiveThresholdOp):
            return
        self.main_app._run_threshold_batch(op, idx)

    def _thresh_cache_ready_for_selected(self) -> bool:
        """Return True when Save Masks should be enabled for the current selection.

        Conditions:
          1. Selected op is AdaptiveThresholdOp OR a downstream is_binary_mask_op.
          2. AdaptiveThresholdOp exists in the pipeline with a non-empty batch cache.
        """
        idx = self._selected_idx
        ops = self.main_app.pipeline.operations
        op = ops[idx] if 0 <= idx < len(ops) else None
        if op is None:
            return False
        # Condition 1: selected op is threshold or a downstream binary-mask op
        is_threshold = isinstance(op, AdaptiveThresholdOp)
        is_binary_downstream = (
            getattr(op, "is_binary_mask_op", False) and not is_threshold
        )
        if not (is_threshold or is_binary_downstream):
            return False
        # Condition 2: AdaptiveThresholdOp has a non-empty batch cache
        thresh_idx, thresh_op = self.main_app._find_op(AdaptiveThresholdOp)
        if thresh_op is None:
            return False
        return bool(
            self.main_app.pipeline._caches[thresh_idx]
            if 0 <= thresh_idx < len(self.main_app.pipeline._caches) else {}
        )

    def update_save_masks_btn(self):
        self.save_masks_btn.setEnabled(self._thresh_cache_ready_for_selected())


# ---------------------------------------------------------------------------
# Facet thickness measurement
# ---------------------------------------------------------------------------

FACET_COLORS = [
    (255, 196,   0),   # amber
    (  0, 200, 255),   # cyan
    (255, 110, 180),   # pink
    (120, 255, 120),   # green
    (200, 140, 255),   # violet
    (255, 140,  60),   # orange
]


class FacetGraphics:
    """The pyqtgraph items that draw one FacetSession on the image.

    Held persistently per session and refreshed with setData rather than
    rebuilt, so dragging a point stays cheap. All coordinates are image /
    ViewBox coordinates, so the overlay stays anchored through zoom and pan.
    """

    def __init__(self, view_box, color):
        self.view_box = view_box
        self.color = color
        pen = pg.mkPen(color=color + (220,), width=1.6)

        # Fitted interface line, extended across the visible image.
        self.line_item = pg.PlotDataItem(pen=pen)
        self.line_item.setZValue(10)

        # Perpendicular drops: deliberately light so they do not obscure the
        # interface being measured.
        self.perp_item = pg.PlotDataItem(
            pen=pg.mkPen(color=color + (110,), width=1.0), connect='pairs'
        )
        self.perp_item.setZValue(10)

        # Interface points: filled circles. Surface points: hollow squares.
        self.interface_scatter = pg.ScatterPlotItem(
            symbol='o', size=9, brush=pg.mkBrush(color + (230,)),
            pen=pg.mkPen(color=(20, 20, 20, 200), width=1),
        )
        self.interface_scatter.setZValue(11)
        self.surface_scatter = pg.ScatterPlotItem(
            symbol='s', size=10, brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen(color=color + (255,), width=2),
        )
        self.surface_scatter.setZValue(11)

        self.labels: list[pg.TextItem] = []

        for item in (self.line_item, self.perp_item,
                     self.interface_scatter, self.surface_scatter):
            view_box.addItem(item)

    # -- helpers ---------------------------------------------------------

    def _label_pool(self, n):
        """Grow/shrink the TextItem pool to exactly n visible labels."""
        while len(self.labels) < n:
            t = pg.TextItem(color=self.color, anchor=(0.0, 0.5),
                            fill=(20, 20, 20, 150))
            t.setZValue(12)
            self.view_box.addItem(t)
            self.labels.append(t)
        for i, t in enumerate(self.labels):
            t.setVisible(i < n)

    def set_visible(self, visible):
        for item in (self.line_item, self.perp_item,
                     self.interface_scatter, self.surface_scatter):
            item.setVisible(visible)
        if not visible:
            for t in self.labels:
                t.setVisible(False)

    def update(self, session, image_hw, show_labels=True):
        """Redraw everything for this session."""
        interface = np.asarray(session.interface_points, dtype=np.float64).reshape(-1, 2)
        surface = np.asarray(session.surface_points, dtype=np.float64).reshape(-1, 2)

        self.interface_scatter.setData(x=interface[:, 0], y=interface[:, 1])
        self.surface_scatter.setData(x=surface[:, 0], y=surface[:, 1])

        fit = session.fit()
        if fit is None or image_hw is None:
            self.line_item.setData(x=[], y=[])
            self.perp_item.setData(x=[], y=[])
            self._label_pool(0)
            return

        origin, direction, _ = fit
        h, w = image_hw
        p0, p1 = line_segment_across_box(origin, direction, 0.0, float(w), 0.0, float(h))
        self.line_item.setData(x=[p0[0], p1[0]], y=[p0[1], p1[1]])

        measurements = session.measurements()
        if not measurements:
            self.perp_item.setData(x=[], y=[])
            self._label_pool(0)
            return

        xs, ys = [], []
        for m in measurements:
            xs += [m.outer_x, m.foot_x]
            ys += [m.outer_y, m.foot_y]
        self.perp_item.setData(x=np.array(xs), y=np.array(ys))

        if not show_labels:
            self._label_pool(0)
            return

        unit = session.unit
        self._label_pool(len(measurements))
        for t, m in zip(self.labels, measurements):
            value = session.to_units(m.thickness_px)
            t.setText(f"{value:.3g} {unit}")
            t.setColor(pg.mkColor((255, 90, 90)) if m.sign_anomalous
                       else pg.mkColor(self.color))
            t.setPos(m.outer_x + 4, m.outer_y)

    def remove(self):
        for item in (self.line_item, self.perp_item,
                     self.interface_scatter, self.surface_scatter):
            self.view_box.removeItem(item)
        for t in self.labels:
            self.view_box.removeItem(t)
        self.labels.clear()


class FacetThicknessPanel(QWidget):
    """Floating tool window driving the two-phase facet thickness workflow.

    Phase A: click points along the film/substrate interface; a total-least-
    squares line is fitted through them live. Phase B: click points on the
    outer film surface; each gets a perpendicular dropped onto the fitted
    interface line and labelled with the thickness.

    Purely additive: nothing here touches the processing pipeline. Points are
    placed on the displayed geometry, in the same coordinate space as the
    existing measurement rays.
    """

    def __init__(self, main_app):
        super().__init__(main_app, Qt.WindowType.Tool)
        self.main_app = main_app
        self.setWindowTitle("Facet Thickness")
        self.setGeometry(1150, 640, 420, 640)
        self._updating = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # --- phase / mode banner ---
        self.phase_label = QLabel("No session. Click “New Session” to begin.")
        self.phase_label.setWordWrap(True)
        self.phase_label.setStyleSheet("font-weight: bold;")
        outer.addWidget(self.phase_label)

        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #ffb020; font-size: 11px;")
        self.hint_label.setVisible(False)
        outer.addWidget(self.hint_label)

        # --- session list ---
        outer.addWidget(QLabel("Sessions (uncheck to hide):"))
        self.session_list = QListWidget()
        self.session_list.setMaximumHeight(110)
        self.session_list.currentRowChanged.connect(self._on_session_selected)
        self.session_list.itemChanged.connect(self._on_session_item_changed)
        outer.addWidget(self.session_list)

        row1 = QHBoxLayout()
        self.new_btn = QPushButton("New Session")
        self.new_btn.clicked.connect(self._on_new_session)
        row1.addWidget(self.new_btn)
        self.finish_btn = QPushButton("Finish Interface")
        self.finish_btn.clicked.connect(self._on_finish_interface)
        row1.addWidget(self.finish_btn)
        outer.addLayout(row1)

        row2 = QHBoxLayout()
        self.undo_btn = QPushButton("Undo Last Point")
        self.undo_btn.clicked.connect(self._on_undo)
        row2.addWidget(self.undo_btn)
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._on_delete_selected)
        row2.addWidget(self.delete_btn)
        self.clear_btn = QPushButton("Clear Session")
        self.clear_btn.clicked.connect(self._on_clear_session)
        row2.addWidget(self.clear_btn)
        outer.addLayout(row2)

        # --- summary ---
        self.summary_label = QLabel("—")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        outer.addWidget(self.summary_label)

        self.angle_hint_label = QLabel("")
        self.angle_hint_label.setWordWrap(True)
        self.angle_hint_label.setStyleSheet("font-size: 11px; color: #888;")
        outer.addWidget(self.angle_hint_label)

        # --- measurement table ---
        from PyQt6.QtWidgets import QTableWidget, QAbstractItemView
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "foot x", "thickness", "flag"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(170)
        self.table.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(self.table)

        # --- thickness vs x plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Foot x (px)')
        self.plot_widget.setLabel('left', 'Thickness (px)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(-10, 10))
        self.plot_widget.setMinimumHeight(150)
        outer.addWidget(self.plot_widget, stretch=1)
        self._plot_items: list[pg.ScatterPlotItem] = []

        # --- persistence ---
        row3 = QHBoxLayout()
        save_btn = QPushButton("Save JSON…")
        save_btn.clicked.connect(main_app.save_facet_sessions_dialog)
        row3.addWidget(save_btn)
        load_btn = QPushButton("Load JSON…")
        load_btn.clicked.connect(main_app.load_facet_sessions_dialog)
        row3.addWidget(load_btn)
        csv_btn = QPushButton("Export CSV…")
        csv_btn.clicked.connect(main_app.export_facet_csv_dialog)
        row3.addWidget(csv_btn)
        outer.addLayout(row3)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def sessions(self):
        return self.main_app.facet_sessions

    def active_session(self):
        return self.main_app.active_facet_session()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_new_session(self):
        from PyQt6.QtWidgets import QInputDialog
        if self.main_app.sequence_manager is None:
            QMessageBox.warning(self, "No Sequence", "Open a TIFF sequence first.")
            return
        default = f"facet{len(self.sessions) + 1}"
        label, ok = QInputDialog.getText(self, "New Facet Session",
                                         "Session label (e.g. upleg, downleg):",
                                         text=default)
        if not ok:
            return
        label = label.strip() or default
        self.main_app.new_facet_session(label)

    def _on_finish_interface(self):
        session = self.active_session()
        if session is None:
            return
        if not session.finish_interface():
            QMessageBox.information(
                self, "Not Enough Points",
                "Click at least 2 interface points before finishing the interface."
            )
            return
        self.main_app.facet_mode = 'surface'
        self.main_app.refresh_facets()

    def _on_undo(self):
        session = self.active_session()
        if session is None:
            return
        session.undo_last()
        self.main_app.refresh_facets()

    def _on_delete_selected(self):
        """Delete the surface points backing the selected table rows."""
        session = self.active_session()
        if session is None:
            return
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()},
                      reverse=True)
        if not rows:
            self.main_app.status_label.setText(
                "Select one or more rows in the facet table to delete."
            )
            return
        for row in rows:
            session.remove_point("surface", row)
        self.main_app.refresh_facets()

    def _on_clear_session(self):
        session = self.active_session()
        if session is None:
            return
        reply = QMessageBox.question(
            self, "Clear Session",
            f"Delete session “{session.label}” and all of its points?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.main_app.delete_facet_session(session)

    def _on_session_selected(self, row):
        if self._updating:
            return
        self.main_app.set_active_facet_session(row)

    def _on_session_item_changed(self, item):
        if self._updating:
            return
        row = self.session_list.row(item)
        if 0 <= row < len(self.sessions):
            self.sessions[row].visible = (item.checkState() == Qt.CheckState.Checked)
            self.main_app.refresh_facets()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self):
        self._updating = True
        try:
            self._refresh_session_list()
            self._refresh_banner()
            self._refresh_summary()
            self._refresh_table()
            self._refresh_plot()
        finally:
            self._updating = False

    def _refresh_session_list(self):
        active_idx = self.main_app.active_facet_idx
        if self.session_list.count() != len(self.sessions):
            self.session_list.clear()
            for s in self.sessions:
                item = QListWidgetItem()
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                self.session_list.addItem(item)
        for i, s in enumerate(self.sessions):
            item = self.session_list.item(i)
            frame_note = "" if s.frame_idx == self.main_app.current_frame else \
                f"  [frame {s.frame_idx + 1}]"
            item.setText(f"{s.label}   ({len(s.interface_points)} iface, "
                         f"{len(s.surface_points)} surf){frame_note}")
            item.setCheckState(Qt.CheckState.Checked if s.visible else Qt.CheckState.Unchecked)
            item.setForeground(QColor(*FACET_COLORS[s.color_idx % len(FACET_COLORS)]))
        if 0 <= active_idx < self.session_list.count():
            self.session_list.setCurrentRow(active_idx)

    def _refresh_banner(self):
        session = self.active_session()
        mode = self.main_app.facet_mode
        if session is None:
            self.phase_label.setText("No session. Click “New Session” to begin.")
        elif session.phase == "interface":
            state = "clicking" if mode == 'interface' else "paused"
            self.phase_label.setText(
                f"“{session.label}” — Phase A ({state}): click points along the "
                f"film/substrate interface, then “Finish Interface”."
            )
        else:
            state = "clicking" if mode == 'surface' else "paused"
            self.phase_label.setText(
                f"“{session.label}” — Phase B ({state}): click points on the outer "
                f"film surface. Each is dropped perpendicular to the interface."
            )

        hints = []
        if self.main_app.pipeline.scale is None:
            hints.append("No pixel scale set — all values shown in pixels. "
                         "Calibrate via Tools → Set Scale… (Ctrl+M).")
        if session is not None and session.geometry_tag and \
                session.geometry_tag != self.main_app._facet_geometry_tag():
            hints.append("Rotation/crop changed since this session was measured — "
                         "the points no longer line up with the displayed image. "
                         "Recorded thicknesses are still valid.")
        self.hint_label.setText("  ".join(hints))
        self.hint_label.setVisible(bool(hints))

    def _refresh_summary(self):
        session = self.active_session()
        if session is None:
            self.summary_label.setText("—")
            self.angle_hint_label.setText("")
            return
        s = session.summary()
        unit = s["unit"]
        if s["angle_deg"] is None:
            self.summary_label.setText(
                f"Interface: {s['n_interface']} point(s) — need ≥ 2 for a fit."
            )
            self.angle_hint_label.setText("")
            return

        text = (
            f"Facet angle: {s['angle_deg']:.2f}°    "
            f"Fit RMS: {s['rms_residual']:.4g} {unit} "
            f"({s['rms_residual_px']:.3g} px, n={s['n_interface']})\n"
        )
        if s["n"]:
            text += (
                f"Thickness ({unit}): n={s['n']}  mean={s['mean']:.4g}  "
                f"median={s['median']:.4g}  std={s['std']:.4g}\n"
                f"                 min={s['min']:.4g}  max={s['max']:.4g}"
            )
        else:
            text += "Thickness: no surface points yet."
        self.summary_label.setText(text)

        colour = "#5ad65a" if s["hint_status"] == "match" else "#888"
        self.angle_hint_label.setStyleSheet(f"font-size: 11px; color: {colour};")
        self.angle_hint_label.setText(s["hint"] or "")

    def _refresh_table(self):
        from PyQt6.QtWidgets import QTableWidgetItem
        session = self.active_session()
        measurements = session.measurements() if session is not None else []
        unit = session.unit if session is not None else "px"
        self.table.setHorizontalHeaderLabels(
            ["#", f"foot x ({unit})", f"thickness ({unit})", "flag"]
        )
        self.table.setRowCount(len(measurements))
        for i, m in enumerate(measurements):
            values = [
                str(i + 1),
                f"{session.to_units(m.foot_x):.4g}",
                f"{session.to_units(m.thickness_px):.4g}",
                "wrong side?" if m.sign_anomalous else "",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if m.sign_anomalous:
                    item.setForeground(QColor(255, 90, 90))
                self.table.setItem(i, col, item)

    def _refresh_plot(self):
        for item in self._plot_items:
            self.plot_widget.removeItem(item)
        self._plot_items.clear()
        legend = self.plot_widget.plotItem.legend
        if legend is not None:
            legend.clear()

        unit = "px"
        symbols = ['o', 's', 't', 'd', '+', 'x']
        for i, session in enumerate(self.sessions):
            measurements = session.measurements()
            if not measurements or not session.visible:
                continue
            if session.unit_per_px is not None:
                unit = session.unit
            xs = np.array([session.to_units(m.foot_x) for m in measurements])
            ys = np.array([session.to_units(m.thickness_px) for m in measurements])
            colour = FACET_COLORS[session.color_idx % len(FACET_COLORS)]
            item = pg.ScatterPlotItem(
                x=xs, y=ys, symbol=symbols[session.color_idx % len(symbols)],
                size=8, brush=pg.mkBrush(colour + (200,)),
                pen=pg.mkPen(colour + (255,)),
            )
            self.plot_widget.addItem(item)
            if legend is not None:
                legend.addItem(item, session.label)
            self._plot_items.append(item)

        self.plot_widget.setLabel('bottom', f'Foot x ({unit})')
        self.plot_widget.setLabel('left', f'Thickness ({unit})')

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class TiffViewerApp(QMainWindow):
    _OF_CACHE_MAX = 20      # max partner-frame display entries in optical flow display cache
    _OF_FLOW_CACHE_MAX = 50  # max (a_idx, b_idx) Farneback result entries

    def __init__(self):
        super().__init__()
        self.setWindowTitle("tiffscope")
        self.setGeometry(100, 100, 1000, 800)

        # --- core state ---
        self.sequence_manager = None
        self.current_frame = 0
        self.num_frames = 0
        self.mouse_pos = None
        self.current_raw_frame = None

        self.optical_flow_enabled = False
        self.optical_flow_mode = 'time_resolved'  # 'time_resolved' | 'pairwise'
        self._of_display_cache: dict[int, np.ndarray] = {}
        self._of_cache_tag: str = ""
        self._of_flow_cache: dict[tuple, object] = {}  # (a_idx,b_idx) → pos ndarray | None
        self._of_flow_tag: str = ""

        # crop draw state (UI only — actual crop lives in CropOp inside pipeline)
        self.crop_mode = False
        self.crop_roi_item = None
        self._drawing_crop = False
        self._crop_draw_start = None
        self._crop_region_snapshot = None

        self.bit_depth_max = 4095
        self.vmin = 0
        self.vmax = 4095
        self.gamma = 1.0
        self._display_lut: np.ndarray | None = None
        self._display_lut_key: tuple | None = None

        self.rays = []
        self.ray_mode = None

        # Facet thickness measurement state. facet_mode is None when idle,
        # 'interface' while collecting phase-A points, 'surface' for phase B.
        # Points live in display (ViewBox) coordinates, the same space as rays.
        self.facet_sessions: list[FacetSession] = []
        self.active_facet_idx = -1
        self.facet_mode = None
        self._facet_graphics: dict[int, FacetGraphics] = {}  # id(session) → items
        self._facet_drag = None  # (session, kind, index) while dragging a point

        # Measurement / pixel-scale state. measure_mode is False when idle,
        # 'await_first' before the first click lands an anchor, then
        # 'await_second' while rubber-banding to the second click.
        self.measure_mode = False
        self._measure_first_pt = None
        self._measure_line_item = None
        self._measure_label_item = None
        self._measure_pending_callback = None

        self._last_frame_for_histogram = None
        self._bg_worker = None
        self._threshold_worker = None
        self._display_hw = None

        # --- pipeline ---
        self.pipeline = Pipeline()

        # display selection: index into pipeline.operations, or -1 = last
        self.display_base_stage = -1    # which stage to display
        self.display_overlay_stage = -1  # -1 = no overlay
        self.overlay_alpha = 160

        # overlay frame for threshold (kept separately so the overlay ImageItem
        # can be updated independently of the main image)
        self._overlay_op_idx = -1  # pipeline idx of the op driving the overlay

        # --- UI ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        _mod = "Cmd" if sys.platform == "darwin" else "Ctrl"
        self.status_label = QLabel(f"Please open a specific TIFF file from the File menu ({_mod}+O).")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        self.pixel_info_label = QLabel("")
        self.pixel_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixel_info_label.setMinimumHeight(20)
        layout.addWidget(self.pixel_info_label)

        self.view_box = self.glw.addViewBox()
        self.view_box.setAspectLocked(True)
        self.view_box.invertY(True)
        self.view_box.setMouseEnabled(x=False, y=False)
        self.view_box.setMenuEnabled(False)

        self.image_item = pg.ImageItem()
        self.view_box.addItem(self.image_item)

        self.vector_field = pg.PlotCurveItem(
            pen=pg.mkPen(color=(0, 255, 150, 200), width=1.5), connect='pairs'
        )
        self.view_box.addItem(self.vector_field)

        self.threshold_overlay = pg.ImageItem()
        self.threshold_overlay.setZValue(5)
        self.view_box.addItem(self.threshold_overlay)

        self.block_grid_overlay = pg.PlotCurveItem(
            pen=pg.mkPen(color=(0, 200, 255, 90), width=0.5), connect='pairs'
        )
        self.block_grid_overlay.setZValue(7)
        self.block_grid_overlay.setVisible(False)
        self.view_box.addItem(self.block_grid_overlay)

        self.glw.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.glw.viewport().installEventFilter(self)

        QShortcut(QKeySequence("Ctrl++"), self).activated.connect(lambda: self.zoom_image(0.8))
        QShortcut(QKeySequence("Ctrl+="), self).activated.connect(lambda: self.zoom_image(0.8))
        QShortcut(QKeySequence("Ctrl+-"), self).activated.connect(lambda: self.zoom_image(1.25))

        # --- tool windows ---
        self.tool_window = ContrastToolWindow(self)
        self.pipeline_panel = PipelinePanel(self)
        self.perf = PerfTracker()
        self.perf_window = PerformanceToolWindow(self)
        self.blob_size_window = BlobSizeAnalysisWindow(self)
        self.region_props_window = RegionPropsWindow(self)
        self.mie_window = MieAnalysisWindow(self)
        self.facet_panel = FacetThicknessPanel(self)

        self.create_menus()

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def _find_op(self, cls):
        """Return (idx, op) for the first op of cls in the pipeline, regardless
        of enabled state. Returns (-1, None) if none exists. Callers that need
        only enabled ops must check op.enabled themselves."""
        for i, op in enumerate(self.pipeline.operations):
            if isinstance(op, cls):
                return i, op
        return -1, None

    def _select_op_in_panel(self, cls):
        idx, _ = self._find_op(cls)
        if idx < 0:
            return
        self.pipeline_panel.show()
        self.pipeline_panel.raise_()
        self.pipeline_panel._selected_idx = idx
        self.pipeline_panel.list_widget.setCurrentRow(idx)

    def _ensure_op(self, cls, at_idx=-1):
        """Return existing op of cls or insert a new one."""
        idx, op = self._find_op(cls)
        if op is not None:
            return idx, op
        op = cls()
        idx = self.pipeline.add_operation(op, at_idx=at_idx)
        self.pipeline_panel.rebuild_list()
        return idx, op

    def _crop_region(self):
        """Return (x,y,w,h) from active CropOp or None."""
        _, op = self._find_op(CropOp)
        if op is None or not op.enabled:
            return None
        return (op.params["x"], op.params["y"], op.params["w"], op.params["h"])

    def _rotation_state(self):
        """Return current rotation k from RotateOp or 0."""
        _, op = self._find_op(RotateOp)
        if op is None:
            return 0
        return op.params["k"]

    def _clear_bg_if_upstream_changed(self, changed_idx: int) -> None:
        """Clear stored background when an op at or before BgSubtractOp changes."""
        bg_idx, bg_op = self._find_op(BgSubtractOp)
        if bg_op is not None and changed_idx <= bg_idx:
            bg_op.set_background(None)
            self.pipeline_panel.update_stale_colors()

    def _clear_bg_unconditionally(self) -> None:
        """Clear stored background after structural pipeline changes (reorder/add/remove)
        where upstream invariants may have shifted in ways that are hard to check."""
        _, bg_op = self._find_op(BgSubtractOp)
        if bg_op is not None:
            bg_op.set_background(None)

    def _display_target_stage(self) -> int:
        """target_stage_idx for the main display image.
        Stops before AdaptiveThresholdOp (shown as RGBA overlay, not base image)
        and before CropOp when in crop_mode (full image shown while drawing ROI)."""
        n_ops = len(self.pipeline.operations)
        stop_at = n_ops if self.display_base_stage == -1 else self.display_base_stage + 1

        thresh_idx, thresh_op = self._find_op(AdaptiveThresholdOp)
        if thresh_op is not None and thresh_op.enabled:
            stop_at = min(stop_at, thresh_idx)

        if self.crop_mode:
            crop_idx, _ = self._find_op(CropOp)
            if crop_idx >= 0:
                stop_at = min(stop_at, crop_idx)

        if stop_at == 0:
            return -2
        if stop_at >= n_ops:
            return -1
        return stop_at - 1

    # ------------------------------------------------------------------
    # Background fit runner
    # ------------------------------------------------------------------

    def _run_bg_fit(self, op: BgSubtractOp, op_idx: int):
        if self.sequence_manager is None:
            return

        sample_n = op.params.get("sample_n", 50)

        progress = QProgressDialog("Loading frames...", None, 0, sample_n, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Background Computation")
        progress.setMinimumDuration(0)
        progress.show()

        loop = QEventLoop()
        error_holder = [None]

        self._bg_worker = BgComputeWorker(op, op_idx, self.pipeline, self.sequence_manager)

        def on_progress(c, t):
            if c > sample_n and progress.maximum() <= sample_n:
                progress.setLabelText("Computing median...")
                progress.setMaximum(t)
            progress.setValue(c)

        def on_finished(result):
            loop.quit()

        def on_error(msg):
            error_holder[0] = msg
            loop.quit()

        self._bg_worker.progress.connect(on_progress)
        self._bg_worker.finished.connect(on_finished)
        self._bg_worker.error.connect(on_error)
        self._bg_worker.start()
        loop.exec()
        self._bg_worker.wait()
        progress.close()

        if error_holder[0]:
            QMessageBox.critical(self, "Background Computation Failed", error_holder[0])
            return

        self.pipeline._invalidate_from(op_idx)
        self._invalidate_threshold_caches()
        self.pipeline_panel.rebuild_list()
        self.update_frame_display()
        self.status_label.setText("Background computed. Enable BgSubtractOp via its checkbox.")

    # ------------------------------------------------------------------
    # Threshold batch runner
    # ------------------------------------------------------------------

    def _run_threshold_batch(self, op: AdaptiveThresholdOp, op_idx: int):
        if self.sequence_manager is None:
            return

        file_paths = [
            os.path.join(self.sequence_manager.folder_path, f)
            for f in self.sequence_manager.files
        ]

        progress = QProgressDialog("Computing threshold masks...", None, 0, self.num_frames, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Adaptive Threshold")
        progress.setMinimumDuration(0)
        progress.show()

        loop = QEventLoop()
        result_holder = [None]
        error_holder = [None]

        self._threshold_worker = ThresholdBatchWorker(
            op, self.pipeline, op_idx, file_paths
        )

        def on_progress(c, t):
            progress.setValue(c)

        def on_finished(masks):
            result_holder[0] = masks
            loop.quit()

        def on_error(msg):
            error_holder[0] = msg
            loop.quit()

        self._threshold_worker.progress.connect(on_progress)
        self._threshold_worker.finished.connect(on_finished)
        self._threshold_worker.error.connect(on_error)
        self._threshold_worker.start()
        loop.exec()
        self._threshold_worker.wait()
        progress.close()

        if error_holder[0]:
            QMessageBox.critical(self, "Threshold Failed", error_holder[0])
            return

        masks = result_holder[0]
        for frame_idx, mask in masks.items():
            self.pipeline.store_batch_cache(op_idx, frame_idx, mask)
        op._batch_was_run = True

        self.pipeline_panel.update_save_masks_btn()
        self.pipeline_panel.rebuild_list()
        self.update_frame_display()
        self.status_label.setText(f"Threshold applied to {self.num_frames} frames.")

    def _invalidate_threshold_caches(self):
        for i, op in enumerate(self.pipeline.operations):
            if isinstance(op, AdaptiveThresholdOp):
                self.pipeline.clear_batch_cache(i)
        self.pipeline_panel.update_save_masks_btn()
        self.pipeline_panel.update_stale_colors()

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def create_menus(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open Sequence...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        save_pl_action = QAction("Save Pipeline...", self)
        save_pl_action.triggered.connect(self.save_pipeline_dialog)
        file_menu.addAction(save_pl_action)
        load_pl_action = QAction("Load Pipeline...", self)
        load_pl_action.triggered.connect(self.load_pipeline_dialog)
        file_menu.addAction(load_pl_action)

        view_menu = menu_bar.addMenu("View")

        self.flow_action = QAction("Live Optical Flow", self, checkable=True)
        self.flow_action.setShortcut("Ctrl+F")
        self.flow_action.triggered.connect(self.toggle_optical_flow)
        view_menu.addAction(self.flow_action)

        flow_settings_action = QAction("Optical Flow Settings…", self)
        flow_settings_action.triggered.connect(self.show_optical_flow_settings)
        view_menu.addAction(flow_settings_action)

        view_menu.addSeparator()

        tools_action = QAction("Contrast & Histogram Tools", self)
        tools_action.setShortcut("Ctrl+T")
        tools_action.triggered.connect(self.toggle_tool_window)
        view_menu.addAction(tools_action)

        perf_action = QAction("Performance Monitor", self)
        perf_action.setShortcut("Ctrl+Shift+M")
        perf_action.triggered.connect(self.toggle_perf_window)
        view_menu.addAction(perf_action)

        pipeline_action = QAction("Pipeline Panel", self)
        pipeline_action.setShortcut("Ctrl+P")
        pipeline_action.triggered.connect(self._toggle_pipeline_panel)
        view_menu.addAction(pipeline_action)

        view_menu.addSeparator()

        reset_action = QAction("Reset View", self)
        reset_action.setShortcut("Ctrl+0")
        reset_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_action)

        rotate_action = QAction("Rotate 90° CW", self)
        rotate_action.setShortcut("Ctrl+R")
        rotate_action.triggered.connect(self.rotate_image)
        view_menu.addAction(rotate_action)

        view_menu.addSeparator()

        self.bit_group = QActionGroup(self)
        self.bit_group.setExclusive(True)
        bit_depths = {
            "8-bit (Max 255)": 255, "12-bit (Max 4095)": 4095,
            "14-bit (Max 16383)": 16383, "16-bit (Max 65535)": 65535,
        }
        for name, max_val in bit_depths.items():
            action = QAction(name, self, checkable=True)
            if max_val == 4095:
                action.setChecked(True)
            action.triggered.connect(lambda checked, v=max_val: self.set_bit_depth(v))
            self.bit_group.addAction(action)
            view_menu.addAction(action)

        view_menu.addSeparator()

        self.crop_action = QAction("Crop Mode", self, checkable=True)
        self.crop_action.setShortcut("Ctrl+Shift+C")
        self.crop_action.triggered.connect(self.toggle_crop_mode)
        view_menu.addAction(self.crop_action)

        clear_crop_action = QAction("Clear Crop", self)
        clear_crop_action.setShortcut("Ctrl+Shift+X")
        clear_crop_action.triggered.connect(self.clear_crop)
        view_menu.addAction(clear_crop_action)

        apply_crop_action = QAction("Apply Crop to Files...", self)
        apply_crop_action.triggered.connect(self.apply_crop_dialog)
        view_menu.addAction(apply_crop_action)

        view_menu.addSeparator()

        add_h_ray_action = QAction("Add Horizontal Ray", self)
        add_h_ray_action.setShortcut("H")
        add_h_ray_action.triggered.connect(lambda: self._enter_ray_mode('h'))
        view_menu.addAction(add_h_ray_action)

        add_v_ray_action = QAction("Add Vertical Ray", self)
        add_v_ray_action.setShortcut("V")
        add_v_ray_action.triggered.connect(lambda: self._enter_ray_mode('v'))
        view_menu.addAction(add_v_ray_action)

        clear_rays_action = QAction("Clear All Rays", self)
        clear_rays_action.triggered.connect(self.clear_all_rays)
        view_menu.addAction(clear_rays_action)

        analysis_menu = menu_bar.addMenu("Analysis")

        compute_bg_action = QAction("Compute Background...", self)
        compute_bg_action.triggered.connect(self.compute_background_dialog)
        analysis_menu.addAction(compute_bg_action)

        analysis_menu.addSeparator()

        self.bg_subtract_action = QAction("Subtract Background", self, checkable=True)
        self.bg_subtract_action.setEnabled(False)
        self.bg_subtract_action.triggered.connect(self.toggle_background_subtraction)
        analysis_menu.addAction(self.bg_subtract_action)

        self.save_bg_action = QAction("Save Background Image...", self)
        self.save_bg_action.setEnabled(False)
        self.save_bg_action.triggered.connect(self.save_background_dialog)
        analysis_menu.addAction(self.save_bg_action)

        clear_bg_action = QAction("Clear Background", self)
        clear_bg_action.triggered.connect(self.clear_background)
        analysis_menu.addAction(clear_bg_action)

        analysis_menu.addSeparator()

        self.threshold_action = QAction("Adaptive Threshold", self, checkable=True)
        self.threshold_action.triggered.connect(self.toggle_threshold)
        analysis_menu.addAction(self.threshold_action)

        analysis_menu.addSeparator()

        self.morphology_action = QAction("Morphology (mask)", self, checkable=True)
        self.morphology_action.triggered.connect(
            lambda checked: self._toggle_filter(MorphologyOp, self.morphology_action, checked))
        analysis_menu.addAction(self.morphology_action)

        self.binary_smooth_action = QAction("Binary Smooth (mask)", self, checkable=True)
        self.binary_smooth_action.triggered.connect(
            lambda checked: self._toggle_filter(BinarySmoothOp, self.binary_smooth_action, checked))
        analysis_menu.addAction(self.binary_smooth_action)

        self.watershed_action = QAction("Watershed Split (mask)", self, checkable=True)
        self.watershed_action.triggered.connect(
            lambda checked: self._toggle_filter(WatershedSplitOp, self.watershed_action, checked))
        analysis_menu.addAction(self.watershed_action)

        self.intensity_watershed_action = QAction("Intensity Watershed Split (mask)", self, checkable=True)
        self.intensity_watershed_action.triggered.connect(
            lambda checked: self._toggle_filter(
                IntensityWatershedSplitOp, self.intensity_watershed_action, checked
            )
        )
        analysis_menu.addAction(self.intensity_watershed_action)

        analysis_menu.addSeparator()
        blob_size_action = QAction("Blob Size Analysis…", self)
        blob_size_action.setToolTip(
            "Histogram of connected-blob areas across sampled frames.\n"
            "Use the tail of the distribution to set min_blob_area in "
            "IntensityWatershedSplitOp."
        )
        blob_size_action.triggered.connect(self._show_blob_size_window)
        analysis_menu.addAction(blob_size_action)

        region_props_action = QAction("Region Props Analysis…", self)
        region_props_action.setToolTip(
            "Per-blob regionprops (area, axes, eccentricity) across sampled frames.\n"
            "Shows stats in px and physical units when a pixel scale is set."
        )
        region_props_action.triggered.connect(self._show_region_props_window)
        analysis_menu.addAction(region_props_action)

        mie_action = QAction("Mie Particle Sizing…", self)
        mie_action.setToolTip(
            "Size particles by treating each blob area (m²) as its Mie scattering\n"
            "cross-section, then fitting a Rosin-Rammler distribution (D10/D50/D90)."
        )
        mie_action.triggered.connect(self._show_mie_window)
        analysis_menu.addAction(mie_action)

        filters_menu = menu_bar.addMenu("Filters")

        self.clahe_action = QAction("CLAHE", self, checkable=True)
        self.clahe_action.triggered.connect(lambda checked: self._toggle_filter(CLAHEOp, self.clahe_action, checked))
        filters_menu.addAction(self.clahe_action)

        self.gaussian_action = QAction("Gaussian Blur", self, checkable=True)
        self.gaussian_action.triggered.connect(lambda checked: self._toggle_filter(GaussianBlurOp, self.gaussian_action, checked))
        filters_menu.addAction(self.gaussian_action)

        self.sharpen_action = QAction("Sharpen", self, checkable=True)
        self.sharpen_action.triggered.connect(lambda checked: self._toggle_filter(SharpenOp, self.sharpen_action, checked))
        filters_menu.addAction(self.sharpen_action)

        filters_menu.addSeparator()

        self.lowpass_action = QAction("Low Pass (box)", self, checkable=True)
        self.lowpass_action.triggered.connect(lambda checked: self._toggle_filter(LowPassOp, self.lowpass_action, checked))
        filters_menu.addAction(self.lowpass_action)

        self.highpass_action = QAction("High Pass (orig − box)", self, checkable=True)
        self.highpass_action.triggered.connect(lambda checked: self._toggle_filter(HighPassOp, self.highpass_action, checked))
        filters_menu.addAction(self.highpass_action)

        self.rollingball_action = QAction("Rolling Ball BG (per-frame)", self, checkable=True)
        self.rollingball_action.triggered.connect(lambda checked: self._toggle_filter(RollingBallBgOp, self.rollingball_action, checked))
        filters_menu.addAction(self.rollingball_action)

        tools_menu = menu_bar.addMenu("Tools")
        scale_action = QAction("Set Scale…", self)
        scale_action.setShortcut("Ctrl+M")
        scale_action.triggered.connect(self.show_scale_dialog)
        tools_menu.addAction(scale_action)

        clear_scale_action = QAction("Clear Scale", self)
        clear_scale_action.triggered.connect(self.clear_scale)
        tools_menu.addAction(clear_scale_action)

        tools_menu.addSeparator()

        facet_action = QAction("Facet Thickness Panel", self)
        facet_action.setShortcut("Ctrl+Shift+P")
        facet_action.setToolTip(
            "Measure thin-film thickness perpendicular to a slanted substrate facet.\n"
            "Two-phase: click the interface, then click the outer surface."
        )
        facet_action.triggered.connect(self.toggle_facet_panel)
        tools_menu.addAction(facet_action)

        facet_start_action = QAction("Start / Resume Facet Clicking", self)
        facet_start_action.setShortcut("Ctrl+Shift+A")
        facet_start_action.triggered.connect(self._facet_start_or_resume)
        tools_menu.addAction(facet_start_action)

    # ------------------------------------------------------------------
    # Legacy Analysis menu shims (delegate to pipeline ops)
    # ------------------------------------------------------------------

    def compute_background_dialog(self):
        if self.sequence_manager is None:
            QMessageBox.warning(self, "No Sequence", "Open a TIFF sequence first.")
            return
        if self.num_frames < 2:
            QMessageBox.warning(self, "Too Few Frames", "Need at least 2 frames to compute background.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Compute Background")
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("Method:"))
        method_combo = QComboBox()
        method_combo.addItems(["Temporal Median (recommended)", "Temporal Mean"])
        layout.addWidget(method_combo)

        layout.addWidget(QLabel("Frames to sample:"))
        sample_spin = QSpinBox()
        sample_spin.setRange(10, self.num_frames)
        sample_spin.setValue(min(50, self.num_frames))

        # Determine the frame size that will actually be stacked: apply the same
        # upstream ops that fit_with_progress will use (everything before BgSubtractOp).
        # BgSubtractOp may not exist yet, so simulate "appended at end" position.
        _bg_idx_preview, _ = self._find_op(BgSubtractOp)
        _preview_op_idx = _bg_idx_preview if _bg_idx_preview >= 0 else len(self.pipeline.operations)
        _upstream_preview = [op for op in self.pipeline.operations[:_preview_op_idx] if op.enabled]
        if self.current_raw_frame is not None:
            _sample_frame = self.current_raw_frame
            for _op in _upstream_preview:
                if not isinstance(_op, RotateOp):  # RotateOp already applied to current_raw_frame
                    _sample_frame = _op.apply(_sample_frame)
            H, W = _sample_frame.shape[:2]
        else:
            H, W = 0, 0

        # Warn if CropOp exists downstream of BgSubtractOp — BG will be full-frame.
        _crop_idx, _ = self._find_op(CropOp)
        _crop_downstream = (_crop_idx >= 0 and _bg_idx_preview >= 0 and _crop_idx > _bg_idx_preview)
        if _crop_downstream:
            warn_lbl = QLabel("Warning: CropOp is after BgSubtractOp — background will be full-frame (slow). "
                              "Drag BgSubtractOp below CropOp in the Pipeline panel first.")
            warn_lbl.setStyleSheet("color: #ffa500; font-weight: bold;")
            warn_lbl.setWordWrap(True)
            layout.addWidget(warn_lbl)

        def _free_mem_mb():
            try:
                with open('/proc/meminfo') as f:
                    for line in f:
                        if line.startswith('MemAvailable:'):
                            return int(line.split()[1]) / 1024
            except Exception:
                return None

        free_mb = _free_mem_mb()
        mem_label = QLabel()

        def update_mem(val):
            est_mb = val * H * W * 4 / 1e6
            if free_mb is not None:
                if est_mb > free_mb:
                    mem_label.setText(f"~{est_mb:.0f} MB needed — WARNING: only {free_mb:.0f} MB free.")
                    mem_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
                elif est_mb > free_mb * 0.75:
                    mem_label.setText(f"~{est_mb:.0f} MB needed — {free_mb:.0f} MB free (tight)")
                    mem_label.setStyleSheet("color: #ffa500;")
                else:
                    mem_label.setText(f"~{est_mb:.0f} MB needed — {free_mb:.0f} MB free")
                    mem_label.setStyleSheet("")
            else:
                mem_label.setText(f"~{est_mb:.0f} MB (float32)")
                mem_label.setStyleSheet("")

        sample_spin.valueChanged.connect(update_mem)
        update_mem(sample_spin.value())
        layout.addWidget(sample_spin)
        layout.addWidget(mem_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        method = 'median' if method_combo.currentIndex() == 0 else 'mean'
        sample_n = sample_spin.value()

        op_idx, op = self._find_op(BgSubtractOp)
        if op is None:
            op = BgSubtractOp()
            op_idx = self.pipeline.add_operation(op, at_idx=-1)
            self.pipeline_panel.rebuild_list()

        op.params["method"] = method
        op.params["sample_n"] = sample_n
        self._run_bg_fit(op, op_idx)

        self.bg_subtract_action.setEnabled(True)
        self.bg_subtract_action.setChecked(True)
        op.enabled = True
        self.save_bg_action.setEnabled(True)
        self._select_op_in_panel(BgSubtractOp)

    def toggle_background_subtraction(self, checked):
        idx, op = self._find_op(BgSubtractOp)
        if op is None:
            self.bg_subtract_action.setChecked(False)
            return
        op.enabled = checked
        self.pipeline._invalidate_from(idx)
        self._invalidate_threshold_caches()
        self.pipeline_panel.rebuild_list()
        self.update_frame_display()

    def clear_background(self):
        idx, op = self._find_op(BgSubtractOp)
        if op is not None:
            op.set_background(None)
            op.enabled = False
            self.pipeline._invalidate_from(idx)
        self.bg_subtract_action.setChecked(False)
        self.bg_subtract_action.setEnabled(False)
        self.save_bg_action.setEnabled(False)
        self._invalidate_threshold_caches()
        self.pipeline_panel.rebuild_list()
        self.update_frame_display()

    def save_background_dialog(self):
        _, op = self._find_op(BgSubtractOp)
        if op is None or op.get_background() is None or self.sequence_manager is None:
            return

        seq_folder = self.sequence_manager.folder_path
        dlg = QDialog(self)
        dlg.setWindowTitle("Save Background Image")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(f"Sequence folder:\n{seq_folder}"))
        form = QFormLayout()
        subdir_edit = QLineEdit("background")
        filename_edit = QLineEdit("background.tif")
        form.addRow("Subdirectory name:", subdir_edit)
        form.addRow("Filename:", filename_edit)
        layout.addLayout(form)
        preview_label = QLabel()
        layout.addWidget(preview_label)

        def _update_preview():
            subdir = subdir_edit.text().strip() or "background"
            fname = filename_edit.text().strip() or "background.tif"
            preview_label.setText(f"→ {os.path.join(seq_folder, subdir, fname)}")

        subdir_edit.textChanged.connect(_update_preview)
        filename_edit.textChanged.connect(_update_preview)
        _update_preview()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        subdir = subdir_edit.text().strip() or "background"
        fname = filename_edit.text().strip() or "background.tif"
        if not fname.lower().endswith((".tif", ".tiff")):
            fname += ".tif"

        save_dir = os.path.join(seq_folder, subdir)
        os.makedirs(save_dir, exist_ok=True)

        try:
            bg = op.get_background()
            if self.current_raw_frame is not None and np.issubdtype(self.current_raw_frame.dtype, np.integer):
                max_val = np.iinfo(self.current_raw_frame.dtype).max
                bg_to_save = np.clip(bg, 0, max_val).astype(self.current_raw_frame.dtype)
            else:
                bg_to_save = bg
            tifffile.imwrite(os.path.join(save_dir, fname), bg_to_save)
            self.status_label.setText(f"Background saved: {os.path.join(save_dir, fname)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def _sync_filter_menu_actions(self):
        """Keep Filters menu checkmarks in sync with pipeline op enabled state."""
        if not hasattr(self, 'clahe_action'):
            return
        for cls, action in [
            (CLAHEOp, self.clahe_action),
            (GaussianBlurOp, self.gaussian_action),
            (SharpenOp, self.sharpen_action),
            (LowPassOp, self.lowpass_action),
            (HighPassOp, self.highpass_action),
            (RollingBallBgOp, self.rollingball_action),
            (MorphologyOp, self.morphology_action),
            (BinarySmoothOp, self.binary_smooth_action),
            (WatershedSplitOp, self.watershed_action),
            (IntensityWatershedSplitOp, self.intensity_watershed_action),
        ]:
            _, op = self._find_op(cls)
            action.blockSignals(True)
            action.setChecked(op is not None and op.enabled)
            action.blockSignals(False)

    def _toggle_filter(self, cls, action, checked):
        if checked:
            _, op = self._ensure_op(cls)
            op.enabled = True
            self.pipeline_panel.rebuild_list()
            self._select_op_in_panel(cls)
        else:
            idx, op = self._find_op(cls)
            if op is not None:
                op.enabled = False
                self.pipeline._invalidate_from(idx)
                self._clear_bg_if_upstream_changed(idx)
                self.pipeline_panel.rebuild_list()
        self.update_frame_display()

    def toggle_threshold(self, checked):
        if checked:
            _, op = self._ensure_op(AdaptiveThresholdOp)
            op.enabled = True
            self.pipeline_panel.rebuild_list()
            self._select_op_in_panel(AdaptiveThresholdOp)
        else:
            idx, op = self._find_op(AdaptiveThresholdOp)
            if op is not None:
                op.enabled = False
                self.pipeline._invalidate_from(idx)
                self.pipeline_panel.rebuild_list()
        self.update_frame_display()

    def save_threshold_masks(self):
        idx, op = self._find_op(AdaptiveThresholdOp)
        if op is None or not self.pipeline._caches[idx]:
            return

        seq_folder = self.sequence_manager.folder_path
        dlg = QDialog(self)
        dlg.setWindowTitle("Save Mask Sequence")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(f"Sequence folder:\n{seq_folder}"))
        form = QFormLayout()
        subdir_edit = QLineEdit("masks")
        form.addRow("Subdirectory name:", subdir_edit)
        layout.addLayout(form)
        preview_label = QLabel()
        layout.addWidget(preview_label)

        def _update_preview():
            subdir = subdir_edit.text().strip() or "masks"
            preview_label.setText(
                f"→ {os.path.join(seq_folder, subdir, '[original_filename].tif')}"
            )

        subdir_edit.textChanged.connect(_update_preview)
        _update_preview()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        subdir = subdir_edit.text().strip() or "masks"
        save_dir = os.path.join(seq_folder, subdir)
        os.makedirs(save_dir, exist_ok=True)

        cache = self.pipeline._caches[idx]
        total = len(cache)
        # Collect downstream binary-mask ops once, outside the loop.
        binary_ops = [
            op for op in self.pipeline.operations[idx + 1:]
            if op.enabled and getattr(op, "is_binary_mask_op", False)
        ]
        # Pre-check: any op needs intensity context → must load raw frame per iteration.
        needs_intensity = any(
            getattr(op, "requires_intensity_context", False) for op in binary_ops
        )
        for count, (frame_idx, mask) in enumerate(sorted(cache.items())):
            self.status_label.setText(f"Saving mask {count + 1} / {total}...")
            QApplication.processEvents()
            raw_for_ctx = None
            if needs_intensity:
                raw_for_ctx = self.sequence_manager.get_frame(frame_idx)
            for op_j in binary_ops:
                context = None
                if getattr(op_j, "requires_intensity_context", False) and raw_for_ctx is not None:
                    src_idx = op_j.params.get("intensity_source_idx", -1)
                    if src_idx < 0:
                        # -1 = pre-threshold frame: apply all ops before AdaptiveThresholdOp
                        # so shape (and rotation/crop) matches the stored mask.
                        if idx > 0:
                            intensity_f = self.pipeline.apply_to_frame(
                                raw_for_ctx, frame_idx, target_stage_idx=idx - 1
                            )
                        else:
                            intensity_f = raw_for_ctx
                    else:
                        intensity_f = self.pipeline.apply_to_frame(
                            raw_for_ctx, frame_idx, target_stage_idx=src_idx
                        )
                    context = {"intensity_frame": intensity_f}
                mask = op_j.apply(mask, context=context)
            fname = self.sequence_manager.files[frame_idx]
            tifffile.imwrite(os.path.join(save_dir, fname), mask)

        self.status_label.setText(f"Saved {total} mask(s) to {save_dir}")

    # ------------------------------------------------------------------
    # Pipeline serialization dialogs
    # ------------------------------------------------------------------

    def save_pipeline_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Pipeline", "", "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w") as f:
                f.write(self.pipeline.to_json())
            self.status_label.setText(f"Pipeline saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def load_pipeline_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Pipeline", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                text = f.read()
            new_pipeline = Pipeline.from_json(text)
            self.pipeline = new_pipeline
            self._invalidate_threshold_caches()
            self.pipeline_panel.rebuild_list()
            # Sync display and ViewBox to the new pipeline before any auto-fit runs.
            # Without this, the screen shows the stale pre-load image (typically the
            # full uncropped frame) while the BG progress dialog is open.
            self.update_frame_display()
            self.view_box.autoRange()
            # re-fit any BgSubtractOp that has no background
            if self.sequence_manager is not None:
                for i, op in enumerate(self.pipeline.operations):
                    if isinstance(op, BgSubtractOp) and op.get_background() is None and op.enabled:
                        self._run_bg_fit(op, i)
            self.update_frame_display()
            self.view_box.autoRange()
            self.status_label.setText(f"Pipeline loaded: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    # ------------------------------------------------------------------
    # Rotate (keyboard shortcut / menu)
    # ------------------------------------------------------------------

    def rotate_image(self):
        idx, op = self._find_op(RotateOp)
        if op is None:
            op = RotateOp()
            op.params["k"] = 1
            self.pipeline.add_operation(op, at_idx=0)
        else:
            op.params["k"] = (op.params["k"] + 1) % 4

        _, bg_op = self._find_op(BgSubtractOp)
        if bg_op is not None:
            bg_op.set_background(None)

        self.pipeline._invalidate_from(0)
        self._invalidate_threshold_caches()
        self.pipeline_panel.rebuild_list()
        self.update_frame_display()
        self.reset_view()

    # ------------------------------------------------------------------
    # Crop (UI draw state → CropOp params)
    # ------------------------------------------------------------------

    def _sync_crop_params(self):
        """Sync crop spinboxes in pipeline panel if CropOp is currently selected."""
        crop = self._crop_region()
        if crop is None:
            return
        x, y, w, h = crop
        pw = self.pipeline_panel.params_scroll.widget()
        if not isinstance(pw, CropParamsWidget):
            return
        if self.current_raw_frame is not None:
            img_h, img_w = self.current_raw_frame.shape[:2]
            pw.update_bounds(img_w, img_h)
        pw.sync(x, y, w, h)

    def _apply_crop_from_panel(self, x, y, w, h):
        if self.current_raw_frame is None:
            return
        img_h, img_w = self.current_raw_frame.shape
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))

        idx, op = self._find_op(CropOp)
        if op is None:
            op = CropOp()
            idx = self.pipeline.add_operation(op)
            self.pipeline_panel.rebuild_list()
        op.params.update({"x": x, "y": y, "w": w, "h": h})
        self.pipeline._invalidate_from(idx)
        self._clear_bg_if_upstream_changed(idx)
        self._sync_crop_params()

        if self.crop_mode and self.crop_roi_item is not None:
            self.crop_roi_item.sigRegionChanged.disconnect(self._on_roi_adjusted)
            self.crop_roi_item.setPos([x, y])
            self.crop_roi_item.setSize([w, h])
            self.crop_roi_item.sigRegionChanged.connect(self._on_roi_adjusted)
        elif self.crop_mode:
            self._create_interactive_roi(x, y, w, h)

        if not self.crop_mode:
            self.update_frame_display()

        self._invalidate_threshold_caches()
        suffix = " — drag handles to adjust. Enter=apply, Esc=cancel." if self.crop_mode else ""
        self.status_label.setText(f"ROI: x={x}, y={y}, w={w}, h={h}{suffix}")

    def toggle_crop_mode(self, checked):
        self.crop_mode = checked
        if self.sequence_manager is None or self.current_raw_frame is None:
            self.crop_mode = False
            self.crop_action.setChecked(False)
            return

        if checked:
            self._crop_region_snapshot = self._crop_region()
            self.glw.setCursor(Qt.CursorShape.CrossCursor)
            cr = self._crop_region()
            if cr is not None:
                cx, cy, cw, ch = cr
                self._create_interactive_roi(cx, cy, cw, ch)
            self.update_frame_display()
            self._sync_crop_params()
            self.status_label.setText("Crop mode — click and drag to draw ROI. Enter=apply, Esc=cancel.")
        else:
            self._exit_crop_mode()

    def _exit_crop_mode(self):
        if self.crop_roi_item is not None:
            self.view_box.removeItem(self.crop_roi_item)
            self.crop_roi_item = None
        self._drawing_crop = False
        self._crop_draw_start = None
        self.crop_mode = False
        self.crop_action.setChecked(False)
        self.glw.setCursor(Qt.CursorShape.ArrowCursor)
        self.update_frame_display()

    def _confirm_crop(self):
        if self._crop_region() is None:
            self._exit_crop_mode()
            return
        crop_idx, _ = self._find_op(CropOp)
        if crop_idx >= 0:
            self._clear_bg_if_upstream_changed(crop_idx)
        self._invalidate_threshold_caches()
        self._exit_crop_mode()
        self.view_box.autoRange()

    def clear_crop(self):
        idx, op = self._find_op(CropOp)
        if op is not None:
            op.enabled = False
            self.pipeline._invalidate_from(idx)
            self._clear_bg_unconditionally()
        self._invalidate_threshold_caches()
        if self.crop_mode:
            self._exit_crop_mode()
        else:
            self.update_frame_display()
        self.view_box.autoRange()

    def apply_crop_dialog(self):
        cr = self._crop_region()
        if cr is None:
            self.status_label.setText("No crop region defined. Use Crop Mode first.")
            return
        if self.sequence_manager is None:
            return

        x, y, w, h = cr
        msg = QMessageBox(self)
        msg.setWindowTitle("Apply Crop to Files")
        msg.setText(
            f"Crop region: x={x}, y={y}, w={w}, h={h}\n\n"
            "This will PERMANENTLY OVERWRITE the original TIFF files on disk.\n"
            "The current rotation will be baked in. This cannot be undone.\n\n"
            "Apply to:"
        )
        btn_all = msg.addButton("All Frames", QMessageBox.ButtonRole.AcceptRole)
        btn_current = msg.addButton("Current Frame Only", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_all:
            self._write_cropped_frames(range(self.num_frames))
        elif clicked == btn_current:
            self._write_cropped_frames([self.current_frame])

    def _write_cropped_frames(self, indices):
        cr = self._crop_region()
        x, y, w, h = cr
        rot_k = self._rotation_state()
        index_list = list(indices)
        total = len(index_list)

        for i, idx in enumerate(index_list):
            self.status_label.setText(f"Writing frame {i + 1} / {total}...")
            QApplication.processEvents()
            file_path = os.path.join(self.sequence_manager.folder_path, self.sequence_manager.files[idx])
            raw = tifffile.imread(file_path)
            rotated = np.rot90(raw, k=-rot_k)
            cropped = rotated[y:y + h, x:x + w]
            tifffile.imwrite(file_path, cropped)

        self.sequence_manager.cache.clear()
        # remove RotateOp and CropOp — baked in
        self.pipeline.operations = [
            op for op in self.pipeline.operations
            if not isinstance(op, (RotateOp, CropOp))
        ]
        self.pipeline._caches = [{} for _ in self.pipeline.operations]
        _, bg_op = self._find_op(BgSubtractOp)
        if bg_op is not None:
            bg_op.set_background(None)
            bg_op.enabled = False
        self.bg_subtract_action.setChecked(False)
        self.bg_subtract_action.setEnabled(False)
        self.pipeline_panel.rebuild_list()
        self.sequence_manager.prefetch_window(self.current_frame)
        self.update_frame_display()
        self.view_box.autoRange()
        self.status_label.setText(f"Done. {total} frame(s) overwritten with crop applied.")

    # ------------------------------------------------------------------
    # Rays
    # ------------------------------------------------------------------

    def _enter_ray_mode(self, mode):
        if self.sequence_manager is None or self.current_raw_frame is None:
            return
        self.ray_mode = mode
        self.glw.setCursor(Qt.CursorShape.CrossCursor)
        label = "horizontal" if mode == 'h' else "vertical"
        self.status_label.setText(f"Click on image to place {label} ray. Esc to cancel.")

    def _place_ray(self, img_pos):
        if self.ray_mode == 'h':
            pos = img_pos.y(); angle = 0; prefix = 'y'
        else:
            pos = img_pos.x(); angle = 90; prefix = 'x'

        line = pg.InfiniteLine(
            pos=pos, angle=angle, movable=True,
            pen=pg.mkPen(color=(255, 100, 100), width=1.5, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen(color=(255, 220, 80), width=2.5),
            label=f'{prefix}={{value:.0f}}',
            labelOpts={'position': 0.05, 'color': (255, 230, 100),
                       'fill': (30, 30, 30, 180), 'movable': True},
        )
        self.view_box.addItem(line)
        self.rays.append(line)
        self.ray_mode = None
        self.glw.setCursor(Qt.CursorShape.ArrowCursor)
        self.status_label.setText(f"Ray placed at {prefix}={int(round(pos))}. Add more via View menu (H / V).")

    def clear_all_rays(self):
        for ray in self.rays:
            self.view_box.removeItem(ray)
        self.rays.clear()

    # ------------------------------------------------------------------
    # Facet thickness measurement
    #
    # Purely additive: none of this reads or mutates the processing pipeline
    # beyond borrowing pipeline.scale for unit conversion. Points are clicked
    # on the displayed geometry, in the same coordinate space as the rays.
    # ------------------------------------------------------------------

    def _facet_geometry_tag(self) -> str:
        """Tag the display geometry so stale sessions can be spotted.

        Points are stored in display coordinates, so a later rotate or crop
        moves the image out from under them. Sessions record the tag in force
        when they were measured; the panel compares and warns.
        """
        _, rot_op = self._find_op(RotateOp)
        k = rot_op.params.get("k", 0) if (rot_op is not None and rot_op.enabled) else 0
        region = self._crop_region()
        crop = "none" if region is None else ",".join(str(v) for v in region)
        return f"rot{k}|crop{crop}"

    def active_facet_session(self):
        if 0 <= self.active_facet_idx < len(self.facet_sessions):
            return self.facet_sessions[self.active_facet_idx]
        return None

    def toggle_facet_panel(self):
        if self.facet_panel.isVisible():
            self.facet_panel.hide()
            self._exit_facet_mode()
        else:
            self.facet_panel.show()
            self.facet_panel.raise_()
            self.refresh_facets()

    def new_facet_session(self, label):
        session = FacetSession(
            label=label,
            source_filename=(self.sequence_manager.files[self.current_frame]
                             if self.sequence_manager is not None else ""),
            folder_path=(self.sequence_manager.folder_path
                         if self.sequence_manager is not None else ""),
            frame_idx=self.current_frame,
            scale=dict(self.pipeline.scale) if self.pipeline.scale else None,
            color_idx=len(self.facet_sessions),
            geometry_tag=self._facet_geometry_tag(),
        )
        self.facet_sessions.append(session)
        self.active_facet_idx = len(self.facet_sessions) - 1
        self.facet_mode = 'interface'
        self.glw.setCursor(Qt.CursorShape.CrossCursor)
        self.refresh_facets()
        self.status_label.setText(
            f"Facet “{label}” — Phase A: click interface points, then "
            f"“Finish Interface”. Esc to pause clicking."
        )

    def set_active_facet_session(self, idx):
        if not (0 <= idx < len(self.facet_sessions)):
            return
        self.active_facet_idx = idx
        session = self.facet_sessions[idx]
        if self.facet_mode is not None:
            self.facet_mode = session.phase
        self.refresh_facets()

    def delete_facet_session(self, session):
        graphics = self._facet_graphics.pop(id(session), None)
        if graphics is not None:
            graphics.remove()
        if session in self.facet_sessions:
            self.facet_sessions.remove(session)
        self.active_facet_idx = min(self.active_facet_idx, len(self.facet_sessions) - 1)
        if not self.facet_sessions:
            self._exit_facet_mode()
        self.refresh_facets()

    def clear_all_facet_sessions(self):
        for graphics in self._facet_graphics.values():
            graphics.remove()
        self._facet_graphics.clear()
        self.facet_sessions.clear()
        self.active_facet_idx = -1
        self._exit_facet_mode()
        if self.facet_panel.isVisible():
            self.facet_panel.session_list.clear()
            self.facet_panel.refresh()

    def _exit_facet_mode(self):
        if self.facet_mode is not None:
            self.facet_mode = None
            self.glw.setCursor(Qt.CursorShape.ArrowCursor)

    def _facet_start_or_resume(self):
        """Ctrl+Shift+A: open the panel and resume clicking on the active session."""
        if self.sequence_manager is None:
            QMessageBox.warning(self, "No Sequence", "Open a TIFF sequence first.")
            return
        self.facet_panel.show()
        self.facet_panel.raise_()
        session = self.active_facet_session()
        if session is None:
            self.facet_panel._on_new_session()
            return
        self.facet_mode = session.phase
        self.glw.setCursor(Qt.CursorShape.CrossCursor)
        self.refresh_facets()

    # -- rendering -------------------------------------------------------

    def _facet_visible_sessions(self):
        """Sessions that belong on the current frame and are toggled visible."""
        return [s for s in self.facet_sessions
                if s.visible and s.frame_idx == self.current_frame]

    def refresh_facets(self):
        """Redraw every facet overlay and refresh the panel."""
        live_ids = set()
        visible = self._facet_visible_sessions()
        image_hw = self._display_hw

        for session in self.facet_sessions:
            key = id(session)
            live_ids.add(key)
            graphics = self._facet_graphics.get(key)
            if graphics is None:
                colour = FACET_COLORS[session.color_idx % len(FACET_COLORS)]
                graphics = FacetGraphics(self.view_box, colour)
                self._facet_graphics[key] = graphics
            if session in visible:
                graphics.set_visible(True)
                graphics.update(session, image_hw)
            else:
                graphics.set_visible(False)

        for key in [k for k in self._facet_graphics if k not in live_ids]:
            self._facet_graphics.pop(key).remove()

        if self.facet_panel.isVisible():
            self.facet_panel.refresh()

    # -- click / drag handling -------------------------------------------

    def _facet_hit_test(self, img_pos):
        """Nearest draggable facet point under the cursor, or None.

        Returns (session, kind, index). The hit radius is defined in screen
        pixels and converted to view units so it stays usable at any zoom.
        """
        try:
            px_w, px_h = self.view_box.viewPixelSize()
        except Exception:
            px_w = px_h = 1.0
        radius = 9.0 * max(px_w, px_h)
        radius_sq = radius * radius

        best = None
        best_dist = radius_sq
        x, y = img_pos.x(), img_pos.y()
        for session in self._facet_visible_sessions():
            for kind in ("surface", "interface"):
                points = (session.surface_points if kind == "surface"
                          else session.interface_points)
                for i, (px, py) in enumerate(points):
                    d = (px - x) ** 2 + (py - y) ** 2
                    if d <= best_dist:
                        best_dist = d
                        best = (session, kind, i)
        return best

    def _facet_on_press(self, img_pos):
        """Left-press in the image. True if the event was consumed."""
        hit = self._facet_hit_test(img_pos)
        if hit is not None:
            self._facet_drag = hit
            session, kind, i = hit
            self.active_facet_idx = self.facet_sessions.index(session)
            self.refresh_facets()
            return True

        if self.facet_mode is None:
            return False

        session = self.active_facet_session()
        if session is None:
            return False
        if session.frame_idx != self.current_frame:
            self.status_label.setText(
                f"Session “{session.label}” belongs to frame {session.frame_idx + 1}. "
                f"Scrub back to it, or start a new session on this frame."
            )
            return True

        phase = session.add_point(img_pos.x(), img_pos.y())
        self.refresh_facets()
        if phase == "interface":
            self.status_label.setText(
                f"Interface points: {len(session.interface_points)}. "
                f"Click “Finish Interface” when the line looks right."
            )
        else:
            measurements = session.measurements()
            if measurements:
                last = measurements[-1]
                value = session.to_units(last.thickness_px)
                self.status_label.setText(
                    f"Thickness #{len(measurements)}: {value:.4g} {session.unit} "
                    f"at foot x = {session.to_units(last.foot_x):.4g} {session.unit}"
                )
        return True

    def _facet_on_drag(self, img_pos):
        if self._facet_drag is None:
            return False
        session, kind, i = self._facet_drag
        session.move_point(kind, i, img_pos.x(), img_pos.y())
        self.refresh_facets()
        return True

    def _facet_on_release(self):
        if self._facet_drag is None:
            return False
        self._facet_drag = None
        self.refresh_facets()
        return True

    # -- persistence ------------------------------------------------------

    def save_facet_sessions_dialog(self):
        if not self.facet_sessions:
            QMessageBox.information(self, "Nothing to Save", "No facet sessions yet.")
            return
        start_dir = self.sequence_manager.folder_path if self.sequence_manager else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Facet Sessions", os.path.join(start_dir, "facet_sessions.json"),
            "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(sessions_to_json(self.facet_sessions))
        except OSError as e:
            QMessageBox.critical(self, "Save Failed", str(e))
            return
        self.status_label.setText(
            f"Saved {len(self.facet_sessions)} facet session(s) to {os.path.basename(path)}"
        )

    def load_facet_sessions_dialog(self):
        start_dir = self.sequence_manager.folder_path if self.sequence_manager else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Facet Sessions", start_dir, "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                loaded = sessions_from_json(fh.read())
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Load Failed", str(e))
            return
        self.clear_all_facet_sessions()
        self.facet_sessions = loaded
        self.active_facet_idx = 0 if loaded else -1
        self.facet_panel.session_list.clear()
        self.facet_panel.show()
        self.refresh_facets()
        self.status_label.setText(
            f"Loaded {len(loaded)} facet session(s) from {os.path.basename(path)}"
        )

    def export_facet_csv_dialog(self):
        if not any(s.measurements() for s in self.facet_sessions):
            QMessageBox.information(
                self, "Nothing to Export",
                "No completed measurements yet — a session needs a fitted "
                "interface and at least one surface point."
            )
            return
        start_dir = self.sequence_manager.folder_path if self.sequence_manager else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Facet Measurements",
            os.path.join(start_dir, "facet_thickness.csv"), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            n = sessions_to_csv(path, self.facet_sessions)
        except OSError as e:
            QMessageBox.critical(self, "Export Failed", str(e))
            return
        self.status_label.setText(
            f"Exported {n} facet session(s) to {os.path.basename(path)}"
        )

    # ------------------------------------------------------------------
    # Pixel scale / measurement
    # ------------------------------------------------------------------

    def _refresh_status_label(self):
        """Rebuild the main status line, including bg-stale and scale info."""
        if self.sequence_manager is None:
            _mod = "Cmd" if sys.platform == "darwin" else "Ctrl"
            self.status_label.setText(
                f"Please open a specific TIFF file from the File menu ({_mod}+O)."
            )
            return
        text = f"Frame: {self.current_frame + 1} / {self.num_frames}   (Bit Depth: {self.bit_depth_max})"
        _, bg_op = self._find_op(BgSubtractOp)
        if bg_op is not None and bg_op.enabled and bg_op.get_background() is None:
            text += "   | BgSubtractOp: no background — open Pipeline, select it, click Fit"
        if self.pipeline.scale is not None:
            s = self.pipeline.scale
            text += f"   |  Scale: {s['px']} px = {_fmt_num(s['value'])} {s['unit']}"
        self.status_label.setText(text)

    def clear_scale(self):
        if self.pipeline.scale is None:
            return
        self.pipeline.scale = None
        self._refresh_status_label()

    def show_scale_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Set Pixel Scale")
        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        form = QFormLayout()
        px_edit = QLineEdit()
        px_edit.setValidator(QIntValidator(1, 1_000_000, dlg))
        px_edit.setPlaceholderText("e.g. 42")
        form.addRow("Pixels:", px_edit)

        val_edit = QLineEdit()
        val_edit.setValidator(QDoubleValidator(0.0, 1e12, 6, dlg))
        val_edit.setPlaceholderText("e.g. 1.0")
        form.addRow("equals:", val_edit)

        unit_combo = QComboBox()
        unit_combo.addItems(["mm", "µm", "nm", "m", "cm"])
        form.addRow("Unit:", unit_combo)
        layout.addLayout(form)

        if self.pipeline.scale is not None:
            s = self.pipeline.scale
            px_edit.setText(str(s["px"]))
            val_edit.setText(_fmt_num(s["value"]))
            idx = unit_combo.findText(s["unit"])
            if idx >= 0:
                unit_combo.setCurrentIndex(idx)

        measure_btn = QPushButton("Measure on image…")
        layout.addWidget(measure_btn)

        info = QLabel("Tip: hold Shift while drawing to constrain to horizontal / vertical. "
                      "Endpoints snap to whole pixels.")
        info.setStyleSheet("color: #888; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(button_box)

        def _on_measured(px_value):
            if px_value is not None:
                px_edit.setText(str(px_value))
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()

        def _do_measure():
            if self.sequence_manager is None:
                QMessageBox.warning(dlg, "No Sequence", "Open a TIFF sequence first.")
                return
            dlg.hide()
            self._enter_measure_mode(_on_measured)

        measure_btn.clicked.connect(_do_measure)

        def _on_ok():
            try:
                px = int(px_edit.text())
                val = float(val_edit.text())
            except (ValueError, TypeError):
                QMessageBox.warning(dlg, "Invalid Input",
                                    "Enter a positive integer pixel count and a positive numeric value.")
                return
            if px <= 0 or val <= 0:
                QMessageBox.warning(dlg, "Invalid Input",
                                    "Pixel count and unit value must both be greater than zero.")
                return
            self.pipeline.scale = {
                "px": px,
                "value": val,
                "unit": unit_combo.currentText(),
            }
            self._refresh_status_label()
            dlg.accept()

        button_box.accepted.connect(_on_ok)
        button_box.rejected.connect(dlg.reject)

        dlg.exec()

    def _enter_measure_mode(self, callback):
        self.measure_mode = 'await_first'
        self._measure_first_pt = None
        self._measure_pending_callback = callback
        if self._measure_line_item is not None:
            self.view_box.removeItem(self._measure_line_item)
            self._measure_line_item = None
        if self._measure_label_item is not None:
            self.view_box.removeItem(self._measure_label_item)
            self._measure_label_item = None
        self.glw.setCursor(Qt.CursorShape.CrossCursor)
        self.status_label.setText("Click first point of measurement line. Esc to cancel.")

    def _exit_measure_mode(self, committed_px=None):
        self.measure_mode = False
        self._measure_first_pt = None
        if self._measure_line_item is not None:
            self.view_box.removeItem(self._measure_line_item)
            self._measure_line_item = None
        if self._measure_label_item is not None:
            self.view_box.removeItem(self._measure_label_item)
            self._measure_label_item = None
        self.glw.setCursor(Qt.CursorShape.ArrowCursor)
        cb = self._measure_pending_callback
        self._measure_pending_callback = None
        self._refresh_status_label()
        if cb is not None:
            cb(committed_px)

    def _measure_start_first(self, img_pos):
        x = int(round(img_pos.x()))
        y = int(round(img_pos.y()))
        self._measure_first_pt = (x, y)
        self._measure_line_item = pg.PlotDataItem(
            x=[x, x], y=[y, y],
            pen=pg.mkPen(color=(255, 230, 0), width=2),
        )
        self._measure_line_item.setZValue(8)
        self.view_box.addItem(self._measure_line_item)
        self._measure_label_item = pg.TextItem(
            "0 px", color=(255, 230, 0), anchor=(0.5, 1.2),
            fill=(30, 30, 30, 180),
        )
        self._measure_label_item.setZValue(8)
        self._measure_label_item.setPos(x, y)
        self.view_box.addItem(self._measure_label_item)
        self.measure_mode = 'await_second'
        self.status_label.setText(
            "Click second point. Hold Shift for horizontal / vertical. Esc to cancel."
        )

    def _measure_update_endpoint(self, img_pos, shift_held):
        """Refresh rubber-band line; returns (length_px, ex_int, ey_int) or None."""
        if self._measure_first_pt is None or self._measure_line_item is None:
            return None
        sx, sy = self._measure_first_pt
        ex = img_pos.x()
        ey = img_pos.y()
        if shift_held:
            if abs(ex - sx) >= abs(ey - sy):
                ey = sy
            else:
                ex = sx
        ex_i = int(round(ex))
        ey_i = int(round(ey))
        self._measure_line_item.setData(x=[sx, ex_i], y=[sy, ey_i])
        length = int(round(((ex_i - sx) ** 2 + (ey_i - sy) ** 2) ** 0.5))
        mx = (sx + ex_i) / 2.0
        my = (sy + ey_i) / 2.0
        self._measure_label_item.setText(f"{length} px")
        self._measure_label_item.setPos(mx, my)
        return length, ex_i, ey_i

    def _measure_commit_second(self, img_pos, shift_held):
        result = self._measure_update_endpoint(img_pos, shift_held)
        if result is None:
            return
        length, _, _ = result
        if length < 1:
            self.status_label.setText(
                "Zero-length line — click a different second point. Esc to cancel."
            )
            return
        self._exit_measure_mode(committed_px=length)

    # ------------------------------------------------------------------
    # Optical flow
    # ------------------------------------------------------------------

    def toggle_optical_flow(self, checked):
        self.optical_flow_enabled = checked
        self.update_frame_display()

    def show_optical_flow_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Optical Flow Settings")
        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        enable_cb = QCheckBox("Enable live optical flow  (Ctrl+F)")
        enable_cb.setChecked(self.optical_flow_enabled)
        layout.addWidget(enable_cb)

        layout.addWidget(_make_separator())

        layout.addWidget(QLabel("Acquisition mode:"))
        from PyQt6.QtWidgets import QButtonGroup
        btn_group = QButtonGroup(dlg)
        tr_radio = QRadioButton("Time-resolved  (A→B, B→C, C→D, …)")
        pw_radio = QRadioButton("Pairwise  (A→B, C→D, E→F, …)")
        btn_group.addButton(tr_radio)
        btn_group.addButton(pw_radio)
        if self.optical_flow_mode == 'pairwise':
            pw_radio.setChecked(True)
        else:
            tr_radio.setChecked(True)
        layout.addWidget(tr_radio)
        layout.addWidget(pw_radio)

        layout.addWidget(_make_separator())

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        def _apply():
            self.optical_flow_enabled = enable_cb.isChecked()
            self.flow_action.blockSignals(True)
            self.flow_action.setChecked(self.optical_flow_enabled)
            self.flow_action.blockSignals(False)
            self.optical_flow_mode = 'pairwise' if pw_radio.isChecked() else 'time_resolved'
            self.update_frame_display()

        enable_cb.toggled.connect(lambda _: _apply())
        tr_radio.toggled.connect(lambda _: _apply())

        dlg.exec()

    def _compute_flow_pos(self, frame_8bit_1, frame_8bit_2):
        """Run Farneback and return pos array (N*6, 2), or None if no vectors."""
        scale = 0.5
        small_1 = cv2.resize(frame_8bit_1, (0, 0), fx=scale, fy=scale)
        small_2 = cv2.resize(frame_8bit_2, (0, 0), fx=scale, fy=scale)
        flow = cv2.calcOpticalFlowFarneback(small_1, small_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        step = 16
        h, w = small_1.shape
        y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        x_orig = x / scale; y_orig = y / scale
        vis_mult = 3.0
        fx_orig = (fx / scale) * vis_mult; fy_orig = (fy / scale) * vis_mult
        mag = np.hypot(fx_orig, fy_orig)
        mask = mag > 1.0
        x_orig, y_orig = x_orig[mask], y_orig[mask]
        fx_orig, fy_orig = fx_orig[mask], fy_orig[mask]
        mag = mag[mask]
        N = len(x_orig)
        if N == 0:
            return None
        x2 = x_orig + fx_orig; y2 = y_orig + fy_orig
        u_norm = fx_orig / mag; v_norm = fy_orig / mag
        head_len = 3.0
        ux = u_norm * head_len; vy = v_norm * head_len
        b1_x = x2 - ux - vy; b1_y = y2 - vy + ux
        b2_x = x2 - ux + vy; b2_y = y2 - vy - ux
        pos = np.empty((N * 6, 2))
        pos[0::6, 0] = x_orig; pos[0::6, 1] = y_orig
        pos[1::6, 0] = x2;     pos[1::6, 1] = y2
        pos[2::6, 0] = x2;     pos[2::6, 1] = y2
        pos[3::6, 0] = b1_x;   pos[3::6, 1] = b1_y
        pos[4::6, 0] = x2;     pos[4::6, 1] = y2
        pos[5::6, 0] = b2_x;   pos[5::6, 1] = b2_y
        return pos

    def calculate_optical_flow(self, frame_8bit_1, frame_8bit_2):
        pos = self._compute_flow_pos(frame_8bit_1, frame_8bit_2)
        if pos is None:
            self.vector_field.setData(x=[], y=[])
        else:
            self.vector_field.setData(x=pos[:, 0], y=pos[:, 1])

    # ------------------------------------------------------------------
    # Core render path
    # ------------------------------------------------------------------

    def update_frame_display(self):
        if self.sequence_manager is None:
            return

        self.perf.begin_frame()
        try:
            with self.perf.span("get_frame"):
                base_raw_frame = self.sequence_manager.get_frame(self.current_frame)
            if base_raw_frame is None:
                return

            with self.perf.span("setup + cache_tags"):
                # Single pipeline walk — captures display frame, pixel-probe frame, and
                # histogram frame in one pass rather than three separate apply_to_frame calls.
                rot_idx, _ = self._find_op(RotateOp)
                bg_idx, _ = self._find_op(BgSubtractOp)
                disp_target = self._display_target_stage()

                # Invalidate OF display cache when pipeline state or disp_target changes.
                _of_tag = str(disp_target) + ":" + "|".join(
                    op.fingerprint() for op in self.pipeline.operations if op.enabled
                )
                if _of_tag != self._of_cache_tag:
                    self._of_display_cache.clear()
                    self._of_cache_tag = _of_tag
                # Invalidate Farneback result cache when pipeline OR display params change.
                _of_flow_tag = _of_tag + f":{self.vmin}:{self.vmax}:{self.gamma:.4f}"
                if _of_flow_tag != self._of_flow_tag:
                    self._of_flow_cache.clear()
                    self._of_flow_tag = _of_flow_tag

                n_ops = len(self.pipeline.operations)
                disp_excl = 0 if disp_target == -2 else (n_ops if disp_target == -1 else disp_target + 1)
                rot_excl  = rot_idx + 1 if rot_idx >= 0 else 0
                bg_excl   = bg_idx  + 1 if bg_idx  >= 0 else 0

                furthest_excl = max(disp_excl, rot_excl, bg_excl)
                if furthest_excl == 0:
                    walk_target = -2
                elif furthest_excl >= n_ops:
                    walk_target = -1
                else:
                    walk_target = furthest_excl - 1

                snap_set = set()
                snap_set.add(rot_idx if rot_idx >= 0 else -1)   # -1 = capture raw input
                if bg_idx >= 0:
                    snap_set.add(bg_idx)
                if 0 <= disp_target < walk_target:               # need mid-walk snapshot
                    snap_set.add(disp_target)

            pipeline_timings = {} if self.perf.enabled else None
            with self.perf.span("pipeline_walk"):
                final_frame, snapshots = self.pipeline.apply_with_snapshots(
                    base_raw_frame, self.current_frame,
                    snapshot_indices=snap_set,
                    final_target=walk_target,
                    timings=pipeline_timings,
                )
            if pipeline_timings:
                self.perf.merge(pipeline_timings)

            if disp_target == -2:
                display_frame = base_raw_frame
            elif disp_target < 0 or disp_target == walk_target:
                display_frame = final_frame
            else:
                display_frame = snapshots[disp_target]

            self.current_raw_frame = snapshots.get(
                -1 if rot_idx < 0 else rot_idx, base_raw_frame
            )
            self._last_frame_for_histogram = (
                snapshots.get(bg_idx, self.current_raw_frame) if bg_idx >= 0
                else self.current_raw_frame
            )

            # Cache current frame's display output so the OF block can reuse it as a partner.
            self._of_display_cache[self.current_frame] = display_frame
            if len(self._of_display_cache) > self._OF_CACHE_MAX:
                del self._of_display_cache[next(iter(self._of_display_cache))]

            # Rebuild LUT only when display params change (rare vs. per-frame scrub).
            _lut_key = (self.vmin, self.vmax, self.gamma)
            if self._display_lut is None or self._display_lut_key != _lut_key:
                self._display_lut = build_display_lut(self.vmin, self.vmax, self.gamma)
                self._display_lut_key = _lut_key

            with self.perf.span("scale_to_8bit"):
                frame_8bit = scale_16bit_to_8bit(display_frame, self.vmin, self.vmax, self.gamma, lut=self._display_lut)
            self._display_hw = display_frame.shape[:2]
            with self.perf.span("setImage (main)"):
                self.image_item.setImage(frame_8bit.T, autoLevels=False)
            with self.perf.span("block_grid"):
                self._update_block_grid()

            # threshold overlay
            with self.perf.span("threshold_overlay (total)"):
                thresh_idx, thresh_op = self._find_op(AdaptiveThresholdOp)
                if thresh_op is not None and thresh_op.enabled:
                    cached = self.pipeline.get_batch_cache(thresh_idx, self.current_frame)
                    if cached is not None:
                        mask = cached
                    else:
                        # display_frame already stops before AdaptiveThresholdOp
                        with self.perf.span("  AdaptiveThresholdOp (live)"):
                            mask = thresh_op.apply(display_frame)
                    # Apply any downstream binary-mask ops (morphology, smoothing) live.
                    # They're cheap per-frame; no batch cache needed.
                    # Ops with requires_intensity_context=True receive the upstream
                    # grayscale frame via context so they can use intensity peaks
                    # rather than the binary distance transform.
                    with self.perf.span("binary_chain (total)"):
                        for j in range(thresh_idx + 1, len(self.pipeline.operations)):
                            op_j = self.pipeline.operations[j]
                            if op_j.enabled and getattr(op_j, "is_binary_mask_op", False):
                                context = None
                                if getattr(op_j, "requires_intensity_context", False):
                                    src_idx = op_j.params.get("intensity_source_idx", -1)
                                    if src_idx < 0:
                                        # -1 = pre-threshold frame; same shape as mask.
                                        # display_frame is the frame fed into AdaptiveThresholdOp.
                                        intensity_f = display_frame
                                    else:
                                        with self.perf.span(f"  binary:{op_j.name} intensity_src"):
                                            intensity_f = self.pipeline.apply_to_frame(
                                                base_raw_frame, self.current_frame,
                                                target_stage_idx=src_idx,
                                            )
                                    context = {"intensity_frame": intensity_f}
                                with self.perf.span(f"  binary:[{j}] {op_j.name}"):
                                    mask = op_j.apply(mask, context=context)

                    # Build overlay — two colours when IntensityWatershedSplitOp ran:
                    #   orange = all non-split pixels (small blobs + large single-particle)
                    #   cyan   = pixels reconstructed as circles (split merged blobs)
                    # Falls back to all-orange when IntensityWatershedSplitOp is absent.
                    _ORANGE = np.array([255, 100,   0, 160], dtype=np.uint8)
                    _CYAN   = np.array([  0, 200, 255, 210], dtype=np.uint8)
                    iw_split = None
                    for _j in range(thresh_idx + 1, len(self.pipeline.operations)):
                        _op = self.pipeline.operations[_j]
                        if (isinstance(_op, IntensityWatershedSplitOp) and _op.enabled
                                and getattr(_op, "_last_split_pixels", None) is not None):
                            iw_split = _op._last_split_pixels
                    if (iw_split is not None and iw_split.shape == mask.shape
                            and (iw_split > 0).any()):
                        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
                        overlay[mask > 0] = _ORANGE          # all threshold pixels
                        overlay[iw_split > 0] = _CYAN        # overwrite split circles
                    else:
                        overlay = (mask >> 7)[:, :, np.newaxis] * _ORANGE
                    self.threshold_overlay.setImage(overlay.swapaxes(0, 1), autoLevels=False)
                else:
                    self.threshold_overlay.clear()

            # optical flow
            if self.optical_flow_enabled:
                with self.perf.span("optical_flow (total)"):
                    def _of_get(idx: int):
                        """Return pipeline display frame for idx, using display cache when available."""
                        fr = self._of_display_cache.get(idx)
                        if fr is not None:
                            return fr
                        with self.perf.span("  of_partner_get_frame"):
                            raw = self.sequence_manager.get_frame(idx)
                        if raw is None:
                            return None
                        with self.perf.span("  of_partner_pipeline_walk"):
                            fr = self.pipeline.apply_to_frame(raw, idx, target_stage_idx=disp_target)
                        self._of_display_cache[idx] = fr
                        if len(self._of_display_cache) > self._OF_CACHE_MAX:
                            del self._of_display_cache[next(iter(self._of_display_cache))]
                        return fr

                    def _of_draw(a_idx, a_8bit, b_idx, b_8bit):
                        """Draw flow for pair (a→b), using Farneback result cache when available."""
                        key = (a_idx, b_idx)
                        _MISS = object.__new__(object)  # sentinel distinguishing "not cached" from None
                        pos = self._of_flow_cache.get(key, _MISS)
                        if pos is _MISS:
                            with self.perf.span("  of_farneback"):
                                pos = self._compute_flow_pos(a_8bit, b_8bit)
                            self._of_flow_cache[key] = pos
                            if len(self._of_flow_cache) > self._OF_FLOW_CACHE_MAX:
                                del self._of_flow_cache[next(iter(self._of_flow_cache))]
                        with self.perf.span("  of_setData"):
                            if pos is None:
                                self.vector_field.setData(x=[], y=[])
                            else:
                                self.vector_field.setData(x=pos[:, 0], y=pos[:, 1])

                    flow_done = False
                    if self.optical_flow_mode == 'pairwise':
                        pair_a_idx = (self.current_frame // 2) * 2
                        pair_b_idx = pair_a_idx + 1
                        if pair_b_idx < self.num_frames:
                            if self.current_frame % 2 == 0:
                                b_disp = _of_get(pair_b_idx)
                                if b_disp is not None:
                                    with self.perf.span("  of_partner_to_8bit"):
                                        b_8bit = scale_16bit_to_8bit(b_disp, self.vmin, self.vmax, self.gamma, lut=self._display_lut)
                                    _of_draw(pair_a_idx, frame_8bit, pair_b_idx, b_8bit)
                                    flow_done = True
                            else:
                                a_disp = _of_get(pair_a_idx)
                                if a_disp is not None:
                                    with self.perf.span("  of_partner_to_8bit"):
                                        a_8bit = scale_16bit_to_8bit(a_disp, self.vmin, self.vmax, self.gamma, lut=self._display_lut)
                                    _of_draw(pair_a_idx, a_8bit, pair_b_idx, frame_8bit)
                                    flow_done = True
                    else:  # time_resolved
                        if self.current_frame < self.num_frames - 1:
                            next_disp = _of_get(self.current_frame + 1)
                            if next_disp is not None:
                                with self.perf.span("  of_partner_to_8bit"):
                                    next_8bit = scale_16bit_to_8bit(next_disp, self.vmin, self.vmax, self.gamma, lut=self._display_lut)
                                _of_draw(self.current_frame, frame_8bit, self.current_frame + 1, next_8bit)
                                flow_done = True
                    if not flow_done:
                        self.vector_field.setData(x=[], y=[])
            else:
                self.vector_field.setData(x=[], y=[])

            with self.perf.span("status_label"):
                self._refresh_status_label()
            with self.perf.span("pixel_info"):
                self.update_pixel_info()
            if self.tool_window.isVisible():
                with self.perf.span("histogram"):
                    self.tool_window.update_histogram(self._last_frame_for_histogram)
            if self.facet_sessions:
                # Only sessions belonging to this frame stay drawn.
                with self.perf.span("facet_overlay"):
                    self.refresh_facets()
        finally:
            self.perf.end_frame()
            if self.perf_window.isVisible():
                self.perf_window.refresh()

    # ------------------------------------------------------------------
    # Block grid overlay
    # ------------------------------------------------------------------

    def _update_block_grid(self):
        ops = self.pipeline.operations
        sel = self.pipeline_panel._selected_idx
        op = ops[sel] if 0 <= sel < len(ops) else None
        if not isinstance(op, AdaptiveThresholdOp) or self._display_hw is None:
            self.block_grid_overlay.setVisible(False)
            return
        H, W = self._display_hw
        block_size = op.params['block_size']
        xs, ys = [], []
        for x in range(block_size, W, block_size):
            xs += [x, x]
            ys += [0, H - 1]
        for y in range(block_size, H, block_size):
            xs += [0, W - 1]
            ys += [y, y]
        if xs:
            self.block_grid_overlay.setData(
                x=np.array(xs, dtype=float), y=np.array(ys, dtype=float)
            )
            self.block_grid_overlay.setVisible(True)
        else:
            self.block_grid_overlay.setVisible(False)

    # ------------------------------------------------------------------
    # Misc display actions
    # ------------------------------------------------------------------

    def toggle_perf_window(self):
        if self.perf_window.isVisible():
            self.perf_window.hide()
        else:
            self.perf_window.show()
            self.perf_window.raise_()
            self.perf_window.refresh()

    def _show_blob_size_window(self):
        self.blob_size_window.show()
        self.blob_size_window.raise_()

    def _show_region_props_window(self):
        self.region_props_window.show()
        self.region_props_window.raise_()

    def _show_mie_window(self):
        self.mie_window.show()
        self.mie_window.raise_()

    def toggle_tool_window(self):
        if self.tool_window.isVisible():
            self.tool_window.hide()
        else:
            self.tool_window.show()
            frame = self._last_frame_for_histogram if self._last_frame_for_histogram is not None else self.current_raw_frame
            if frame is not None:
                self.tool_window.update_histogram(frame)

    def _toggle_pipeline_panel(self):
        if self.pipeline_panel.isVisible():
            self.pipeline_panel.hide()
        else:
            self.pipeline_panel.show()

    def set_bit_depth(self, new_max):
        self.bit_depth_max = new_max
        self.vmax = min(self.vmax, new_max)
        self.vmin = max(0, min(self.vmin, new_max))
        if self.vmin >= self.vmax:
            self.vmin = 0
            self.vmax = new_max
        self.tool_window.sync_from_main()
        self.update_frame_display()

    def reset_view(self):
        if self.sequence_manager is not None:
            self.view_box.autoRange()

    def zoom_image(self, scale_factor):
        zoom_center = None
        if self.mouse_pos is not None:
            if self.view_box.sceneBoundingRect().contains(self.mouse_pos):
                zoom_center = self.view_box.mapSceneToView(self.mouse_pos)
        if zoom_center is None:
            zoom_center = self.view_box.viewRect().center()
        self.view_box.scaleBy(x=scale_factor, y=scale_factor, center=zoom_center)

    # ------------------------------------------------------------------
    # File open / load
    # ------------------------------------------------------------------

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Starting TIFF File", "", "TIFF Images (*.tif *.tiff *.TIF *.TIFF)"
        )
        if file_path:
            self.status_label.setText("Caching local window into RAM, please wait...")
            QApplication.processEvents()
            self.load_and_display_data(file_path)

    def load_and_display_data(self, full_file_path):
        folder_path = os.path.dirname(full_file_path)
        target_filename = os.path.basename(full_file_path)

        try:
            self.sequence_manager = LazyTiffSequence(folder_path, buffer_radius=20)
            self.num_frames = self.sequence_manager.num_frames
            self.current_frame = self.sequence_manager.get_index_from_filename(target_filename)
            self.clear_all_rays()
            self.clear_all_facet_sessions()
            if self.measure_mode:
                self._exit_measure_mode(committed_px=None)
            self._of_display_cache.clear()
            self._of_cache_tag = ""
            self._of_flow_cache.clear()
            self._of_flow_tag = ""
            # reset pipeline
            self.pipeline = Pipeline()
            self._invalidate_threshold_caches()
            self.pipeline_panel.rebuild_list()
            self.threshold_overlay.clear()
            self.sequence_manager.prefetch_window(self.current_frame)

            first = self.sequence_manager.get_frame(self.current_frame)
            if first is not None:
                if first.dtype == np.uint8:
                    self.bit_depth_max = 255
                else:
                    self.bit_depth_max = 4095
                self.vmin = 0
                self.vmax = self.bit_depth_max
                target = str(self.bit_depth_max)
                for action in self.bit_group.actions():
                    action.setChecked(target in action.text())
                self.tool_window.sync_from_main()

            self.update_frame_display()
            self.view_box.autoRange()

        except Exception as e:
            self.status_label.setText(f"Error loading sequence: {str(e)}")

    # ------------------------------------------------------------------
    # Mouse / pixel probe
    # ------------------------------------------------------------------

    def on_mouse_moved(self, pos):
        self.mouse_pos = pos
        self.update_pixel_info()
        if self.measure_mode == 'await_second':
            if self.view_box.sceneBoundingRect().contains(pos):
                img_pos = self.view_box.mapSceneToView(pos)
                shift = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
                self._measure_update_endpoint(img_pos, shift)

    def update_pixel_info(self):
        if self.current_raw_frame is None or self.mouse_pos is None:
            self.pixel_info_label.setText("")
            return

        if not self.view_box.sceneBoundingRect().contains(self.mouse_pos):
            self.pixel_info_label.setText("")
            return

        mouse_point = self.view_box.mapSceneToView(self.mouse_pos)
        x, y = int(mouse_point.x()), int(mouse_point.y())

        cr = self._crop_region()
        crop_op_enabled = cr is not None and not self.crop_mode
        if crop_op_enabled:
            cx, cy, cw, ch = cr
            disp_w, disp_h = cw, ch
            probe_x, probe_y = x + cx, y + cy
        else:
            disp_h, disp_w = self.current_raw_frame.shape
            probe_x, probe_y = x, y

        if 0 <= x < disp_w and 0 <= y < disp_h:
            raw_intensity = self.current_raw_frame[probe_y, probe_x]
            if np.issubdtype(self.current_raw_frame.dtype, np.floating):
                val_str = f"{raw_intensity:.2f}"
            else:
                val_str = f"{int(raw_intensity):5d}"
            self.pixel_info_label.setText(f"X: {x:4d}  |  Y: {y:4d}  |  Raw Value: {val_str}")
        else:
            self.pixel_info_label.setText("")

    # ------------------------------------------------------------------
    # Key events
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        if self.measure_mode:
            if event.key() == Qt.Key.Key_Escape:
                self._exit_measure_mode(committed_px=None)
            return

        if self.ray_mode is not None:
            if event.key() == Qt.Key.Key_Escape:
                self.ray_mode = None
                self.glw.setCursor(Qt.CursorShape.ArrowCursor)
                self.status_label.setText(f"Frame: {self.current_frame + 1} / {self.num_frames}")
            return

        if self.facet_mode is not None:
            if event.key() == Qt.Key.Key_Escape:
                self._exit_facet_mode()
                self.refresh_facets()
                self._refresh_status_label()
                return

        if self.crop_mode:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._confirm_crop()
                return
            elif event.key() == Qt.Key.Key_Escape:
                # restore snapshot
                snap = self._crop_region_snapshot
                idx, op = self._find_op(CropOp)
                if snap is None:
                    if op is not None:
                        op.enabled = False
                else:
                    if op is None:
                        op = CropOp()
                        self.pipeline.add_operation(op)
                    op.params.update({"x": snap[0], "y": snap[1], "w": snap[2], "h": snap[3]})
                    op.enabled = True
                self._exit_crop_mode()
                self._sync_crop_params()
                self.view_box.autoRange()
                return

        if self.sequence_manager is None:
            return

        if event.key() == Qt.Key.Key_Right:
            if self.current_frame < self.num_frames - 1:
                self.current_frame += 1
                self.update_frame_display()
        elif event.key() == Qt.Key.Key_Left:
            if self.current_frame > 0:
                self.current_frame -= 1
                self.update_frame_display()

    # ------------------------------------------------------------------
    # Event filter (crop draw + ray placement)
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if obj is not self.glw.viewport():
            return False

        etype = event.type()

        if self.measure_mode:
            if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                scene_pos = self.glw.mapToScene(event.pos())
                img_pos = self.view_box.mapSceneToView(scene_pos)
                shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                if self.measure_mode == 'await_first':
                    self._measure_start_first(img_pos)
                else:
                    self._measure_commit_second(img_pos, shift)
                return True
            return False

        if self.ray_mode is not None:
            if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                scene_pos = self.glw.mapToScene(event.pos())
                img_pos = self.view_box.mapSceneToView(scene_pos)
                self._place_ray(img_pos)
                return True
            return False

        # Facet points. Dragging an existing point works whenever the panel is
        # open; placing a new one requires facet_mode. Events are consumed only
        # when a point is actually hit or placed, so crop mode is untouched.
        if self.facet_panel.isVisible() or self.facet_mode is not None:
            if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                scene_pos = self.glw.mapToScene(event.pos())
                img_pos = self.view_box.mapSceneToView(scene_pos)
                if self._facet_on_press(img_pos):
                    return True
            elif etype == QEvent.Type.MouseMove and self._facet_drag is not None:
                scene_pos = self.glw.mapToScene(event.pos())
                img_pos = self.view_box.mapSceneToView(scene_pos)
                if self._facet_on_drag(img_pos):
                    return True
            elif etype == QEvent.Type.MouseButtonRelease and self._facet_drag is not None:
                if self._facet_on_release():
                    return True

        if not self.crop_mode:
            return False

        if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.glw.mapToScene(event.pos())
            if self.crop_roi_item is not None and not self._drawing_crop:
                handle_items = {h['item'] for h in self.crop_roi_item.handles}
                items_at = set(self.glw.scene().items(scene_pos))
                if self.crop_roi_item in items_at or handle_items & items_at:
                    return False
            img_pos = self.view_box.mapSceneToView(scene_pos)
            self._start_crop_draw(img_pos)
            return True

        elif etype == QEvent.Type.MouseMove and self._drawing_crop:
            scene_pos = self.glw.mapToScene(event.pos())
            img_pos = self.view_box.mapSceneToView(scene_pos)
            self._update_crop_draw(img_pos)
            return True

        elif etype == QEvent.Type.MouseButtonRelease and self._drawing_crop:
            scene_pos = self.glw.mapToScene(event.pos())
            img_pos = self.view_box.mapSceneToView(scene_pos)
            self._end_crop_draw(img_pos)
            return True

        return False

    # ------------------------------------------------------------------
    # ROI draw helpers (unchanged logic)
    # ------------------------------------------------------------------

    def _create_interactive_roi(self, x, y, w, h):
        if self.crop_roi_item is not None:
            self.view_box.removeItem(self.crop_roi_item)
        bounds = None
        if self.current_raw_frame is not None:
            img_h, img_w = self.current_raw_frame.shape
            bounds = QRectF(0, 0, img_w, img_h)
        self.crop_roi_item = pg.RectROI(
            [x, y], [w, h],
            pen=pg.mkPen(color=(255, 220, 0), width=2, style=Qt.PenStyle.DashLine),
            handlePen=pg.mkPen(color=(255, 220, 0), width=2),
            handleHoverPen=pg.mkPen(color=(255, 255, 255), width=3),
            maxBounds=bounds,
        )
        self.crop_roi_item.addScaleHandle([0, 0], [1, 1])
        self.crop_roi_item.addScaleHandle([0, 1], [1, 0])
        self.crop_roi_item.addScaleHandle([1, 0], [0, 1])
        self.crop_roi_item.sigRegionChanged.connect(self._on_roi_adjusted)
        self.view_box.addItem(self.crop_roi_item)

    def _start_crop_draw(self, img_pos):
        self._drawing_crop = True
        self._crop_draw_start = (img_pos.x(), img_pos.y())
        if self.crop_roi_item is not None:
            self.view_box.removeItem(self.crop_roi_item)
        self.crop_roi_item = pg.ROI(
            [img_pos.x(), img_pos.y()], [0.1, 0.1],
            pen=pg.mkPen(color=(255, 220, 0), width=2, style=Qt.PenStyle.DashLine),
            movable=False, resizable=False, rotatable=False,
        )
        self.view_box.addItem(self.crop_roi_item)

    def _update_crop_draw(self, img_pos):
        if self.crop_roi_item is None:
            return
        sx, sy = self._crop_draw_start
        ex, ey = img_pos.x(), img_pos.y()
        if self.current_raw_frame is not None:
            img_h, img_w = self.current_raw_frame.shape
            ex = max(0.0, min(ex, float(img_w)))
            ey = max(0.0, min(ey, float(img_h)))
        x, y = min(sx, ex), min(sy, ey)
        w, h = abs(ex - sx), abs(ey - sy)
        self.crop_roi_item.setPos([x, y])
        self.crop_roi_item.setSize([max(w, 0.1), max(h, 0.1)])

    def _end_crop_draw(self, img_pos):
        self._drawing_crop = False
        sx, sy = self._crop_draw_start
        ex, ey = img_pos.x(), img_pos.y()
        if self.current_raw_frame is not None:
            img_h, img_w = self.current_raw_frame.shape
            ex = max(0.0, min(ex, float(img_w)))
            ey = max(0.0, min(ey, float(img_h)))
        x = int(round(min(sx, ex))); y = int(round(min(sy, ey)))
        w = int(round(abs(ex - sx))); h = int(round(abs(ey - sy)))
        if self.current_raw_frame is not None:
            img_h, img_w = self.current_raw_frame.shape
            x = max(0, min(x, img_w - 1)); y = max(0, min(y, img_h - 1))
            w = max(1, min(w, img_w - x)); h = max(1, min(h, img_h - y))

        idx, op = self._find_op(CropOp)
        is_new = op is None
        if op is None:
            op = CropOp()
            self.pipeline.add_operation(op)
        op.params.update({"x": x, "y": y, "w": w, "h": h})
        op.enabled = True
        self.pipeline._invalidate_from(self.pipeline.operations.index(op))
        self._create_interactive_roi(x, y, w, h)
        if is_new:
            self.pipeline_panel.rebuild_list()
        crop_idx, _ = self._find_op(CropOp)
        if crop_idx >= 0:
            self.pipeline_panel._selected_idx = crop_idx
            self.pipeline_panel.list_widget.setCurrentRow(crop_idx)
        self._sync_crop_params()
        self.status_label.setText(
            f"ROI: x={x}, y={y}, w={w}, h={h} — drag handles to adjust. Enter=apply, Esc=cancel, click image to redraw."
        )

    def _on_roi_adjusted(self):
        if self.crop_roi_item is None:
            return
        pos = self.crop_roi_item.pos()
        size = self.crop_roi_item.size()
        x = int(round(pos.x())); y = int(round(pos.y()))
        w = int(round(size.x())); h = int(round(size.y()))
        idx, op = self._find_op(CropOp)
        if op is None:
            op = CropOp()
            self.pipeline.add_operation(op)
            self.pipeline_panel.rebuild_list()
        op.params.update({"x": x, "y": y, "w": w, "h": h})
        op.enabled = True
        self._sync_crop_params()
        self.status_label.setText(
            f"ROI: x={x}, y={y}, w={w}, h={h} — drag handles to adjust. Enter=apply, Esc=cancel, click image to redraw."
        )

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self.tool_window.close()
        self.pipeline_panel.close()
        self.facet_panel.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Stylesheet (unchanged)
# ---------------------------------------------------------------------------

modern_stylesheet = """
QWidget {
    background-color: #1e1e1e;
    color: #ececec;
    font-family: "Helvetica Neue", Helvetica, "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QMenuBar {
    background-color: #252526;
    border-bottom: 1px solid #333333;
}
QMenuBar::item:selected { background: #3e3e42; }
QMenu { background-color: #252526; border: 1px solid #333333; }
QMenu::item:selected { background-color: #007acc; color: white; }
QSlider::groove:horizontal {
    border-radius: 4px; height: 6px; background: #333333;
}
QSlider::add-page:horizontal { background: #333333; border-radius: 4px; }
QSlider::sub-page:horizontal { background: #00c8ff; border-radius: 4px; }
QSlider::handle:horizontal {
    background: #ffffff; width: 14px; height: 14px;
    margin: -4px 0; border-radius: 7px; border: 1px solid #1e1e1e;
}
QSlider::handle:horizontal:hover {
    background: #00c8ff; width: 16px; height: 16px;
    margin: -5px 0; border-radius: 8px;
}
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(modern_stylesheet)
    window = TiffViewerApp()
    window.show()
    sys.exit(app.exec())
