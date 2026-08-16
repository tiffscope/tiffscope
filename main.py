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
    QSizePolicy, QRadioButton, QButtonGroup, QFrame, QGridLayout, QProgressBar,
)
from PyQt6.QtCore import Qt, QRectF, QThread, pyqtSignal, QEventLoop
from PyQt6.QtGui import QShortcut, QKeySequence, QAction, QActionGroup, QColor, QIntValidator, QDoubleValidator

from image_engine import LazyTiffSequence, scale_16bit_to_8bit, build_display_lut
from operations import (
    Operation, RotateOp, CropOp, BgSubtractOp, RollingBallBgOp, AdaptiveThresholdOp,
    GlobalThresholdOp,
    CLAHEOp, GaussianBlurOp, SharpenOp, LowPassOp, HighPassOp,
    MorphologyOp, BinarySmoothOp, BlobSizeFilterOp, WatershedSplitOp,
    IntensityWatershedSplitOp,
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

    def __init__(self, op: Operation, pipeline: Pipeline, op_idx: int,
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


class PlotPlaceholder:
    """Centred hint shown on an analysis plot until it has data.

    Every analysis window opens on an empty plot whose auto-range invents a
    0.1–0.9 tick grid — it reads as a broken chart rather than an idle one.
    Hiding the axes and stating what to do instead makes the idle state honest.
    """

    def __init__(self, plot_widget: pg.PlotWidget, text: str):
        self._plot = plot_widget
        self._item = pg.TextItem(text, color=(138, 138, 138), anchor=(0.5, 0.5))
        self._attach()
        self._shown = True
        self._apply(True)

    def _attach(self):
        """(Re)add the hint item. PlotWidget.clear() drops every item, so a
        window that clears before each redraw would otherwise lose it for good."""
        if self._item not in self._plot.getPlotItem().items:
            self._plot.addItem(self._item)
        self._item.setPos(0.5, 0.5)

    def _apply(self, showing: bool):
        pi = self._plot.getPlotItem()
        pi.showAxis('bottom', not showing)
        pi.showAxis('left', not showing)
        pi.showGrid(x=not showing, y=not showing, alpha=0.3)
        if showing:
            # Pin the view so the hint sits dead centre.
            self._plot.setXRange(0, 1, padding=0)
            self._plot.setYRange(0, 1, padding=0)
        else:
            # Hand the view back to autorange. Without this the 0-1 window above
            # survives into the real plot and the data draws off-screen.
            pi.enableAutoRange(axis='xy', enable=True)

    def set_text(self, text: str):
        self._item.setText(text)

    def hide(self):
        """Call once real data is plotted. Idempotent."""
        if not self._shown:
            return
        self._shown = False
        self._item.setVisible(False)
        self._apply(False)

    def show(self, text: str | None = None):
        if text is not None:
            self._item.setText(text)
        self._shown = True
        self._attach()
        self._item.setVisible(True)
        self._apply(True)


def format_stats_grid(pairs, columns: int = 4, notes=None) -> str:
    """Render (label, value) pairs as an aligned HTML grid for a QLabel.

    The stats bars were single run-on strings joined by " | ", which wrapped
    mid-item at any narrow window width ("Max: 165" landing alone on line 2)
    and gave nothing to scan down. A table keeps labels and values aligned in
    columns and wraps at cell boundaries.

    `notes` are appended as full-width lines under the grid (scale reminders and
    similar), which is where they read as context rather than as another stat.
    """
    cells = []
    for label, value in pairs:
        cells.append(
            f'<td style="padding:0 14px 2px 0; white-space:nowrap;">'
            f'<span style="color:#8a8a8a;">{label}</span>&nbsp;'
            f'<span style="color:#ececec; font-weight:600;">{value}</span></td>'
        )
    rows = []
    for i in range(0, len(cells), columns):
        rows.append("<tr>" + "".join(cells[i:i + columns]) + "</tr>")
    html = f'<table style="border-collapse:collapse;">{"".join(rows)}</table>'
    for note in (notes or []):
        html += f'<div style="color:#8a8a8a; padding-top:3px;">{note}</div>'
    return html


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
        # The header is a single long line; without wrapping the trailing frame
        # count is clipped at the default window width.
        self.header_label.setWordWrap(True)

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

        # --- Stage row ---
        # Which mask the sizes are measured on. Without this the window silently
        # measured the full binary chain, which became genuinely ambiguous once
        # BlobSizeFilterOp existed — "blob sizes" before or after filtering?
        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel("Binary stage:"))
        self.stage_combo = QComboBox()
        self.stage_combo.setMinimumWidth(220)
        self.stage_combo.setToolTip(
            "Mask stage the sizes are measured on.\n"
            "IntensityWatershedSplitOp is never included — this window exists to "
            "pick its min_blob_area, so it must show the mask that op receives."
        )
        stage_row.addWidget(self.stage_combo)
        stage_row.addStretch()
        layout.addLayout(stage_row)

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
        self._placeholder = PlotPlaceholder(
            self.plot_widget, "Run Analysis to collect blob areas")
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

    def _chain_candidates(self):
        """Enabled binary-chain ops eligible for this window.

        IntensityWatershedSplitOp is always excluded: this window's job is to
        pick that op's min_blob_area, so it must report the mask going *into*
        it, never its output.
        """
        app = self.main_app
        thresh_idx, thresh_op = app._find_threshold_op()
        if thresh_op is None:
            return -1, None, []
        ops = [
            op for op in app.pipeline.operations[thresh_idx + 1:]
            if op.enabled
            and getattr(op, "is_binary_mask_op", False)
            and not isinstance(op, IntensityWatershedSplitOp)
        ]
        return thresh_idx, thresh_op, ops

    def _rebuild_stage_combo(self):
        app = self.main_app
        self.stage_combo.blockSignals(True)
        prev = self.stage_combo.currentText()
        self.stage_combo.clear()
        _, thresh_op, ops = self._chain_candidates()
        if thresh_op is None:
            self.stage_combo.addItem("(No threshold op)")
            self.stage_combo.blockSignals(False)
            return
        self.stage_combo.addItem(f"{type(thresh_op).display()} output")
        for op in ops:
            self.stage_combo.addItem(f"After {type(op).display()}")
        idx = self.stage_combo.findText(prev)
        # Default to the end of the eligible chain — the mask
        # IntensityWatershedSplitOp actually receives.
        self.stage_combo.setCurrentIndex(idx if idx >= 0 else self.stage_combo.count() - 1)
        self.stage_combo.blockSignals(False)

    def _binary_ops_for_selected_stage(self):
        _, thresh_op, ops = self._chain_candidates()
        if thresh_op is None:
            return []
        text = self.stage_combo.currentText()
        if text == f"{type(thresh_op).display()} output":
            return []
        out = []
        for op in ops:
            out.append(op)
            if text == f"After {type(op).display()}":
                break
        return out

    def _stage_description(self) -> str:
        text = self.stage_combo.currentText()
        n = len(self._binary_ops_for_selected_stage())
        if n == 0:
            return f"{text} (no binary-chain ops applied)"
        return f"{text} ({n} binary-chain op{'s' if n != 1 else ''} applied)"

    def _run(self):
        app = self.main_app
        if app.sequence_manager is None:
            self.stats_label.setText("No sequence loaded.")
            return

        thresh_idx, thresh_op, _ = self._chain_candidates()
        if thresh_op is None or not thresh_op.enabled:
            self.stats_label.setText("No enabled threshold op in pipeline.")
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
        self._placeholder.hide()

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
        # Frame on the data. pyqtgraph's autorange pads generously, leaving a
        # long empty tail (axis to 175 px² for data ending near 90).
        self.plot_widget.setXRange(float(edges[0]), float(edges[-1]), padding=0.02)

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
        pairs = [
            ("Blobs", f"{len(arr):,}"),
            ("Median", f"{median:.0f} px²"),
            ("95th pct", f"{p95:.0f} px²"),
            ("99th pct", f"{p99:.0f} px²"),
            ("Max", f"{arr.max():.0f} px²"),
        ]
        notes = [f"Measured at: {self._stage_description()}"]
        if iw_op is not None and iw_op.enabled:
            mba = iw_op.params.get("min_blob_area", 50)
            n_above = int((arr >= mba).sum())
            pairs.append((f"≥ min_blob_area ({mba} px²)", f"{n_above:,} blobs"))
        self.stats_label.setText(format_stats_grid(pairs, columns=3, notes=notes))

    def showEvent(self, event):
        super().showEvent(event)
        # Pipeline may have changed while hidden — restock the stage list first.
        self._rebuild_stage_combo()
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
        self.plot_widget.setLabel('bottom', 'Area (px²)')
        self.plot_widget.setLabel('left', 'Count')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._bar_item: pg.BarGraphItem | None = None
        self._placeholder = PlotPlaceholder(
            self.plot_widget, "Run Analysis to measure region properties")
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

        thresh_idx, thresh_op = app._find_threshold_op()
        if thresh_op is None:
            self.stage_combo.addItem("(No threshold op)")
            self.stage_combo.blockSignals(False)
            return

        self.stage_combo.addItem(f"{type(thresh_op).display()} output")
        for op in app.pipeline.operations[thresh_idx + 1:]:
            if op.enabled and getattr(op, 'is_binary_mask_op', False):
                self.stage_combo.addItem(f"After {type(op).display()}")

        idx = self.stage_combo.findText(prev_text)
        if idx >= 0:
            self.stage_combo.setCurrentIndex(idx)
        else:
            # Default to the END of the binary chain, not the bare threshold
            # output. Mie sizing consumes the full chain, so defaulting to the
            # threshold stage made the two windows report different numbers for
            # the same data with nothing on screen to explain the difference.
            self.stage_combo.setCurrentIndex(self.stage_combo.count() - 1)
        self.stage_combo.blockSignals(False)

    def _stage_description(self) -> str:
        """Human sentence naming the mask stage these numbers came from."""
        text = self.stage_combo.currentText()
        n_ops = len(self._binary_ops_for_selected_stage())
        if n_ops == 0:
            return f"{text} (no binary-chain ops applied)"
        return f"{text} ({n_ops} binary-chain op{'s' if n_ops != 1 else ''} applied)"

    def _binary_ops_for_selected_stage(self):
        """Return binary chain ops to apply up to and including selected stage."""
        app = self.main_app
        thresh_idx, thresh_op = app._find_threshold_op()
        if thresh_op is None:
            return []
        stage_text = self.stage_combo.currentText()
        if stage_text == f"{type(thresh_op).display()} output":
            return []
        ops = []
        for op in app.pipeline.operations[thresh_idx + 1:]:
            if op.enabled and getattr(op, 'is_binary_mask_op', False):
                ops.append(op)
                if stage_text == f"After {type(op).display()}":
                    break
        return ops

    def _run(self):
        app = self.main_app
        if app.sequence_manager is None:
            self.stats_label.setText("No sequence loaded.")
            return

        thresh_idx, thresh_op = app._find_threshold_op()
        if thresh_op is None or not thresh_op.enabled:
            self.stats_label.setText("No enabled threshold op in pipeline.")
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
        self._placeholder.hide()

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
        # See BlobSizeAnalysisWindow._redraw — frame on the data, not on
        # pyqtgraph's padded autorange.
        self.plot_widget.setXRange(float(edges[0]), float(edges[-1]), padding=0.02)

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

        notes = [f"Measured at: {self._stage_description()}"]
        if dim_power == 0:
            # Dimensionless (eccentricity) — single-unit display
            pairs = [
                ("Blobs", f"{n_blobs:,}"), ("Mean", _fmtv(mean_v)),
                ("Median", _fmtv(median_v)), ("Std", _fmtv(std_v)),
                ("P95", _fmtv(p95_v)), ("P99", _fmtv(p99_v)),
                ("Max", _fmtv(max_v)),
            ]
            stats_text = format_stats_grid(pairs, columns=4, notes=notes)
        elif factor is not None:
            def _fmtu(v):
                return f"{v * factor:.3g}"
            pairs = [("Blobs", f"{n_blobs:,}")] + [
                (lbl, f"{_fmtv(v)} {px_unit} <span style='color:#8ab4f8;'>"
                      f"({_fmtu(v)} {unit_label})</span>")
                for lbl, v in (("Mean", mean_v), ("Median", median_v), ("Std", std_v),
                               ("P95", p95_v), ("P99", p99_v), ("Max", max_v))
            ]
            scale = self.main_app.pipeline.scale
            notes.append(
                f"Scale: {scale['px']} px = {_fmt_num(scale['value'])} {scale['unit']}"
                f" &nbsp;→&nbsp; 1 px = {_fmt_num(scale['value'] / scale['px'])} {scale['unit']}"
            )
            stats_text = format_stats_grid(pairs, columns=3, notes=notes)
        else:
            pairs = [("Blobs", f"{n_blobs:,}")] + [
                (lbl, f"{_fmtv(v)} {px_unit}")
                for lbl, v in (("Mean", mean_v), ("Median", median_v), ("Std", std_v),
                               ("P95", p95_v), ("P99", p99_v), ("Max", max_v))
            ]
            notes.append("Set a scale via Tools → Set Scale… (Ctrl+M) to see physical units.")
            stats_text = format_stats_grid(pairs, columns=4, notes=notes)

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
    n_max) and can take minutes for large particles / fine grids, so it must
    never block the UI.

    Reports determinate progress and supports cooperative cancellation: the Mie
    order loop calls back once per order (see mie.csca_curve's progress_cb), and
    returning False there aborts the sweep. QThread.terminate() is never used —
    killing a thread mid-numpy is a good way to corrupt the interpreter.
    """
    finished = pyqtSignal(object)   # mie.MieResult
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(int, int)   # orders done, orders total

    def __init__(self, areas_m2, n_real, n_imag, a_start, a_stop,
                 lam_nm, n_medium, mu, num_a, inversion):
        super().__init__()
        self._args = (areas_m2, n_real, n_imag, a_start, a_stop)
        self._kw = dict(lam_nm=lam_nm, n_medium=n_medium, mu=mu, num_a=num_a,
                        inversion=inversion)
        self._abort = False

    def cancel(self):
        self._abort = True

    def run(self):
        try:
            import mie
            res = mie.analyze(*self._args, progress_cb=self._on_order, **self._kw)
            self.finished.emit(res)
        except Exception as e:
            import mie
            if isinstance(e, mie.MieCancelled):
                self.cancelled.emit()
                return
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _on_order(self, done, total):
        self.progress.emit(done, total)
        return not self._abort


class _StatCard(QFrame):
    """Big-number readout for a headline result (D10/D50/D90 …).

    The D-values are what the whole window exists to produce; before this they
    were three numbers inside a wrapped monospace paragraph at the bottom.
    """

    def __init__(self, title: str, accent: str = "#8ab4f8"):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ background:#252526; border:1px solid #3a3a3d;"
            f" border-radius:6px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 8)
        lay.setSpacing(1)
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(
            f"color:{accent}; font-size:11px; font-weight:600; border:none;")
        self.value_lbl = QLabel("—")
        self.value_lbl.setStyleSheet(
            "color:#ececec; font-size:20px; font-weight:600; border:none;")
        self.sub_lbl = QLabel("")
        self.sub_lbl.setStyleSheet("color:#8a8a8a; font-size:10px; border:none;")
        for w in (self.title_lbl, self.value_lbl, self.sub_lbl):
            lay.addWidget(w)

    def set_value(self, value: str, sub: str = ""):
        self.value_lbl.setText(value)
        self.sub_lbl.setText(sub)


class MieAnalysisWindow(QWidget):
    """Mie-scattering particle sizing.

    Treats each measured blob area (in m²) as that particle's scattering
    cross-section C_sca, inverts a forward Mie curve to per-particle radii,
    doubles them to diameters, and fits a Rosin-Rammler CDF for D10/D50/D90.
    See mie.py.

    D10/D50/D90 are DIAMETERS. The C_sca plot stays a function of radius (the
    Mie size parameter is x = k*a); the lower plot and every D-marker are in
    diameter. Both axes say which.

    Method from:
      Dasgupta, Raut, Vadukut, Bose, "Design and Development of a Pneumatic
      Atomizer for Seeding Tracers in PIV Experiments", J. Flow Visualization
      and Image Processing 32(3), 2025.

    Areas come from either the in-memory Region Props path (px² → m² via the
    Ctrl+M pixel scale) or a loaded CSV whose areas are already physical.

    Layout is a fixed-width control column beside a plot column: the controls
    are a numbered three-step sequence (areas → optics → sweep), so they read
    top-to-bottom instead of as three crammed rows of unlabelled fields.
    """

    # Radii are entered and displayed in µm. Metres are the storage unit
    # (mie.py is SI throughout) but nobody sizing aerosols thinks in metres —
    # typing 4.1924e-07 to mean "0.42 µm" is an invitation to slip a decade.
    UM = 1e-6

    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.setWindowTitle("Mie Particle Sizing")
        self.setWindowFlags(Qt.WindowType.Tool)
        self.resize(1120, 760)
        self._worker: RegionPropsWorker | None = None
        self._mie_worker: MieComputeWorker | None = None
        self._areas_m2: np.ndarray | None = None   # particle areas (== C_sca), m²
        self._areas_source: str = ""               # human-readable provenance
        self._result = None                         # mie.MieResult

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ============================ left: controls ======================
        ctrl_col = QVBoxLayout()
        ctrl_col.setSpacing(8)
        ctrl_host = QWidget()
        ctrl_host.setLayout(ctrl_col)
        ctrl_host.setFixedWidth(330)
        root.addWidget(ctrl_host)

        # --- 1. area source ---
        src_box = QGroupBox("1 · Particle areas")
        src_form = QFormLayout(src_box)
        src_form.setContentsMargins(10, 6, 10, 8)
        src_form.setSpacing(6)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Region Props (in-memory)", "CSV file"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        src_form.addRow("Source:", self.source_combo)

        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 99999)
        self.spin_n.setValue(200)
        self.all_check = QCheckBox("All frames")
        self.all_check.toggled.connect(lambda c: self.spin_n.setEnabled(not c))
        frames_row = QHBoxLayout()
        frames_row.setContentsMargins(0, 0, 0, 0)
        frames_row.addWidget(self.spin_n)
        frames_row.addWidget(self.all_check)
        self.frames_row_host = QWidget()
        self.frames_row_host.setLayout(frames_row)
        self.sample_lbl = QLabel("Sample frames:")
        src_form.addRow(self.sample_lbl, self.frames_row_host)

        self.load_csv_btn = QPushButton("Load CSV…")
        self.load_csv_btn.clicked.connect(self._load_csv)
        self.csv_unit_combo = QComboBox()
        self.csv_unit_combo.addItems(["mm", "µm", "m", "cm", "nm"])
        self.csv_unit_lbl = QLabel("CSV area unit:")
        self.csv_file_lbl = QLabel("CSV file:")
        src_form.addRow(self.csv_file_lbl, self.load_csv_btn)
        src_form.addRow(self.csv_unit_lbl, self.csv_unit_combo)

        self.areas_lbl = QLabel("no areas loaded")
        self.areas_lbl.setStyleSheet("color:#8a8a8a; font-size:11px;")
        self.areas_lbl.setWordWrap(True)
        src_form.addRow(self.areas_lbl)
        ctrl_col.addWidget(src_box)

        # --- 2. optics ---
        opt_box = QGroupBox("2 · Optics")
        opt_form = QFormLayout(opt_box)
        opt_form.setContentsMargins(10, 6, 10, 8)
        opt_form.setSpacing(6)
        self.n_real = QLineEdit("1.5")
        self.n_real.setValidator(QDoubleValidator())
        self.n_imag = QLineEdit("0.0")
        self.n_imag.setValidator(QDoubleValidator())
        n_row = QHBoxLayout()
        n_row.setContentsMargins(0, 0, 0, 0)
        n_row.addWidget(self.n_real)
        n_row.addWidget(QLabel("+"))
        n_row.addWidget(self.n_imag)
        n_row.addWidget(QLabel("i"))
        n_host = QWidget()
        n_host.setLayout(n_row)
        n_host.setToolTip(
            "Complex refractive index of the particle.\n"
            "Water 1.33, NaCl 1.544, glass ~1.5. Imaginary part = absorption."
        )
        opt_form.addRow("n particle:", n_host)

        self.lam = QLineEdit("527")
        self.lam.setValidator(QDoubleValidator(1.0, 1e5, 3))
        opt_form.addRow("λ (nm):", self.lam)
        self.n_medium = QLineEdit("1.0")
        self.n_medium.setValidator(QDoubleValidator(1e-3, 10.0, 4))
        opt_form.addRow("n medium:", self.n_medium)
        self.mu = QLineEdit("1.0")
        self.mu.setValidator(QDoubleValidator())
        self.mu.setToolTip("Permeability ratio sphere/medium. Almost always 1.")
        opt_form.addRow("μ:", self.mu)
        ctrl_col.addWidget(opt_box)

        # --- 3. radius sweep ---
        rad_box = QGroupBox("3 · Radius sweep")
        rad_form = QFormLayout(rad_box)
        rad_form.setContentsMargins(10, 6, 10, 8)
        rad_form.setSpacing(6)
        self.a_start = QLineEdit("")
        self.a_start.setValidator(QDoubleValidator(0.0, 1e9, 6))
        self.a_start.setPlaceholderText("press Auto range")
        self.a_stop = QLineEdit("")
        self.a_stop.setValidator(QDoubleValidator(0.0, 1e9, 6))
        self.a_stop.setPlaceholderText("press Auto range")
        rad_form.addRow("start (µm):", self.a_start)
        rad_form.addRow("stop (µm):", self.a_stop)
        self.auto_btn = QPushButton("Auto range from areas")
        self.auto_btn.setToolTip(
            "Seed the sweep from the measured areas (geometric estimate a≈√(A/π)).\n"
            "A tight bracket matters: runtime grows with the largest radius."
        )
        self.auto_btn.clicked.connect(self._auto_range)
        rad_form.addRow(self.auto_btn)

        self.num_a = QLineEdit("50000")
        self.num_a.setValidator(QIntValidator(100, 500000))
        self.num_a.setToolTip(
            "How many radii the forward Mie curve is evaluated at.\n"
            "More points resolve the Mie ripples better and cost runtime\n"
            "in direct proportion. 4000 is enough while you are tuning;\n"
            "50000 for a final number."
        )
        self.num_a.textChanged.connect(self._update_cost_hint)
        rad_form.addRow("grid points:", self.num_a)

        self.inversion_combo = QComboBox()
        self.inversion_combo.addItem("Smoothed trend", "trend")
        self.inversion_combo.addItem("Legacy (unsmoothed)", "legacy")
        self.inversion_combo.setToolTip(
            "Smoothed trend: inverts the monotone mean of C_sca(a). Grid-stable,\n"
            "and the right choice for sizing a population.\n"
            "Legacy: inverts the raw rippled curve, where one cross-section maps\n"
            "to several radii, so the answer shifts with the grid. Kept only to\n"
            "reproduce runs made before the trend method existed."
        )
        rad_form.addRow("inversion:", self.inversion_combo)

        self.cost_lbl = QLabel("")
        self.cost_lbl.setStyleSheet("color:#8a8a8a; font-size:10px;")
        self.cost_lbl.setWordWrap(True)
        rad_form.addRow(self.cost_lbl)
        ctrl_col.addWidget(rad_box)

        # --- run / cancel ---
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setStyleSheet(
            "QPushButton { background:#0e639c; padding:6px; font-weight:600; }"
            "QPushButton:hover { background:#1177bb; }"
            "QPushButton:disabled { background:#333; color:#777; }"
        )
        self.run_btn.clicked.connect(self._run)
        run_row.addWidget(self.run_btn, stretch=2)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        run_row.addWidget(self.cancel_btn, stretch=1)
        ctrl_col.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        ctrl_col.addWidget(self.progress_bar)

        self.status_label = QLabel(
            "Load areas, then Run. Each blob area (m²) is its scattering "
            "cross-section C_sca. In-memory areas need a pixel scale (Ctrl+M)."
        )
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#8a8a8a; font-size:11px;")
        ctrl_col.addWidget(self.status_label)

        ctrl_col.addStretch()

        self.export_btn = QPushButton("Export particle sizes CSV…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_csv)
        ctrl_col.addWidget(self.export_btn)

        cite = QLabel(
            "Method: Dasgupta et al., J. Flow Visualization and Image "
            "Processing 32(3), 2025."
        )
        cite.setStyleSheet("color: #6a6a6a; font-size: 10px;")
        cite.setWordWrap(True)
        ctrl_col.addWidget(cite)

        # ============================ right: results ======================
        res_col = QVBoxLayout()
        res_col.setSpacing(8)
        root.addLayout(res_col, stretch=1)

        cards = QHBoxLayout()
        cards.setSpacing(8)
        self.card_d10 = _StatCard("D10", "#6fa8ff")
        self.card_d50 = _StatCard("D50", "#5fd08a")
        self.card_d90 = _StatCard("D90", "#ff7b6b")
        self.card_fit = _StatCard("Rosin-Rammler fit", "#c9a0ff")
        for c in (self.card_d10, self.card_d50, self.card_d90, self.card_fit):
            cards.addWidget(c)
        res_col.addLayout(cards)

        self.warn_label = QLabel("")
        self.warn_label.setStyleSheet(
            "color:#ffb020; font-size:11px; background:#3a2f14;"
            " border:1px solid #6b5410; border-radius:4px; padding:5px;"
        )
        self.warn_label.setWordWrap(True)
        self.warn_label.setVisible(False)
        res_col.addWidget(self.warn_label)

        self.csca_plot = pg.PlotWidget()
        self.csca_plot.showGrid(x=True, y=True, alpha=0.3)
        self.csca_plot.setLabel('bottom', 'particle radius a', units='m')
        self.csca_plot.setLabel('left', 'C_sca', units='m²')
        self.csca_plot.setLogMode(x=True, y=True)
        self.csca_plot.setTitle("Forward Mie curve — cross-section vs radius")
        self.csca_legend = self.csca_plot.addLegend(offset=(-10, 10))
        self.csca_ph = PlotPlaceholder(
            self.csca_plot, "Forward Mie curve appears here after Run")
        res_col.addWidget(self.csca_plot, stretch=1)

        lower_row = QHBoxLayout()
        lower_row.addWidget(QLabel("Lower plot:"))
        self.lower_mode = QComboBox()
        self.lower_mode.addItem("Cumulative (Rosin-Rammler)", "cdf")
        self.lower_mode.addItem("Size distribution (histogram)", "pdf")
        self.lower_mode.currentIndexChanged.connect(self._redraw)
        lower_row.addWidget(self.lower_mode)
        lower_row.addStretch()
        res_col.addLayout(lower_row)

        self.cdf_plot = pg.PlotWidget()
        self.cdf_plot.showGrid(x=True, y=True, alpha=0.3)
        # Diameter, not radius: D10/D50/D90 are diameters, and the markers sit
        # on this axis. The C_sca plot above stays in radius because the Mie
        # size parameter x = k*a is defined on radius.
        self.cdf_plot.setLabel('bottom', 'particle diameter', units='m')
        self.cdf_plot.setLabel('left', 'cumulative fraction')
        self.cdf_legend = self.cdf_plot.addLegend(offset=(-10, 10))
        self.cdf_ph = PlotPlaceholder(
            self.cdf_plot, "Size distribution appears here after Run")
        res_col.addWidget(self.cdf_plot, stretch=1)

        self._on_source_changed()
        self._update_cost_hint()

    # ------------------------------------------------------------------

    def _set_status(self, text: str, kind: str = "info"):
        """Transient status line. Separate from the result cards so an error
        never reads as a result (they used to share one label)."""
        colour = {"info": "#8a8a8a", "busy": "#8ab4f8",
                  "error": "#ff7b6b", "ok": "#5fd08a"}.get(kind, "#8a8a8a")
        self.status_label.setStyleSheet(f"color:{colour}; font-size:11px;")
        self.status_label.setText(text)

    def _on_source_changed(self):
        csv = self.source_combo.currentText() == "CSV file"
        self.load_csv_btn.setVisible(csv)
        self.csv_file_lbl.setVisible(csv)
        self.csv_unit_combo.setVisible(csv)
        self.csv_unit_lbl.setVisible(csv)
        self.sample_lbl.setVisible(not csv)
        self.frames_row_host.setVisible(not csv)

    def _update_cost_hint(self):
        """Show what the current grid + sweep will cost. The forward sweep is
        O(num_a · n_max) with n_max set by the largest radius, and it is easy to
        ask for a multi-minute run without realising it."""
        try:
            num_a = int(self.num_a.text())
            a_stop = float(self.a_stop.text()) * self.UM
            lam = float(self.lam.text()) * 1e-9
            n_med = float(self.n_medium.text())
        except ValueError:
            self.cost_lbl.setText("Runtime estimate appears once the sweep is set.")
            return
        x_max = 2.0 * np.pi * n_med * a_stop / lam
        n_max = int(round(2 + x_max + 4 * x_max ** (1.0 / 3.0)))
        # ~1.4e-5 s per (grid point x order), calibrated on two measured runs of
        # the imgs_test set: 12.4 s at 4000 points and 153.7 s at 50000, both at
        # n_max=222. Only the order of magnitude matters — it exists so nobody
        # starts a multi-minute sweep without being told.
        est = num_a * n_max * 1.4e-5
        note = "" if est < 20 else "  — lower the grid or tighten the sweep to speed this up"
        self.cost_lbl.setText(
            f"{n_max} Mie orders × {num_a} grid points ≈ {est:.0f} s{note}"
        )

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
        unit = self.csv_unit_combo.currentText()
        self._areas_source = f"{len(areas)} areas from {os.path.basename(path)} ({unit}²)"
        self.areas_lbl.setText(f"✓ {len(areas)} areas — {os.path.basename(path)} ({unit}²)")
        self._set_status("CSV loaded. Set optics and Run.", "ok")
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
            self._set_status("Load areas first — Auto range reads them.", "error")
            return
        a_geom = np.sqrt(np.clip(self._areas_m2, 0, None) / np.pi)
        a_geom = a_geom[a_geom > 0]
        if a_geom.size == 0:
            return
        # Geometric radius already over-estimates the true radius (in the optics
        # limit C_sca→2πa², so a≈√(area/2π) < √(area/π)), so the upper bracket
        # only needs a little headroom. It used to be 1.5×, which on real data
        # pushed a_stop far past the largest particle — that inflates n_max
        # (runtime) and leaves most of the CDF x-axis empty for no benefit.
        a_start = max(a_geom.min() * 0.5, 1e-10)
        a_stop = a_geom.max() * 1.15
        self.a_start.setText(f"{a_start / self.UM:.4g}")
        self.a_stop.setText(f"{a_stop / self.UM:.4g}")
        self._update_cost_hint()

    # --- in-memory area collection (reuses RegionPropsWorker) ----------

    def _collect_inmemory(self):
        app = self.main_app
        if app.sequence_manager is None:
            self._set_status("No sequence loaded.", "error")
            return False
        if app.pipeline.scale is None:
            self._set_status(
                "No pixel scale set. Set one via Ctrl+M (Tools → Set Scale…) so "
                "areas can be converted to m², or switch to a CSV source.", "error")
            return False
        thresh_idx, thresh_op = app._find_threshold_op()
        if thresh_op is None or not thresh_op.enabled:
            self._set_status("No enabled threshold op in pipeline.", "error")
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
        self._set_busy(True)
        self.areas_lbl.setText(f"collecting from {len(sample_indices)} frames…")
        self.progress_bar.setFormat("Collecting areas — frame %v of %m")
        self.progress_bar.setRange(0, len(sample_indices))
        self.progress_bar.setValue(0)
        self._set_status(f"Collecting areas from {len(sample_indices)} frames…", "busy")
        self._worker.start()
        return True

    def _on_areas_ready(self, results: dict):
        self._worker = None
        areas_px2 = np.asarray(results.get('area', []), dtype=float)
        if areas_px2.size == 0:
            self._set_idle()
            self._set_status("No blobs found in sampled frames.", "error")
            return
        import mie
        scale = self.main_app.pipeline.scale
        unit_per_px = scale["value"] / scale["px"]
        unit_factor = mie.area_unit_to_m2_factor(scale["unit"])
        if unit_factor is None:
            self._set_idle()
            self._set_status(
                f"Scale unit '{scale['unit']}' not recognised (need m/cm/mm/µm/nm).",
                "error")
            return
        # px² → (scale unit)² → m²
        self._areas_m2 = areas_px2 * (unit_per_px ** 2) * unit_factor
        self._areas_source = f"{areas_px2.size} blobs (in-memory, scale {scale['unit']})"
        self.areas_lbl.setText(
            f"✓ {areas_px2.size:,} blobs — scale {scale['px']:g} px = "
            f"{scale['value']:g} {scale['unit']}".replace(",", " ")
        )
        if not self.a_start.text() or not self.a_stop.text():
            self._auto_range()
        self._compute()

    # --- run orchestration ---------------------------------------------

    def _run(self):
        if self.source_combo.currentText() == "CSV file":
            if self._areas_m2 is None:
                self._set_status("Load a CSV first.", "error")
                return
            self._compute()
        else:
            self._collect_inmemory()

    def _compute(self):
        if self._areas_m2 is None or self._areas_m2.size == 0:
            self._set_idle()
            self._set_status("No areas to analyse.", "error")
            return
        if not self.a_start.text() or not self.a_stop.text():
            self._set_idle()
            self._set_status("Radius sweep is empty — press “Auto range”.", "error")
            return
        try:
            a_start = float(self.a_start.text()) * self.UM
            a_stop = float(self.a_stop.text()) * self.UM
            params = dict(
                n_real=float(self.n_real.text()), n_imag=float(self.n_imag.text()),
                lam_nm=float(self.lam.text()), n_medium=float(self.n_medium.text()),
                mu=float(self.mu.text()), num_a=int(self.num_a.text()),
            )
        except ValueError as e:
            self._set_idle()
            self._set_status(f"Invalid parameter: {e}", "error")
            return
        if a_stop <= a_start:
            self._set_idle()
            self._set_status("Radius stop must be greater than start.", "error")
            return

        # Forward Mie is heavy — run it off the GUI thread.
        self._mie_worker = MieComputeWorker(
            self._areas_m2, params["n_real"], params["n_imag"], a_start, a_stop,
            params["lam_nm"], params["n_medium"], params["mu"], params["num_a"],
            self.inversion_combo.currentData(),
        )
        self._mie_worker.finished.connect(self._on_mie_finished)
        self._mie_worker.error.connect(self._on_error)
        self._mie_worker.cancelled.connect(self._on_mie_cancelled)
        self._mie_worker.progress.connect(self._on_mie_progress)
        self._set_busy(True)
        self.progress_bar.setFormat("Mie sweep — order %v of %m")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status(
            f"Computing Mie inversion for {self._areas_m2.size} particles "
            f"(grid {params['num_a']})…", "busy")
        self._mie_worker.start()

    def _on_mie_progress(self, done: int, total: int):
        if self.progress_bar.maximum() != total:
            self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)

    def _on_mie_finished(self, res):
        self._mie_worker = None
        self._set_idle()
        self._result = res
        self.export_btn.setEnabled(True)
        self._redraw()

    def _on_mie_cancelled(self):
        self._mie_worker = None
        self._set_idle()
        self._set_status("Mie sweep cancelled.", "info")

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait()
            self._worker = None
            self._set_idle()
            self._set_status("Cancelled.", "info")
        if self._mie_worker is not None:
            # Cooperative: the order loop notices on its next callback. Never
            # terminate() a thread sitting inside numpy.
            self._mie_worker.cancel()
            self._set_status("Cancelling Mie sweep…", "busy")
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, done: int, total: int):
        self.progress_bar.setValue(done)

    def _on_error(self, msg: str):
        self._worker = None
        self._mie_worker = None
        self._set_idle()
        self._set_status(f"Error: {msg.splitlines()[0]}", "error")
        print(f"[Mie] {msg}")

    def _set_busy(self, busy: bool):
        self.run_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.progress_bar.setVisible(busy)

    def _set_idle(self):
        self._set_busy(False)

    # --- drawing --------------------------------------------------------

    def _redraw(self):
        res = self._result
        if res is None:
            return
        um = lambda v: v * 1e6

        # --- headline cards ---
        src_tag = "RR fit" if res.rr_ok else "empirical"
        for card, d in ((self.card_d10, res.D10), (self.card_d50, res.D50),
                        (self.card_d90, res.D90)):
            if np.isfinite(d):
                card.set_value(f"{um(d):.3g} µm",
                               f"diameter · {d:.3e} m · {src_tag}")
            else:
                card.set_value("—", "not available")
        if res.rr_ok:
            self.card_fit.set_value(
                f"R² {res.r2:.4f}", f"b = {res.rr_b:.3e}   c = {res.rr_c:.3g}")
        else:
            self.card_fit.set_value("failed", "showing empirical percentiles")

        # --- warnings the user must not miss ---
        warns = []
        if not res.rr_ok:
            warns.append(
                f"Rosin-Rammler fit did not converge (n = {res.radii.size}); "
                f"D-values above are empirical percentiles."
            )
        # Radii pinned at a sweep bound mean the sweep does not cover the data.
        r = res.radii[np.isfinite(res.radii)]
        if r.size:
            lo = int(np.sum(np.isclose(r, res.a[0], rtol=1e-9)))
            hi = int(np.sum(np.isclose(r, res.a[-1], rtol=1e-9)))
            # Advice differs by end, so report them separately. A low-end clamp
            # is usually not a sweep problem at all: C_sca falls off as a^6 in
            # the Rayleigh limit, so 1-2 px noise blobs land below the curve
            # minimum no matter how the sweep is bracketed. Telling the user to
            # widen the sweep there sends them the wrong way.
            if lo:
                warns.append(
                    f"{lo} of {r.size} particles ({100.0 * lo / r.size:.0f}%) sit "
                    f"below the curve minimum and were pinned to a_start. Blobs of "
                    f"1-2 px scatter far less than the sweep floor, so this usually "
                    f"means noise blobs are being sized — clean up the mask "
                    f"(min blob area / binary smoothing) rather than widening the sweep."
                )
            if hi:
                warns.append(
                    f"{hi} of {r.size} particles ({100.0 * hi / r.size:.0f}%) exceed "
                    f"the curve maximum and were pinned to a_stop — raise the sweep "
                    f"stop so the largest particles are covered."
                )
        self.warn_label.setText("  ⚠  " + "\n  ⚠  ".join(warns) if warns else "")
        self.warn_label.setVisible(bool(warns))

        # --- C_sca curve ---
        self.csca_plot.clear()
        self.csca_ph.hide()
        self.csca_legend.clear()
        self.csca_plot.plot(res.a, res.csca, pen=pg.mkPen((90, 160, 255), width=2),
                            name="C_sca(a)")
        # Mark where the measured cross-sections actually sit on this curve —
        # without it there is no way to see whether the sweep brackets the data.
        areas = self._areas_m2
        if areas is not None and areas.size:
            fin = areas[np.isfinite(areas) & (areas > 0)]
            if fin.size:
                self.csca_plot.addItem(pg.InfiniteLine(
                    pos=np.log10(max(fin.min(), 1e-30)), angle=0,
                    pen=pg.mkPen((200, 200, 200), style=Qt.PenStyle.DotLine)))
                self.csca_plot.addItem(pg.InfiniteLine(
                    pos=np.log10(max(fin.max(), 1e-30)), angle=0,
                    pen=pg.mkPen((200, 200, 200), style=Qt.PenStyle.DotLine)))
                self.csca_plot.plot([], [], pen=pg.mkPen((200, 200, 200),
                                    style=Qt.PenStyle.DotLine),
                                    name="measured C_sca range")

        # --- lower plot ---
        self.cdf_plot.clear()
        self.cdf_ph.hide()
        self.cdf_legend.clear()
        mode = self.lower_mode.currentData()
        if mode == "pdf":
            self._draw_pdf(res)
        else:
            self._draw_cdf(res)

        n_txt = f"{res.radii.size:,}".replace(",", " ")
        self._set_status(
            f"{self._areas_source} · n = {res.n_sphere} · λ = {res.lam_nm:g} nm · "
            f"{n_txt} particles sized.", "ok")

    def _d_lines(self, res):
        """Dashed D10/D50/D90 markers, each carrying its own label. Previously
        these were three unlabelled dashes with no legend entry."""
        for name, d, col in (("D10", res.D10, (111, 168, 255)),
                             ("D50", res.D50, (95, 208, 138)),
                             ("D90", res.D90, (255, 123, 107))):
            if not np.isfinite(d):
                continue
            line = pg.InfiniteLine(
                pos=d, angle=90,
                pen=pg.mkPen(col, width=2, style=Qt.PenStyle.DashLine),
                label=f"{name} {d * 1e6:.2f} µm",
                labelOpts={"color": col, "position": 0.92, "movable": False,
                           "fill": (30, 30, 30, 200)},
            )
            self.cdf_plot.addItem(line)

    def _draw_cdf(self, res):
        self.cdf_plot.setLabel('left', 'cumulative fraction')
        self.cdf_plot.setTitle("Cumulative size distribution")
        self.cdf_plot.plot(
            res.cdf_bin_centers, res.cdf_values,
            pen=None, symbol='o', symbolSize=5,
            symbolBrush=(190, 190, 190), symbolPen=None, name='measured',
        )
        # Rosin-Rammler curve only when the fit converged (NaN params otherwise).
        # Evaluated on the DIAMETER grid: the fit is on diameters, so rr_b is a
        # diameter scale and plotting it against res.a (radius) would put the
        # curve at half the x of the D-markers.
        if res.rr_ok and np.isfinite(res.rr_b) and np.isfinite(res.rr_c):
            d = 2.0 * res.a
            rr = 1.0 - np.exp(-((d / res.rr_b) ** res.rr_c))
            self.cdf_plot.plot(d, rr, pen=pg.mkPen((255, 140, 90), width=2),
                               name='Rosin-Rammler fit')
        self._d_lines(res)
        self._fit_x_to_data(res)

    def _draw_pdf(self, res):
        """Diameter histogram. The CDF alone hides the mode and any bimodality —
        the shape most people look for first in a spray."""
        self.cdf_plot.setLabel('left', 'particle count')
        self.cdf_plot.setTitle("Size distribution")
        r = res.diameters[np.isfinite(res.diameters) & (res.diameters > 0)]
        if r.size == 0:
            return
        nbins = int(np.clip(np.sqrt(r.size), 20, 200))
        counts, edges = np.histogram(r, bins=nbins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = float(edges[1] - edges[0]) * 0.9
        self.cdf_plot.addItem(pg.BarGraphItem(
            x=centers, height=counts, width=width,
            brush=(90, 160, 255, 160), pen=pg.mkPen((90, 160, 255))))
        self.cdf_plot.plot([], [], pen=pg.mkPen((90, 160, 255), width=6),
                           name='particle count')
        self._d_lines(res)
        self._fit_x_to_data(res)

    def _fit_x_to_data(self, res):
        """Frame the x-axis on the particles, not the sweep. The RR curve is
        evaluated over the whole size grid, which used to stretch the axis far
        past the largest particle and leave most of the plot empty."""
        r = res.diameters[np.isfinite(res.diameters) & (res.diameters > 0)]
        if r.size == 0:
            return
        hi = max(float(np.percentile(r, 99.9)), float(res.D90) if np.isfinite(res.D90) else 0.0)
        self.cdf_plot.setXRange(0.0, hi * 1.08, padding=0)

    def _export_csv(self):
        res = self._result
        if res is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export particle sizes CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return
        lines = [
            f"# Mie particle sizing — Dasgupta et al., J. Flow Vis. Image Proc. 32(3), 2025",
            f"# D-values are DIAMETERS (m). The inversion works in radius; both are given per particle.",
            f"# n_sphere={res.n_sphere}, lambda_nm={res.lam_nm}, n_medium={res.n_medium}, mu={res.mu}",
            f"# D10_m={res.D10:.6e}, D50_m={res.D50:.6e}, D90_m={res.D90:.6e}, "
            f"source={'rosin_rammler' if res.rr_ok else 'empirical_percentiles'}",
            f"# emp_D10_m={res.emp_D10:.6e}, emp_D50_m={res.emp_D50:.6e}, emp_D90_m={res.emp_D90:.6e}",
            f"# RR_b={res.rr_b:.6e} (diameter scale), RR_c={res.rr_c:.6e}, R2={res.r2:.6f}, RR_ok={res.rr_ok}",
            "particle_idx,area_m2,radius_m,radius_um,diameter_m,diameter_um",
        ]
        for i, (area, r) in enumerate(zip(self._areas_m2, res.radii)):
            lines.append(f"{i},{area:.6e},{r:.6e},{r * 1e6:.6g},"
                         f"{2 * r:.6e},{2 * r * 1e6:.6g}")
        try:
            with open(path, 'w') as f:
                f.write("\n".join(lines))
            self._set_status(f"Exported {res.radii.size} particles → {path}", "ok")
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

        # --- histogram controls -------------------------------------------
        # After background subtraction nearly every pixel sits near zero, so on
        # a 0-4095 axis the data occupies the leftmost few percent and the
        # vmin/vmax handles are pinned at the extremes — unusable without a way
        # to zoom in or to have the range chosen for you.
        hist_ctrl = QHBoxLayout()
        self.auto_btn = QPushButton("Auto contrast")
        self.auto_btn.setToolTip(
            "Set vmin/vmax to the 0.1 - 99.9 percentile of the current frame,\n"
            "ignoring the empty tail. Display only — pixel data is untouched."
        )
        self.auto_btn.clicked.connect(self.auto_contrast)
        hist_ctrl.addWidget(self.auto_btn)

        self.full_btn = QPushButton("Full range")
        self.full_btn.setToolTip("Reset vmin/vmax to the full bit depth.")
        self.full_btn.clicked.connect(self.full_range)
        hist_ctrl.addWidget(self.full_btn)

        self.hist_log_check = QCheckBox("Log count")
        self.hist_log_check.setToolTip(
            "Log-scale the histogram counts. The background peak is orders of "
            "magnitude taller than the particle tail and flattens it otherwise."
        )
        self.hist_log_check.toggled.connect(self._replot_histogram)
        hist_ctrl.addWidget(self.hist_log_check)

        self.hist_zoom_check = QCheckBox("Zoom to data")
        self.hist_zoom_check.setChecked(True)
        self.hist_zoom_check.setToolTip(
            "Frame the x-axis on the occupied intensity range instead of the "
            "full bit depth."
        )
        self.hist_zoom_check.toggled.connect(self._replot_histogram)
        hist_ctrl.addWidget(self.hist_zoom_check)
        hist_ctrl.addStretch()
        layout.addLayout(hist_ctrl)

        self.curve_plot = pg.PlotWidget(title="8-Bit Mapping Preview")
        self.curve_plot.setLabel('left', "Output (0-255)")
        self.curve_plot.setLabel('bottom', "Input Intensity")
        self.curve_plot.setBackground((0, 0, 0, 0))
        self.curve_plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve_plot.setMouseEnabled(x=False, y=False)
        self.curve_plot.hideButtons()
        self.curve_plot.setYRange(-10, 265, padding=0)
        # The mapping preview is one monotone curve; the histogram is the plot
        # people actually work in. 3:1 instead of the old equal split.
        self.curve_plot.setMaximumHeight(170)
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
        # X range is owned by _draw_histogram — it honours "Zoom to data", which
        # a hardcoded full-bit-depth range here would silently override.
        self._draw_histogram()
        self.update_curve_preview()
        self._updating = False

    def update_histogram(self, raw_frame):
        # 4× subsample for large frames — negligible accuracy loss for display histogram
        sample = raw_frame[::2, ::2] if raw_frame.size > 262144 else raw_frame
        y, x = np.histogram(sample, bins=250, range=(0, self.main_app.bit_depth_max))
        # Kept so the log/zoom toggles can re-render without re-reading a frame.
        self._hist_x, self._hist_y = x, y
        self._draw_histogram()

    def _draw_histogram(self):
        x, y = getattr(self, "_hist_x", None), getattr(self, "_hist_y", None)
        if x is None or y is None:
            return
        if self.hist_log_check.isChecked():
            plot_y = np.log10(y.astype(np.float64) + 1.0)
            self.hist_plot.setLabel('left', "log₁₀(count+1)")
        else:
            plot_y = y
            self.hist_plot.setLabel('left', "Pixel Count")
        self.hist_curve.setData(x, plot_y)

        if self.hist_zoom_check.isChecked():
            lo, hi = self._occupied_range()
            span = max(hi - lo, 1.0)
            # Stretch to keep the vmin/vmax handles grabbable, but only so far —
            # at the default vmax (full bit depth) an unbounded stretch would
            # restore the very full-range view this toggle exists to escape.
            # Beyond the margin the handle is off-screen and "Full range" /
            # "Auto contrast" are the way back to it.
            margin = 0.5 * span
            lo = max(min(lo, self.main_app.vmin), lo - margin)
            hi = min(max(hi, self.main_app.vmax), hi + margin)
            span = max(hi - lo, 1.0)
            self.hist_plot.setXRange(lo - 0.02 * span, hi + 0.02 * span, padding=0)
        else:
            self.hist_plot.setXRange(0, self.main_app.bit_depth_max, padding=0.02)

    def _replot_histogram(self):
        self._draw_histogram()

    def _occupied_range(self, mass: float = 0.9995):
        """Intensity range holding `mass` of the pixels.

        Not first/last non-empty bin: a handful of hot pixels sit far out in the
        tail, so "any pixels here" stretched the view to ~2900 for data that is
        entirely below 300. Cumulative mass ignores that tail while still
        covering everything the eye can see.
        """
        x, y = getattr(self, "_hist_x", None), getattr(self, "_hist_y", None)
        if x is None or y is None or not y.any():
            return 0.0, float(self.main_app.bit_depth_max)
        c = np.cumsum(y.astype(np.float64))
        total = c[-1]
        if total <= 0:
            return 0.0, float(self.main_app.bit_depth_max)
        c /= total
        lo_i = int(np.searchsorted(c, (1.0 - mass) / 2.0))
        hi_i = int(np.searchsorted(c, 1.0 - (1.0 - mass) / 2.0))
        lo_i = max(0, min(lo_i, len(x) - 2))
        hi_i = max(lo_i + 1, min(hi_i + 1, len(x) - 1))
        return float(x[lo_i]), float(x[hi_i])

    def auto_contrast(self):
        """vmin/vmax from the 0.1-99.9 percentile of the current frame.

        Display-only, like every other control in this window — the pixel data
        is never touched. Percentiles rather than min/max because a handful of
        hot pixels otherwise stretch the range back to useless.
        """
        app = self.main_app
        frame = app._last_frame_for_histogram
        if frame is None:
            frame = app.current_raw_frame
        if frame is None:
            return
        sample = frame[::2, ::2] if frame.size > 262144 else frame
        lo, hi = (float(v) for v in np.percentile(sample, [0.1, 99.9]))
        if hi <= lo:
            hi = lo + 1.0
        app.vmin, app.vmax = lo, hi
        app._display_lut = None          # force LUT rebuild
        self.sync_from_main()
        app.update_frame_display()

    def full_range(self):
        app = self.main_app
        app.vmin, app.vmax = 0, app.bit_depth_max
        app._display_lut = None
        self.sync_from_main()
        app.update_frame_display()

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
        # Same reason as the generic ParamsWidget: this form has the longest
        # labels in the app ("Invert intensity (dark particles):" at 207px), and
        # QFormLayout sizes its label column to the widest one.
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # --- dynamic intensity-source combo ---
        self.source_combo = QComboBox()
        # AdjustToContents sizes the combo to its longest entry ("Pre-threshold
        # frame (default)"), which pushed the form wider than the params
        # viewport and forced a horizontal scrollbar. Cap the demanded width and
        # let the popup show the full text instead.
        self.source_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source_combo.setMinimumContentsLength(10)
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

        # --- invert_intensity ---
        # Peak detection finds maxima; dark-on-bright sources (shadowgraphy)
        # need the intensity negated first.  Should match the threshold op's
        # polarity setting.
        self.combo_invert = QComboBox()
        self.combo_invert.addItems(["no", "yes"])
        self.combo_invert.setCurrentText(op.params.get("invert_intensity", "no"))
        self.combo_invert.setToolTip(
            "Set to 'yes' when particles are darker than the background\n"
            "(shadowgraphy / backlit imaging).  Peak detection finds maxima,\n"
            "so dark particle cores must be inverted to be found.\n"
            "Match this to the threshold op's Polarity setting."
        )
        self.combo_invert.currentTextChanged.connect(
            lambda v: self._emit("invert_intensity", v)
        )
        layout.addRow("Invert intensity (dark particles):", self.combo_invert)

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
        thresh_idx, _ = self.main_app._find_threshold_op()

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
                self._source_choices.append((f"{i}: {type(op).display()}", i))

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
        # Long labels ("Invert intensity (dark particles):") otherwise set the
        # label-column width for the whole form, pushing its minimum width past
        # the params viewport and forcing a horizontal scrollbar. WrapLongRows
        # drops the field under its label instead of widening the form.
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

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
                # keyboardTracking=False: without it every intermediate value
                # while typing "125" fires valueChanged -> a full pipeline
                # recompute. Now typing commits once (Enter / focus-out), while
                # the arrows still preview live. CropParamsWidget has always
                # used editingFinished for the same reason.
                w.setKeyboardTracking(False)
                w.valueChanged.connect(lambda val, k=key: self._emit(k, val))
                # `step == 2` marks a kernel/window param that OpenCV requires to
                # be odd. The ops defensively do `if k % 2 == 0: k += 1`, so an
                # even entry used to leave the spinbox showing a value the op was
                # not using. Snap on commit (editingFinished, not valueChanged —
                # snapping mid-typing would fight the user on the first digit).
                if step == 2:
                    w.editingFinished.connect(
                        lambda k=key, sb=w, lo=lo, hi=hi: self._snap_odd(k, sb, lo, hi)
                    )
            elif widget_hint == "combo":
                w = QComboBox()
                # See IntensityWatershedParamsWidget: AdjustToContents makes the
                # combo demand its longest entry's width, which can force a
                # horizontal scrollbar in the params panel ("bright_on_dark").
                w.setSizeAdjustPolicy(
                    QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                w.setMinimumContentsLength(10)
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

            help_text = spec.get("help")
            if help_text:
                w.setToolTip(help_text)

            self._widgets[key] = w
            label = QLabel(spec.get("label", key) + ":")
            if help_text:
                # Tooltip on the label too — users hover the name, not the box.
                label.setToolTip(help_text)
            layout.addRow(label, w)

    def _snap_odd(self, key, spin, lo, hi):
        """Round a committed even value up to the next odd one, in range.

        Keeps the displayed value equal to the value the op actually applies.
        Writing through setValue() re-enters _emit via valueChanged, which is
        what we want — the params dict and the preview both follow.
        """
        val = spin.value()
        if val % 2 == 1:
            return
        snapped = val + 1 if val + 1 <= hi else val - 1
        if snapped < lo:
            return
        spin.setValue(snapped)

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
        # Taller than the old 520 so the largest params form (6 rows, for
        # IntensityWatershedSplitOp) fits without scrolling, and wider so combo
        # values are not clipped.
        self.setGeometry(1150, 100, 380, 720)
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

        # The old "Display" group box lived here. It was dead UI — its two
        # labels were written once at construction and never updated, so it
        # permanently claimed "Base stage: (all) / Overlay stage: (none)" even
        # while the threshold overlay was rendering. Removed rather than wired
        # up: the information is already visible in the image itself, and the
        # ~60px it occupied is exactly what the params area was starved of.

        # --- params area ---
        self.params_group = QGroupBox("Stage Params")
        self.params_group.setMinimumHeight(200)
        params_outer = QVBoxLayout(self.params_group)
        params_outer.setContentsMargins(4, 4, 4, 4)

        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Let the form set the viewport width so long combo values ("Pre-threshold
        # frame (default)") don't trigger a horizontal scrollbar.
        self.params_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        params_outer.addWidget(self.params_scroll)
        self._show_params(None)  # set initial placeholder

        # stretch=1: the params area is the only thing that should absorb spare
        # height. Without it the group got its bare minimum (~110px) while the
        # stage list took 160px, hiding most ops' params behind two scrollbars —
        # AdaptiveThresholdOp's `polarity` and IntensityWatershedSplitOp's
        # `invert_intensity` were unreachable at the default window size.
        outer.addWidget(self.params_group, stretch=1)

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
        if getattr(op, "is_threshold_op", False):
            cache = pipeline._caches[idx] if 0 <= idx < len(pipeline._caches) else {}
            return op._batch_was_run and not bool(cache)
        return False

    def rebuild_list(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for i, op in enumerate(self.main_app.pipeline.operations):
            item = QListWidgetItem(type(op).display())
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
        # Show human labels, carry the registry key as itemData — the key is the
        # serialization name and must not leak into the UI.
        for key, cls in sorted(OPERATION_REGISTRY.items(), key=lambda kv: kv[1].display()):
            combo.addItem(cls.display(), key)
        layout.addWidget(combo)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cls = OPERATION_REGISTRY[combo.currentData()]
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
        if not getattr(op, "is_threshold_op", False):
            return
        self.main_app._run_threshold_batch(op, idx)

    def _thresh_cache_ready_for_selected(self) -> bool:
        """Return True when Save Masks should be enabled for the current selection.

        Conditions:
          1. Selected op is a threshold op (AdaptiveThresholdOp / GlobalThresholdOp)
             OR a downstream is_binary_mask_op.
          2. The active threshold op exists in the pipeline with a non-empty batch cache.
        """
        idx = self._selected_idx
        ops = self.main_app.pipeline.operations
        op = ops[idx] if 0 <= idx < len(ops) else None
        if op is None:
            return False
        # Condition 1: selected op is threshold or a downstream binary-mask op
        is_threshold = getattr(op, "is_threshold_op", False)
        is_binary_downstream = (
            getattr(op, "is_binary_mask_op", False) and not is_threshold
        )
        if not (is_threshold or is_binary_downstream):
            return False
        # Condition 2: AdaptiveThresholdOp has a non-empty batch cache
        thresh_idx, thresh_op = self.main_app._find_threshold_op()
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
        # Kept as attributes (not locals) so _refresh_button_states can gate
        # them — exporting nothing is a no-op the user has to discover.
        self.save_btn = QPushButton("Save JSON…")
        self.save_btn.clicked.connect(main_app.save_facet_sessions_dialog)
        row3.addWidget(self.save_btn)
        load_btn = QPushButton("Load JSON…")     # always valid — nothing needed
        load_btn.clicked.connect(main_app.load_facet_sessions_dialog)
        row3.addWidget(load_btn)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.clicked.connect(main_app.export_facet_csv_dialog)
        row3.addWidget(self.export_btn)
        outer.addLayout(row3)

        self._plot_placeholder = PlotPlaceholder(
            self.plot_widget, "Thickness vs position appears here once you\n"
                              "finish an interface and click surface points")
        self._refresh_button_states()

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
            self._refresh_button_states()
        finally:
            self._updating = False

    def _refresh_button_states(self):
        """Gate the buttons on what actually exists.

        All five used to sit enabled with no session at all — clicking them was
        a silent no-op, which reads as a broken button rather than an
        inapplicable one.
        """
        session = self.active_session()
        has_session = session is not None
        has_points = bool(session and (session.interface_points or session.surface_points))
        # "Finish Interface" only means something during phase A, and the fit
        # needs at least 2 points to define a line.
        self.finish_btn.setEnabled(
            has_session and session.phase == "interface"
            and len(session.interface_points) >= 2
        )
        self.undo_btn.setEnabled(has_points)
        self.clear_btn.setEnabled(has_points)
        self.delete_btn.setEnabled(self.session_list.currentRow() >= 0 and bool(self.sessions))
        any_measurements = any(
            sess.interface_points and sess.surface_points for sess in self.sessions
        )
        self.export_btn.setEnabled(any_measurements)
        self.save_btn.setEnabled(bool(self.sessions))

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

        # Hide the idle hint only once there is something to draw, otherwise an
        # empty plot auto-ranges into a meaningless 0.1-0.9 tick grid.
        if any(sess.visible and sess.measurements() for sess in self.sessions):
            self._plot_placeholder.hide()
        else:
            self._plot_placeholder.show()

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
        # Guards the slider <-> spinbox <-> current_frame sync against feedback.
        self._scrub_sync = False
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

        # --- frame scrubber ---------------------------------------------
        # Arrow keys alone made frame 800 of 2000 an 800-keypress trip, in the
        # one tool whose headline feature is scrubbing long sequences.
        scrub_row = QHBoxLayout()
        scrub_row.setContentsMargins(6, 0, 6, 0)
        scrub_row.setSpacing(8)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setPageStep(10)
        self.frame_slider.setToolTip(
            "Drag to scrub. Arrow keys step one frame; PageUp/PageDown step 10;\n"
            "Home/End jump to the first/last frame."
        )
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        scrub_row.addWidget(self.frame_slider, stretch=1)

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setEnabled(False)
        self.frame_spin.setPrefix("frame ")
        self.frame_spin.setToolTip("Type a frame number to jump straight to it.")
        self.frame_spin.setKeyboardTracking(False)   # commit on Enter / focus-out
        self.frame_spin.valueChanged.connect(self._on_frame_spin)
        scrub_row.addWidget(self.frame_spin)
        layout.addLayout(scrub_row)

        # Bottom bar: pixel probe on the left, active-pipeline summary on the
        # right. Previously a single centred label that was empty unless the
        # cursor was over the image — ~50px of permanently blank window, and no
        # way to see which stages were live without opening the pipeline panel.
        info_row = QHBoxLayout()
        info_row.setContentsMargins(8, 0, 8, 0)
        self.pixel_info_label = QLabel("")
        self.pixel_info_label.setMinimumHeight(20)
        info_row.addWidget(self.pixel_info_label)
        info_row.addStretch()
        self.pipeline_summary_label = QLabel("")
        self.pipeline_summary_label.setStyleSheet("color:#8a8a8a;")
        self.pipeline_summary_label.setToolTip("Enabled pipeline stages, in order (Ctrl+P to edit).")
        info_row.addWidget(self.pipeline_summary_label)
        layout.addLayout(info_row)

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

    def _find_threshold_op(self):
        """Return (idx, op) for the first threshold-stage op in the pipeline
        (any op with is_threshold_op=True — AdaptiveThresholdOp or
        GlobalThresholdOp), regardless of enabled state. Returns (-1, None)
        if none exists. At most one such op should be enabled at a time
        (enforced by _toggle_threshold_op); overlay/binary-chain/batch-cache
        logic assumes a single active threshold stage."""
        for i, op in enumerate(self.pipeline.operations):
            if getattr(op, "is_threshold_op", False):
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
        Stops before the active threshold op (shown as RGBA overlay, not base image)
        and before CropOp when in crop_mode (full image shown while drawing ROI)."""
        n_ops = len(self.pipeline.operations)
        stop_at = n_ops if self.display_base_stage == -1 else self.display_base_stage + 1

        thresh_idx, thresh_op = self._find_threshold_op()
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

    def _run_threshold_batch(self, op: Operation, op_idx: int):
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
            if getattr(op, "is_threshold_op", False):
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
        self.threshold_action.triggered.connect(
            lambda checked: self._toggle_threshold_op(AdaptiveThresholdOp, self.threshold_action, checked))
        analysis_menu.addAction(self.threshold_action)

        self.global_threshold_action = QAction("Global Threshold (Otsu)", self, checkable=True)
        self.global_threshold_action.triggered.connect(
            lambda checked: self._toggle_threshold_op(GlobalThresholdOp, self.global_threshold_action, checked))
        analysis_menu.addAction(self.global_threshold_action)

        analysis_menu.addSeparator()

        self.morphology_action = QAction("Morphology (mask)", self, checkable=True)
        self.morphology_action.triggered.connect(
            lambda checked: self._toggle_filter(MorphologyOp, self.morphology_action, checked))
        analysis_menu.addAction(self.morphology_action)

        self.binary_smooth_action = QAction("Binary Smooth (mask)", self, checkable=True)
        self.binary_smooth_action.triggered.connect(
            lambda checked: self._toggle_filter(BinarySmoothOp, self.binary_smooth_action, checked))
        analysis_menu.addAction(self.binary_smooth_action)

        self.blob_size_filter_action = QAction("Blob Size Filter (mask)", self, checkable=True)
        self.blob_size_filter_action.setToolTip(
            "Drop connected blobs outside an area range.\n"
            "Area-exact: surviving blobs are copied pixel-for-pixel, unlike\n"
            "morphological opening which erodes what it keeps."
        )
        self.blob_size_filter_action.triggered.connect(
            lambda checked: self._toggle_filter(BlobSizeFilterOp, self.blob_size_filter_action, checked))
        analysis_menu.addAction(self.blob_size_filter_action)

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

        help_menu = menu_bar.addMenu("Help")
        shortcuts_action = QAction("Keyboard Shortcuts…", self)
        shortcuts_action.setShortcut("F1")
        shortcuts_action.triggered.connect(self.show_shortcuts_dialog)
        help_menu.addAction(shortcuts_action)
        about_action = QAction("About tiffscope", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    # Single source for the in-app shortcut list. Keep in sync with README.
    SHORTCUTS = [
        ("Navigation", [
            ("← / →", "Previous / next frame"),
            ("PageUp / PageDown", "Jump 10 frames"),
            ("Home / End", "First / last frame"),
            ("Ctrl+O", "Open sequence"),
        ]),
        ("View", [
            ("Ctrl+T", "Contrast & histogram"),
            ("Ctrl+F", "Optical flow overlay"),
            ("Ctrl+R", "Rotate 90° CW"),
            ("Ctrl+0", "Reset zoom / pan"),
            ("Ctrl++ / Ctrl+-", "Zoom in / out"),
        ]),
        ("Pipeline & analysis", [
            ("Ctrl+P", "Pipeline panel"),
            ("Ctrl+Shift+M", "Performance monitor"),
            ("Ctrl+Shift+C", "Toggle crop mode"),
            ("Ctrl+Shift+X", "Clear crop"),
            ("Enter / Esc", "Confirm / cancel ROI"),
        ]),
        ("Measurement", [
            ("Ctrl+M", "Set pixel scale"),
            ("H / V", "Place horizontal / vertical ray"),
            ("Ctrl+Shift+P", "Facet thickness panel"),
            ("Ctrl+Shift+A", "Start / resume facet clicking"),
        ]),
    ]

    def _refresh_pipeline_summary(self):
        """Right-hand bottom-bar text: enabled stages in order."""
        ops = [op for op in self.pipeline.operations if op.enabled]
        if not ops:
            self.pipeline_summary_label.setText("Pipeline: (none)")
            return
        names = " → ".join(type(op).display() for op in ops)
        n_off = len(self.pipeline.operations) - len(ops)
        suffix = f"   ({n_off} disabled)" if n_off else ""
        self.pipeline_summary_label.setText(f"Pipeline: {names}{suffix}")

    def show_shortcuts_dialog(self):
        rows = []
        for section, items in self.SHORTCUTS:
            rows.append(
                f'<tr><td colspan="2" style="padding:10px 0 3px 0; color:#8ab4f8;'
                f' font-weight:600;">{section}</td></tr>'
            )
            for keys, desc in items:
                rows.append(
                    f'<tr><td style="padding:1px 18px 1px 0; white-space:nowrap;">'
                    f'<code>{keys}</code></td>'
                    f'<td style="padding:1px 0;">{desc}</td></tr>'
                )
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        lay = QVBoxLayout(dlg)
        lbl = QLabel(f'<table style="border-collapse:collapse;">{"".join(rows)}</table>')
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(lbl)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec()

    def show_about_dialog(self):
        QMessageBox.about(
            self, "About tiffscope",
            "<b>tiffscope</b><br>"
            "Fast scrubbing viewer and preprocessing pipeline for scientific "
            "TIFF image sequences.<br><br>"
            "MIT licensed &middot; "
            "<a href='https://github.com/tiffscope/tiffscope'>"
            "github.com/tiffscope/tiffscope</a><br><br>"
            "Mie particle sizing method: Dasgupta et al., <i>J. Flow "
            "Visualization and Image Processing</i> 32(3), 2025."
        )

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
            (AdaptiveThresholdOp, self.threshold_action),
            (GlobalThresholdOp, self.global_threshold_action),
            (CLAHEOp, self.clahe_action),
            (GaussianBlurOp, self.gaussian_action),
            (SharpenOp, self.sharpen_action),
            (LowPassOp, self.lowpass_action),
            (HighPassOp, self.highpass_action),
            (RollingBallBgOp, self.rollingball_action),
            (MorphologyOp, self.morphology_action),
            (BinarySmoothOp, self.binary_smooth_action),
            (BlobSizeFilterOp, self.blob_size_filter_action),
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

    def _toggle_threshold_op(self, cls, action, checked):
        """Like _toggle_filter, but for threshold-stage ops (AdaptiveThresholdOp /
        GlobalThresholdOp). Enabling one disables any other threshold op first —
        overlay/binary-chain/batch-cache logic assumes a single active threshold
        stage (see _find_threshold_op)."""
        if checked:
            for other_cls in (AdaptiveThresholdOp, GlobalThresholdOp):
                if other_cls is cls:
                    continue
                other_idx, other_op = self._find_op(other_cls)
                if other_op is not None and other_op.enabled:
                    other_op.enabled = False
                    self.pipeline._invalidate_from(other_idx)
            _, op = self._ensure_op(cls)
            op.enabled = True
            self.pipeline_panel.rebuild_list()
            self._select_op_in_panel(cls)
        else:
            idx, op = self._find_op(cls)
            if op is not None:
                op.enabled = False
                self.pipeline._invalidate_from(idx)
                self.pipeline_panel.rebuild_list()
        self._sync_filter_menu_actions()
        self.update_frame_display()

    def save_threshold_masks(self):
        idx, op = self._find_threshold_op()
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
                thresh_idx, thresh_op = self._find_threshold_op()
                if thresh_op is not None and thresh_op.enabled:
                    cached = self.pipeline.get_batch_cache(thresh_idx, self.current_frame)
                    if cached is not None:
                        mask = cached
                    else:
                        # display_frame already stops before the threshold op.
                        # Label from the actual op — this span used to be the
                        # literal "AdaptiveThresholdOp" and mislabelled every
                        # GlobalThresholdOp run.
                        with self.perf.span(f"  {type(thresh_op).display()} (live)"):
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
                self._refresh_pipeline_summary()
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

            self._sync_scrubber()
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

        # All frame navigation goes through _goto_frame so the slider and the
        # spinbox stay in sync no matter which route the user took.
        key = event.key()
        if key == Qt.Key.Key_Right:
            self._goto_frame(self.current_frame + 1)
        elif key == Qt.Key.Key_Left:
            self._goto_frame(self.current_frame - 1)
        elif key == Qt.Key.Key_PageDown:
            self._goto_frame(self.current_frame + 10)
        elif key == Qt.Key.Key_PageUp:
            self._goto_frame(self.current_frame - 10)
        elif key == Qt.Key.Key_Home:
            self._goto_frame(0)
        elif key == Qt.Key.Key_End:
            self._goto_frame(self.num_frames - 1)

    # ------------------------------------------------------------------
    # Frame navigation
    # ------------------------------------------------------------------

    def _goto_frame(self, idx: int):
        """Single entry point for frame changes (keys, slider, spinbox).

        Clamps, redraws, and mirrors the new index into both scrubber widgets.
        `_scrub_sync` breaks the feedback loop: setValue() on either widget
        emits valueChanged, which would call back in here and re-render.
        """
        if self.sequence_manager is None:
            return
        idx = max(0, min(int(idx), self.num_frames - 1))
        if idx == self.current_frame:
            return
        self.current_frame = idx
        self._sync_scrubber()
        self.update_frame_display()

    def _sync_scrubber(self):
        """Push current_frame into the slider + spinbox without re-triggering."""
        if self._scrub_sync:
            return
        self._scrub_sync = True
        try:
            n = self.num_frames if self.sequence_manager is not None else 0
            has_seq = n > 0
            self.frame_slider.setEnabled(has_seq and n > 1)
            self.frame_spin.setEnabled(has_seq)
            if has_seq:
                self.frame_slider.setRange(0, n - 1)
                self.frame_spin.setRange(1, n)
                self.frame_spin.setSuffix(f" / {n}")
                self.frame_slider.setValue(self.current_frame)
                self.frame_spin.setValue(self.current_frame + 1)   # 1-based for humans
        finally:
            self._scrub_sync = False

    def _on_frame_slider(self, value: int):
        if self._scrub_sync:
            return
        self._goto_frame(value)

    def _on_frame_spin(self, value: int):
        if self._scrub_sync:
            return
        self._goto_frame(value - 1)   # spinbox is 1-based

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
