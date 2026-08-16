"""Regenerate the annotated screenshots for docs/MIE-WALKTHROUGH.md.

Same approach as make_media.py: the real app is driven programmatically and
each window is captured with QWidget.grab(), so the images are reproducible
rather than hand-recorded. Rerun after a UI change.

    python docs/make_walkthrough.py <sequence-folder> <pipeline.json> <out-dir>

Callout badges are positioned from live widget geometry (widget.mapTo(root)),
not hardcoded pixel coordinates. A moved or resized control therefore keeps its
badge; only a *renamed* attribute breaks the script, and it breaks loudly.

Two traps inherited from make_media.py that also apply here:
  * QWidget.grab() returns a pixmap with the screen's devicePixelRatio, so
    widget coordinates must be multiplied by it before painting. Skipping this
    puts every badge in the top-left quadrant on a HiDPI display.
  * A modal QDialog blocks inside exec(), so it cannot be grabbed from the
    calling code. Schedule the grab with QTimer.singleShot and reach it via
    QApplication.activeModalWidget() (see shot_scale_dialog).
"""
import glob
import os
import sys

if len(sys.argv) != 4:
    sys.exit(__doc__)

SEQ, PLJSON, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
# repo root is this file's parent — keep it path-independent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(OUT, "walkthrough")
os.makedirs(OUT, exist_ok=True)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QEventLoop, QPoint, QRect, Qt
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont

import main as tv
from pipeline import Pipeline
from operations import BgSubtractOp, AdaptiveThresholdOp, BlobSizeFilterOp

app = QApplication(sys.argv)
app.setStyleSheet(tv.modern_stylesheet)

ACCENT = QColor("#ffb020")
INK = QColor("#1a1a1a")


def pump(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _rect_of(root, widget, dpr, pad=3):
    """Widget bounds in the grabbed pixmap's coordinate space."""
    tl = widget.mapTo(root, QPoint(0, 0))
    return QRect(
        int((tl.x() - pad) * dpr), int((tl.y() - pad) * dpr),
        int((widget.width() + 2 * pad) * dpr),
        int((widget.height() + 2 * pad) * dpr),
    )


def _badge_centre(rect, place, r, pm):
    """Badge sits OUTSIDE the outline, so it never covers a field's value.

    That was the first version's bug: anchoring the badge on the rect corner
    put it straight on top of the spinbox contents it was pointing at.
    """
    if place == "l":
        cx, cy = rect.left() - r - 2, rect.center().y()
    elif place == "r":
        cx, cy = rect.right() + r + 2, rect.center().y()
    elif place == "tr":
        cx, cy = rect.right() + r, rect.top() - r
    else:                                    # "tl"
        cx, cy = rect.left() - r, rect.top() - r
    return QPoint(max(r, min(cx, pm.width() - r)),
                  max(r, min(cy, pm.height() - r)))


def shot(root, name, marks=(), width=1000):
    """Grab `root`, outline each marked widget, number it, save.

    marks: iterable of (badge_number, widget_or_QRect) with an optional third
    element placing the badge: "tl" (default), "l", "r" or "tr".
    """
    root.repaint()
    pump(250)
    pm = root.grab()
    dpr = pm.devicePixelRatio()

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont()
    font.setBold(True)
    font.setPixelSize(int(15 * dpr))
    p.setFont(font)

    for mark in marks:
        num, target = mark[0], mark[1]
        place = mark[2] if len(mark) > 2 else "tl"
        rect = (QRect(int(target.x() * dpr), int(target.y() * dpr),
                      int(target.width() * dpr), int(target.height() * dpr))
                if isinstance(target, QRect) else _rect_of(root, target, dpr))
        p.setPen(QPen(ACCENT, max(2, int(2 * dpr))))
        p.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        p.drawRoundedRect(rect, 5 * dpr, 5 * dpr)

        r = int(13 * dpr)
        c = _badge_centre(rect, place, r, pm)
        p.setBrush(QBrush(ACCENT))
        p.setPen(QPen(INK, max(1, int(1 * dpr))))
        p.drawEllipse(c, r, r)
        p.setPen(QPen(INK))
        p.drawText(QRect(c.x() - r, c.y() - r, 2 * r, 2 * r),
                   Qt.AlignmentFlag.AlignCenter, str(num))
    p.end()

    if width and pm.width() > width * dpr:
        pm = pm.scaledToWidth(int(width * dpr),
                              Qt.TransformationMode.SmoothTransformation)
    path = os.path.join(OUT, name)
    pm.save(path)
    print(f"  {name}: {os.path.getsize(path) / 1e3:.0f} kB")


def wait_for(pred, timeout_s=600):
    for _ in range(int(timeout_s * 2)):
        pump(500)
        if pred():
            return True
    raise TimeoutError("worker did not finish")


# ------------------------------------------------------------------ setup
win = tv.TiffViewerApp()
win.resize(1180, 780)
win.show()
pump(600)
win.load_and_display_data(sorted(glob.glob(os.path.join(SEQ, "*.TIF")))[0])
pump(1200)

# ------------------------------------------------- 1. raw sequence loaded
print("01-open.png")
win.tool_window.auto_contrast()
win._goto_frame(12)
pump(400)
shot(win, "01-open.png", marks=[
    (1, win.status_label),
    (2, win.frame_slider),
    (3, win.frame_spin),
])

# ------------------------------------------- 2. crop + background subtract
print("02-preprocess.png")
pl = Pipeline.from_json(open(PLJSON).read())
bg_idx = None
for i, op in enumerate(pl.operations):
    if isinstance(op, BgSubtractOp) and op.enabled:
        op.fit_with_progress(pl, i, win.sequence_manager, lambda d, t: None)
        bg_idx = i
win.pipeline = pl
win._invalidate_threshold_caches()
win.pipeline_panel.rebuild_list()
win._sync_filter_menu_actions()
win.update_frame_display()
win.view_box.autoRange()
win.tool_window.auto_contrast()
pump(600)

panel = win.pipeline_panel
panel.show()
panel.setGeometry(win.x() + win.width() - 400, win.y() + 60, 380, 720)
panel.list_widget.setCurrentRow(bg_idx)
pump(500)
shot(panel, "02-preprocess.png", marks=[
    (1, panel.list_widget),
    (2, panel.params_group),
    (3, panel.fit_btn),
], width=420)

# ------------------------------------------------- 3. threshold + overlay
print("03-threshold.png")
_, thresh = win._find_threshold_op()
thresh.enabled = True
win._sync_filter_menu_actions()
win.update_frame_display()
th_idx = win.pipeline.operations.index(thresh)
panel.list_widget.setCurrentRow(th_idx)
pump(600)
pw = panel.params_scroll.widget()
# "l" (outside left, vertically centred) is the only placement that is
# unambiguous on a QFormLayout: a badge above a row reads as belonging to the
# row above it.
shot(panel, "03-threshold-params.png", marks=[
    (1, pw._widgets["block_size"], "l"),
    (2, pw._widgets["C"], "l"),
    (3, pw._widgets["polarity"], "l"),
], width=420)
panel.hide()
win.view_box.autoRange()
pump(400)
shot(win, "03-threshold.png", marks=[
    (1, win.glw),
    (2, win.pipeline_summary_label, "tl"),
])

# --------------------------------------------------- 4. blob size analysis
print("04-blobsize.png")
bs = win.blob_size_window
bs.show()
bs.resize(760, 560)
pump(400)
bs.all_check.setChecked(True)
bs.log_y_check.setChecked(True)
bs._run()
wait_for(lambda: bs._worker is None and bs._areas)
pump(600)
shot(bs, "04-blobsize.png", marks=[
    (1, bs.stage_combo),
    (2, bs.plot_widget),
    (3, bs.stats_label),
], width=820)
bs.hide()

# ----------------------------------------------------- 5. blob size filter
print("05-blobfilter.png")
bf = BlobSizeFilterOp()
bf.params["min_area"] = 2
win.pipeline.add_operation(bf, th_idx + 1)
win._invalidate_threshold_caches()
panel.rebuild_list()
panel.show()
panel.list_widget.setCurrentRow(th_idx + 1)
win.update_frame_display()
pump(600)
bpw = panel.params_scroll.widget()
shot(panel, "05-blobfilter.png", marks=[
    (1, panel.list_widget),
    (2, bpw._widgets["min_area"], "l"),
    (3, bpw._widgets["max_area"], "l"),
], width=420)
panel.hide()

# ------------------------------------------------------ 6. pixel scale
print("06-scale.png")


def shot_scale_dialog():
    """The dialog blocks in exec(), so grab it from a timer callback."""
    dlg = app.activeModalWidget()
    if dlg is None:
        return
    # No badges: the dialog is three self-explanatory fields, and it is too
    # narrow to place a badge without covering one of them.
    shot(dlg, "06-scale.png", width=420)
    dlg.reject()


QTimer.singleShot(900, shot_scale_dialog)
win.show_scale_dialog()
pump(300)
# The dialog was cancelled after capture; set the calibration directly.
win.pipeline.scale = {"px": 672.875, "value": 1.0, "unit": "mm"}
win._refresh_status_label()
pump(200)

# --------------------------------------------------- 7. Mie, before Run
print("07-mie-setup.png")
m = win.mie_window
m.show()
m.resize(1120, 760)
pump(500)
m.all_check.setChecked(True)
m.n_real.setText("1.544")
m.n_imag.setText("0.0")
m.lam.setText("527")
m.num_a.setText("4000")
pump(400)
# Captured in the genuine first-open state: no areas yet, sweep fields empty.
# Run collects the areas and seeds the sweep itself, so the cost hint is only
# populated afterwards — it gets its callout on the results shot instead.
shot(m, "07-mie-setup.png", marks=[
    (1, m.source_combo, "tr"),
    (2, m.frames_row_host, "tr"),
    (3, m.n_real.parentWidget(), "tr"),
    (4, m.num_a, "tr"),
    (5, m.run_btn),
], width=1000)

# ------------------------------------------------------ 8. Mie results
print("08-mie-results.png")
m._run()
wait_for(lambda: m._result is not None and m._worker is None
         and m._mie_worker is None)
pump(800)
shot(m, "08-mie-results.png", marks=[
    (1, m.card_d50),
    (2, m.card_fit),
    (3, m.csca_plot),
    (4, m.cost_lbl),
    (5, m.cdf_plot),
], width=1000)

# ------------------------------------------------- 9. size distribution
print("09-mie-histogram.png")
m.lower_mode.setCurrentIndex(1)   # "Size distribution (histogram)"
pump(600)
shot(m, "09-mie-histogram.png", marks=[
    (1, m.lower_mode),
    (2, m.cdf_plot),
], width=1000)

r = m._result
print(f"\nD10={r.D10 * 1e6:.3f} µm  D50={r.D50 * 1e6:.3f} µm  "
      f"D90={r.D90 * 1e6:.3f} µm  n={len(r.radii)}")
print("done")
