"""Regenerate the README media (GIFs + stills) deterministically.

No screen recorder involved: the app is driven programmatically and each frame
is captured with QWidget.grab(), then assembled by ffmpeg's palettegen /
paletteuse pair. That means the assets are reproducible — rerun it after a UI
change instead of re-recording by hand.

Usage:
    python make_media.py <sequence-folder> <pipeline.json> <out-dir>
"""
import os
import shutil
import subprocess
import sys
import glob

if len(sys.argv) != 4:
    sys.exit(__doc__)
SEQ, PLJSON, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
# repo root is this file's parent — keep it path-independent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(OUT, exist_ok=True)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QEventLoop

import main as tv
from pipeline import Pipeline
from operations import BgSubtractOp, AdaptiveThresholdOp

app = QApplication(sys.argv)
app.setStyleSheet(tv.modern_stylesheet)


def pump(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def gif(frames_dir, out_name, fps=12, width=900):
    """frames/%04d.png -> optimised gif via a generated 128-colour palette.

    The frame rate is set with -framerate on the INPUT. An `fps=` filter would
    resample against ffmpeg's default 25fps assumption for image sequences and
    silently drop half the captured frames.

    To inspect the result, ALWAYS coalesce first:
        convert out.gif -coalesce frame_%02d.png
    GIF stores frames after the first as deltas, so pulling `out.gif[8]` shows
    only the changed pixels — which on a moving particle field looks exactly
    like severe dithering noise and will send you chasing a bug that is not
    there.
    """
    pal = os.path.join(frames_dir, "palette.png")
    vf = f"scale={width}:-1:flags=lanczos"
    rate = ["-framerate", str(fps)]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *rate, "-i", f"{frames_dir}/%04d.png",
         "-vf", f"{vf},palettegen=max_colors=128:stats_mode=diff", pal],
        check=True)
    out = os.path.join(OUT, out_name)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *rate, "-i", f"{frames_dir}/%04d.png",
         "-i", pal, "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
         "-loop", "0", out],
        check=True)
    mb = os.path.getsize(out) / 1e6
    print(f"  {out_name}: {mb:.2f} MB")
    return out


def shot(widget, name, width=None):
    widget.repaint()
    pump(250)
    path = os.path.join(OUT, name)
    pm = widget.grab()
    if width:
        from PyQt6.QtCore import Qt as _Qt
        pm = pm.scaledToWidth(width, _Qt.TransformationMode.SmoothTransformation)
    pm.save(path)
    print(f"  {name}: {os.path.getsize(path) / 1e3:.0f} kB")


# ---------------------------------------------------------------- setup
win = tv.TiffViewerApp()
win.resize(1180, 780)
win.show()
pump(600)
win.load_and_display_data(sorted(glob.glob(os.path.join(SEQ, "*.TIF")))[0])
pump(1200)

pl = Pipeline.from_json(open(PLJSON).read())
for i, op in enumerate(pl.operations):
    if isinstance(op, BgSubtractOp) and op.enabled:
        op.fit_with_progress(pl, i, win.sequence_manager, lambda d, t: None)
win.pipeline = pl
# Dataset-specific calibration, only needed so the Mie still has real units.
win.pipeline.scale = {"px": 672.875, "value": 1.0, "unit": "mm"}
win._invalidate_threshold_caches()
win.pipeline_panel.rebuild_list()
win._sync_filter_menu_actions()
win.update_frame_display()
win.view_box.autoRange()
pump(800)

# Display range deliberately left at the full bit depth. This camera has a
# real noise floor just above the background, so ANY contrast stretch (auto
# contrast lands near vmax=740) pulls it up into full-frame speckle — accurate,
# but it reads as a broken image and, being unique per frame, destroys GIF
# inter-frame compression (14 MB vs ~2 MB).

_, thresh = win._find_threshold_op()

# ------------------------------------------------- 1. scrub + overlay demo
print("demo.gif")
tmp = os.path.join(OUT, "_frames_demo")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
n = 0


def cap():
    global n
    win.repaint()
    pump(40)
    win.grab().save(f"{tmp}/{n:04d}.png")
    n += 1


thresh.enabled = True
win._sync_filter_menu_actions()
win.update_frame_display()
pump(200)
for f in list(range(0, 41, 2)) + list(range(38, 0, -2)):
    win._goto_frame(f)
    cap()
gif(tmp, "demo.gif", fps=12, width=860)
shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------- 2. live param tuning demo
print("tuning.gif")
win.pipeline_panel.show()
win.pipeline_panel.setGeometry(win.x() + 40, win.y() + 40, 380, 720)
idx = win.pipeline.operations.index(thresh)
win.pipeline_panel.list_widget.setCurrentRow(idx)
pump(500)
pw = win.pipeline_panel.params_scroll.widget()
spin = pw._widgets["block_size"]

tmp = os.path.join(OUT, "_frames_tune")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
n = 0
win._goto_frame(12)
pump(300)
# sweep block size up and back down; every step previews live
seq = list(range(11, 60, 5)) + list(range(56, 10, -5))
for v in seq:
    spin.setValue(v)
    pump(120)
    cap()
gif(tmp, "tuning.gif", fps=7, width=760)
shutil.rmtree(tmp, ignore_errors=True)
win.pipeline_panel.hide()

# ------------------------------------------------------ 3. Mie still
print("mie.png")
m = win.mie_window
m.show()
m.resize(1100, 720)
m.n_real.setText("1.544")
m.n_imag.setText("0.0")
m.num_a.setText("4000")
m.all_check.setChecked(True)
m._run()
for _ in range(600):
    pump(500)
    if m._result is not None and m._worker is None and m._mie_worker is None:
        break
pump(700)
shot(m, "mie.png", width=1000)
print("done")
