# tiffscope

**Fast scrubbing viewer and preprocessing pipeline for scientific TIFF image sequences.**

Built for looking at *thousands* of frames quickly: high-speed camera output, PIV/PTV particle images, SEM series, time-lapse microscopy — anything that lands on disk as a folder of grayscale TIFFs. Scrub with arrow keys at interactive speed, build a non-destructive processing pipeline, and export binary masks ready for downstream analysis.

![tiffscope demo](docs/demo.gif)

## Why not ImageJ / Fiji?

tiffscope does one thing fast instead of everything slowly:

- **Instant scrubbing** through large sequences (sliding RAM buffer, lazy loading, precomputed display LUT — no full-stack load, no waiting).
- **Raw pixel fidelity** — 12/16-bit values shown unscaled in the pixel probe; display contrast/gamma never touches the data.
- **Live, reorderable pipeline** — every operation previews on the current frame as you tune parameters, at full scrub speed.

## Features

- **Lazy TIFF sequence loading** — opens folders with thousands of 12/16-bit frames instantly; only a ±20-frame window lives in RAM.
- **Non-destructive processing pipeline** — add, reorder, enable/disable operations live:
  - Rotate, Crop (interactive ROI)
  - Background subtraction (median/mean of sampled frames)
  - CLAHE, Gaussian blur, sharpen, low-pass, high-pass
  - Adaptive thresholding with live overlay
  - Binary mask chain: morphology (erode/dilate/open/close/…), median smoothing, watershed splitting of merged blobs
  - Intensity-guided watershed: splits touching particles using upstream grayscale peaks and reconstructs per-particle circles
- **Pipelines save/load as JSON** — reproducible preprocessing, shareable with collaborators.
- **Mask export** — write the full binary chain out as 0/255 uint8 TIFFs for your PIV/PTV/tracking software.
- **Analysis tools**
  - Blob size histogram — pick size thresholds empirically from the actual data
  - Region props (area, equivalent diameter, axes, eccentricity) with physical-unit conversion and CSV export
  - Live optical flow overlay (Farnebäck) for a qualitative look at motion before running full analysis
- **Measurement** — pixel-to-physical scale calibration, horizontal/vertical measurement rays.
- **Performance monitor** — per-operation timing of the live pipeline (`Ctrl+Shift+M`).

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/tiffscope/tiffscope.git
cd tiffscope
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart

```bash
python main.py
```

1. `Ctrl+O` → pick any TIFF inside your sequence folder — the whole folder loads as a sequence (natural sort order).
2. Scrub with `←` / `→`.
3. `Ctrl+P` opens the pipeline panel — add operations, drag to reorder, tune parameters with live preview.
4. `Ctrl+T` for contrast/gamma (display only — data is untouched).
5. Add an *Adaptive Threshold* op to get a live binary overlay; "Apply to All" caches masks for the whole sequence; "Save Masks…" exports them as TIFFs.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / next frame |
| `Ctrl+O` | Open sequence |
| `Ctrl+P` | Pipeline panel |
| `Ctrl+T` | Contrast & histogram |
| `Ctrl+F` | Optical flow overlay |
| `Ctrl+R` | Rotate 90° CW |
| `Ctrl+0` | Reset zoom/pan |
| `Ctrl++` / `Ctrl+-` | Zoom in / out |
| `Ctrl+Shift+C` | Toggle crop mode |
| `Ctrl+Shift+X` | Clear crop |
| `Enter` / `Esc` | Confirm / cancel ROI |
| `H` / `V` | Place horizontal / vertical measurement ray |
| `Ctrl+Shift+M` | Performance monitor |

## Typical workflows

**PIV/PTV preprocessing** — rotate → crop to ROI → background subtract → save cropped, cleaned frames or thresholded particle masks for your PIV software.

**Particle sizing** — adaptive threshold → morphology cleanup → intensity watershed to split touching particles → region props with physical units → CSV.

**Just looking** — open a folder, scrub, probe raw pixel values, drop measurement rays. Sometimes that's all you need.

## Architecture

Four files, no packages: `image_engine.py` (I/O + pixel math), `operations.py` (processing ops), `pipeline.py` (ordering/caching/serialization), `main.py` (PyQt6 + pyqtgraph GUI). The processing layers have zero GUI imports and are usable headless.

## Roadmap

- Variable pre-cache size and smarter buffering
- Rolling-ball background subtraction
- Scale-image overlay for calibration embedding
- Faster peak detection in intensity watershed (pure-OpenCV NMS, dropping the scikit-image dependency)

Issues and PRs welcome.

## License

[MIT](LICENSE). Note that [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) (a dependency you install separately) is GPL-licensed; this affects redistribution of bundled binaries, not use of this source code.
