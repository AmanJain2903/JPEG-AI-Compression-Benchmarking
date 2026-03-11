# JPEG-AI · Image Compression Benchmark

A comprehensive benchmarking tool comparing **JPEG-AI Compression** (CompressAI `mbt2018-mean`) against traditional **JPEG** codec. Evaluates rate-distortion performance via BPP, PSNR, and MS-SSIM, and measures machine-vision accuracy degradation using **YOLOv8** object detection.

## Project Structure

```
JPEG-AI/
├── app.py              # Streamlit dashboard (UI)
├── engine.py           # Compression codecs + MetricsEngine
├── vision.py           # YOLOv8 inference & visualization
├── requirements.txt    # Python dependencies
├── TestImages          # COCO 2017 Dataset Validation Set Images
└── README.md
```

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch will be installed automatically. If you need GPU support, install the CUDA-enabled build first — see [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Run the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

1. **Upload** any image (PNG, JPEG, BMP, TIFF, WebP).
2. **Adjust** quality sliders in the sidebar for each codec.
3. **Compare** — use the dual-view slider to visually inspect artifacts.
4. **Analyze** the results table for BPP vs. PSNR vs. YOLO confidence.
5. **Inspect** YOLO bounding-box overlays on each variant.

## Components

### `engine.py`

| Function / Class | Purpose |
|---|---|
| `compress_neural()` | CompressAI `mbt2018-mean` codec (quality 1-8) |
| `compress_jpeg()` | Standard JPEG via Pillow (quality 1-95) |
| `MetricsEngine` | Computes BPP, PSNR (dB), and MS-SSIM |

### `vision.py`

| Function / Class | Purpose |
|---|---|
| `run_detection()` | YOLOv8-nano inference on a single image |
| `compare_detections()` | Batch detection across original + all variants |
| `draw_detections()` | Annotate an image with bounding boxes and labels |

### `app.py`

Streamlit frontend with sidebar controls, image upload, dual-view comparison (`streamlit-image-comparison`), metrics table, and YOLO visualization grid.

## Key Dependencies

| Package | Role |
|---|---|
| `compressai` | Neural image compression models |
| `torch` / `torchvision` | Deep learning backbone |
| `ultralytics` | YOLOv8 object detection |
| `streamlit` | Dashboard UI |
| `streamlit-image-comparison` | Side-by-side image slider |
| `pytorch-msssim` | MS-SSIM metric |
| `Pillow` | Image I/O and traditional codecs |

## License

Academic project — CSEN 338, Santa Clara University.
