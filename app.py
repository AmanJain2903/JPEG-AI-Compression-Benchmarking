"""
app.py — Streamlit Dashboard for JPEG-AI Image Compression Benchmarking
================================================================
Run with:  ``streamlit run app.py``
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

from engine import (
    CompressionResult,
    MetricsEngine,
    compress_jpeg,
    compress_neural,
)
from vision import compare_detections, draw_detections

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="JPEG-AI VS JPEG Compression Benchmark",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .stMetric {background: #f8f9fa; border-radius: 0.5rem; padding: 0.75rem;}
    h1 {font-size: 1.8rem !important;}
    h2 {font-size: 1.3rem !important; border-bottom: 2px solid #4e8df5; padding-bottom: 0.3rem;}
    /* Make image comparison use full available width */
    [data-testid="stImageComparison"] { width: 100% !important; max-width: 100% !important; }
    [data-testid="stImageComparison"] iframe { width: 100% !important; max-width: 100% !important; }
    div[data-testid="stImageComparison"] { width: 100% !important; }
    /* Tighten space between image comparison panel and metrics table */
    div[data-testid="stVerticalBlock"]:has(iframe) { margin-bottom: 0 !important; padding-bottom: 0 !important; }
    .stFrame { margin-bottom: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Codec's Settings")

    st.subheader("Neural Codec (Mean-Scale Hyperprior)")
    neural_quality = st.slider(
        "Quality level",
        min_value=1,
        max_value=8,
        value=4,
        help="CompressAI quality parameter — higher means better quality / larger bitstream.",
    )

    st.divider()

    st.subheader("JPEG")
    jpeg_quality = st.slider(
        "JPEG quality",
        min_value=1,
        max_value=99,
        value=20,
        help="Pillow JPEG quality factor.",
    )

    st.divider()

    st.subheader("YOLO Object Detection Settings")
    yolo_conf = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum confidence for YOLO detections.",
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("JPEG-AI VS JPEG - Image Compression Benchmark")
st.caption(
    "Compare JPEG-AI (CompressAI mbt2018-mean) against standard JPEG "
    "on quality metrics and machine-vision accuracy."
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
)

if uploaded is None:
    st.info("Upload an image to begin the benchmark.")
    st.stop()

original = Image.open(uploaded).convert("RGB")

# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

st.divider()

st.header("Compression Results")

results: dict[str, CompressionResult] = {}

with st.spinner("Compressing with JPEG..."):
    results["jpeg"] = compress_jpeg(original, quality=jpeg_quality)

with st.spinner("Compressing with JPEG-AI (Neural) — this may take a moment..."):
    results["neural"] = compress_neural(original, quality=neural_quality)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

metrics_engine = MetricsEngine()
metrics = {k: metrics_engine.evaluate(original, v) for k, v in results.items()}

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

st.header("Metrics Table")

with st.spinner("Running YOLO on all variants..."):
    detection_results = compare_detections(
        original,
        {r.codec: r.reconstructed for r in results.values()},
        conf_threshold=yolo_conf,
    )

det_map = {dr.label: dr for dr in detection_results}

rows = []
for key, m in metrics.items():
    dr = det_map.get(results[key].codec)
    rows.append(
        {
            "Codec": m.codec,
            "BPP": f"{m.bpp:.4f}",
            "PSNR (dB)": f"{m.psnr_db:.2f}",
            "MS-SSIM": f"{m.ms_ssim_val:.4f}",
            "YOLO Detections": dr.num_detections if dr else "—",
            "YOLO Mean Conf": f"{dr.mean_confidence:.3f}" if dr else "—",
        }
    )

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Machine vision visualization
# ---------------------------------------------------------------------------

st.divider()
st.header("Machine Vision — YOLO Detections")
st.caption("Bounding boxes drawn on each image variant to visualize detection confidence shifts.")

vis_cols = st.columns(3)

variant_images = [
    ("Original", original),
    (results["jpeg"].codec, results["jpeg"].reconstructed),
    (results["neural"].codec, results["neural"].reconstructed),
]

for col, (label, img) in zip(vis_cols, variant_images):
    dr = det_map.get(label)
    if dr:
        annotated = draw_detections(img, dr)
    else:
        annotated = img
    with col:
        st.image(annotated, caption=f"{label}", use_container_width=True)
        if dr:
            st.caption(f"Detections: {dr.num_detections} | Mean confidence: {dr.mean_confidence:.3f}")

# ---------------------------------------------------------------------------
# Visual comparison (at end)
# ---------------------------------------------------------------------------

st.divider()

st.header("Visual Comparison")

col_left, col_right = st.columns(2)
with col_left:
    compare_a = st.selectbox("Left image", ["Original", "JPEG", "JPEG-AI (Neural)"], index=1, key="cmp_a")
with col_right:
    compare_b = st.selectbox("Right image", ["Original", "JPEG", "JPEG-AI (Neural)"], index=2, key="cmp_b")


def _resolve_image(label: str) -> Image.Image:
    mapping = {
        "Original": original,
        "JPEG": results["jpeg"].reconstructed,
        "JPEG-AI (Neural)": results["neural"].reconstructed,
    }
    return mapping[label]


image_comparison(
    img1=_resolve_image(compare_a),
    img2=_resolve_image(compare_b),
    label1=compare_a,
    label2=compare_b,
    width=original.width,
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "JPEG-AI Compression Benchmark · CSEN 338 · Santa Clara University"
    "</div>",
    unsafe_allow_html=True,
)
