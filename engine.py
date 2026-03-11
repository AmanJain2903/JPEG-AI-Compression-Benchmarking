"""
engine.py — Compression & Metrics Engine
=========================================
Provides JPEG and JPEG-AI (Neural) compression and a unified MetricsEngine
that evaluates BPP, PSNR, and MS-SSIM for any codec output.

Neural compression uses CompressAI's ``mbt2018-mean`` pretrained model,
which mirrors the architecture foundations of JPEG AI.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ms_ssim
from pytorch_msssim import ssim

from compressai.zoo import mbt2018_mean

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompressionResult:
    """Immutable container returned by every codec."""

    codec: str
    reconstructed: Image.Image
    compressed_bytes: int
    width: int
    height: int


@dataclass(frozen=True)
class MetricsResult:
    """Quality / rate metrics for a single compression result."""

    codec: str
    bpp: float
    psnr_db: float
    ms_ssim_val: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a float32 tensor of shape (1, 3, H, W) in [0, 1]."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, 3, H, W) float tensor back to a PIL RGB image."""
    arr = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def _pad_to_multiple(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad spatial dims of (1, C, H, W) tensor to the next multiple. Returns padded tensor and original (H, W)."""
    _, _, h, w = tensor.shape
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    padded = torch.nn.functional.pad(tensor, (0, new_w - w, 0, new_h - h), mode="reflect")
    return padded, (h, w)

# ---------------------------------------------------------------------------
# Codec: Standard JPEG (Pillow)
# ---------------------------------------------------------------------------

def compress_jpeg(img: Image.Image, quality: int = 50) -> CompressionResult:
    """
    Compress *img* with Pillow's JPEG encoder at the given quality (1-95).

    Round-trips through an in-memory buffer so the reconstructed image
    reflects real JPEG artifacts.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    compressed_bytes = buf.tell()

    buf.seek(0)
    reconstructed = Image.open(buf).copy()

    return CompressionResult(
        codec=f"JPEG (q={quality})",
        reconstructed=reconstructed,
        compressed_bytes=compressed_bytes,
        width=img.width,
        height=img.height,
    )


# ---------------------------------------------------------------------------
# Codec: Neural (CompressAI mbt2018-mean)
# ---------------------------------------------------------------------------

_neural_cache: dict[int, torch.nn.Module] = {}

def _get_neural_model(quality: int) -> torch.nn.Module:
    """Load (and cache) the pretrained mbt2018-mean model for a given quality level."""
    if quality not in _neural_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = mbt2018_mean(quality=quality, pretrained=True).eval().to(device)
        net.update(force=True)
        _neural_cache[quality] = net
    return _neural_cache[quality]


def compress_neural(img: Image.Image, quality: int = 4) -> CompressionResult:
    """
    Compress *img* with the mbt2018-mean neural codec at the given quality
    level (1-8, mapping to CompressAI lambda points).

    Returns a ``CompressionResult`` with the reconstructed image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _get_neural_model(quality)

    x = _pil_to_tensor(img).to(device)
    x_padded, (orig_h, orig_w) = _pad_to_multiple(x, 64)

    with torch.no_grad():
        out = net.compress(x_padded)

    compressed_size = sum(len(s[0]) for s in out["strings"])

    with torch.no_grad():
        rec = net.decompress(out["strings"], out["shape"])

    x_hat = rec["x_hat"][:, :, :orig_h, :orig_w]
    reconstructed = _tensor_to_pil(x_hat)

    return CompressionResult(
        codec=f"JPEG-AI (q={quality})",
        reconstructed=reconstructed,
        compressed_bytes=compressed_size,
        width=orig_w,
        height=orig_h,
    )


# ---------------------------------------------------------------------------
# MetricsEngine
# ---------------------------------------------------------------------------

class MetricsEngine:
    """Compute BPP, PSNR, and MS-SSIM between an original image and a codec output."""

    @staticmethod
    def bpp(result: CompressionResult) -> float:
        """Bits per pixel"""
        return (result.compressed_bytes * 8) / (result.width * result.height)

    @staticmethod
    def psnr(original: Image.Image, reconstructed: Image.Image) -> float:
        """Peak Signal-to-Noise Ratio (dB) between two PIL images."""
        orig = np.asarray(original.convert("RGB")).astype(np.float64)
        recon = np.asarray(reconstructed.convert("RGB")).astype(np.float64)
        mse = np.mean((orig - recon) ** 2)
        if mse == 0:
            return float("inf")
        return 10.0 * math.log10(255.0**2 / mse)

    @staticmethod
    def compute_ms_ssim(original: Image.Image, reconstructed: Image.Image) -> float:
        """MS-SSIM in [0, 1] using ``pytorch_msssim``.

        Falls back to SSIM when either dimension is below 160 px
        """
        orig_t = _pil_to_tensor(original)
        recon_t = _pil_to_tensor(reconstructed.resize(original.size, Image.LANCZOS))
        min_side = min(original.size)
        if min_side < 160:
            return ssim(orig_t, recon_t, data_range=1.0, size_average=True).item()
        return ms_ssim(orig_t, recon_t, data_range=1.0, size_average=True).item()

    @classmethod
    def evaluate(cls, original: Image.Image, result: CompressionResult) -> MetricsResult:
        """Run all metrics and return a ``MetricsResult``."""
        return MetricsResult(
            codec=result.codec,
            bpp=cls.bpp(result),
            psnr_db=cls.psnr(original, result.reconstructed),
            ms_ssim_val=cls.compute_ms_ssim(original, result.reconstructed),
        )
