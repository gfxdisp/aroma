"""Colour correction, Section 7.

Tone mapping that compresses contrast also desaturates -- or, when contrast is
expanded in the dark tones to fight ambient light, oversaturates (the
unrealistic reddish skin tone of Fig. 7).  We use the method of Mantiuk et
al. [2009], adapted to work with an arbitrary tone curve.
"""

from __future__ import annotations

import numpy as np
import torch

# Mantiuk, Mantiuk, Tomaszewska and Heidrich 2009, "Color correction for tone
# mapping": the empirical fit of the saturation exponent to the contrast
# compression factor k.  
_K1 = 1.6774
_K2 = 0.9925


def saturation_exponent(k):
    """s(k) = ((1 + k1) k^k2) / (1 + k1 k^k2).

    s(1) = 1 (no change).  k < 1 (contrast compressed) gives s < 1, which
    reduces saturation; k > 1 gives s > 1.
    """
    if isinstance(k, torch.Tensor):
        kp = k.clamp_min(1e-4) ** _K2
        return ((1.0 + _K1) * kp) / (1.0 + _K1 * kp)
    kp = np.maximum(np.asarray(k, dtype=np.float64), 1e-4) ** _K2
    return ((1.0 + _K1) * kp) / (1.0 + _K1 * kp)


def smooth_slopes(slopes: np.ndarray, taps: int = 21) -> np.ndarray:
    """Section 7: "smooth the slopes (contrast values) across segments using a
    21-tap triangle filter", so saturation does not jump between segments."""
    if slopes.size < 2:
        return slopes.copy()

    half = taps // 2
    kernel = np.concatenate([np.arange(1, half + 2), np.arange(half, 0, -1)]).astype(
        np.float64
    )
    kernel /= kernel.sum()

    padded = np.pad(slopes.astype(np.float64), half, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def apply(
    l_c: torch.Tensor,
    m: torch.Tensor,
    m_prime: torch.Tensor,
    s_per_pixel: torch.Tensor,
) -> torch.Tensor:
    """Eq. (36): l'_c = (l_c - m) . s(d_t(m)) + m'.

    This is Eq. (35) of Mantiuk et al. [2009] rewritten in the logarithmic
    domain: ``l_c - m`` is the log colour ratio, scaled by the saturation
    exponent and re-centred on the tone-mapped, detail-enhanced tone value.
    """
    return (l_c - m.unsqueeze(0)) * s_per_pixel.unsqueeze(0) + m_prime.unsqueeze(0)
