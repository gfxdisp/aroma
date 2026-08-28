"""Tone importance, Sections 5.2 and 5.3.

The importance p_k weights each tone-curve segment in the objective of
Eq. (20).  It is a histogram of tone values weighted by the local contrast of
the frame, so that flat regions (large but perceptually unimportant areas) do
not dominate the contrast budget.
"""

from __future__ import annotations

import numpy as np
import torch

from .pyramid import build_mip, pad_to_multiple, upsample


def local_contrast(m: torch.Tensor, level: int = 5) -> torch.Tensor:
    """Eq. (21)/(22): local contrast as a windowed standard deviation,
    approximated from MIP-map levels of m and m^2::

        c(x, y) ~ sqrt(M_2^(n)(x, y) - (M_1^(n)(x, y))^2)

    ``m`` is (H, W); the result is (H, W) at full resolution.
    """
    # Eq. (22) subtracts two nearly equal numbers, which in float32 leaves
    # spurious contrast in flat regions (m^2 ~ 36 for a dark tone, while the
    # true variance is ~0).  Centring on the mean first is algebraically
    # identical and removes the cancellation.
    mu = m.mean()
    mc = m - mu

    stacked = torch.stack((mc, mc * mc), dim=0)  # M_1 and M_2 channels
    padded, h, w = pad_to_multiple(stacked, 2**level)
    coarse = build_mip(padded, level)[level]
    coarse_up = upsample(coarse, padded.shape[-2:])[:, :h, :w]

    var = coarse_up[1] - coarse_up[0] ** 2
    return var.clamp_min(0.0).sqrt()


def tone_importance(
    m: torch.Tensor,
    *,
    bins: int,
    y_smin: float,
    level: int = 5,
    mask: torch.Tensor | None = None,
) -> np.ndarray:
    """The p_k of Eq. (14)/(20), one weight per tone-curve segment.

    Section 5.2 defines the local contrast c(x, y) but never states how it
    enters the histogram; we accumulate c(x, y) as the weight of the bin
    containing m(x, y), which is the reading that makes Eq. (21) a "histogram
    weighted by the local contrast of the frame".

    Section 10 additionally mentions "32x32 zonal tiles (stride 32) with
    per-tile weighting"; that weighting rule is never defined and reads as a
    GPU tiling strategy, so we compute a single full-image histogram instead.

    ``mask`` marks pixels to include (Section 8 excludes transparent, i.e.
    zero-valued, pixels from all analysis).
    """
    c = local_contrast(m, level=level)

    lo = float(np.log2(y_smin))
    idx = ((m - lo) / (0.0 - lo) * bins).floor().long().clamp(0, bins - 1)

    weights = c
    if mask is not None:
        weights = torch.where(mask, weights, torch.zeros_like(weights))

    p = torch.zeros(bins, dtype=torch.float64, device=m.device)
    p.scatter_add_(0, idx.reshape(-1), weights.reshape(-1).double())

    total = p.sum()
    if total <= 0:
        p = torch.full_like(p, 1.0 / bins)
    else:
        p = p / total

    return shape_importance(p.cpu().numpy())


def shape_importance(p: np.ndarray) -> np.ndarray:
    """Eq. (23): fractional exponents reduce the spread between bins.

        p'_k = p_k^(1/2)  for k = 1..K-5
        p'_k = p_k^(1/4)  for k = K-4..K

    The 4th root on the top five bins keeps them important, which reduces
    clipping once details are enhanced (Section 6).  The paper does not say
    whether to renormalise afterwards; we do, so that lambda in Eq. (20) keeps
    a consistent meaning across frames.
    """
    p = np.asarray(p, dtype=np.float64)
    out = np.power(p, 0.5)
    if p.size >= 5:
        out[-5:] = np.power(p[-5:], 0.25)
    total = out.sum()
    return out / total if total > 0 else np.full_like(out, 1.0 / out.size)
