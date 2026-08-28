"""MIP-map helpers shared by Sections 5.2 and 6.

The paper builds a MIP-map of the tone values m (and of m^2, and later of the
tone-mapped values t(m)) and reads level n = 5, i.e. 32x downsampling.  The
downsampling filter is not specified; we use the 2x2 box average that a
hardware MIP-map chain performs.  Non-power-of-two inputs are also not
discussed; we replicate-pad the image up to a multiple of 2^n, build the
pyramid on the padded image, and crop back after upsampling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    """Replicate-pad ``x`` (..., H, W) so H and W are multiples of ``multiple``.

    Returns the padded tensor and the original (H, W).
    """
    h, w = x.shape[-2:]
    ph = (-h) % multiple
    pw = (-w) % multiple
    if ph or pw:
        x = F.pad(x.unsqueeze(0) if x.dim() == 3 else x, (0, pw, 0, ph), mode="replicate")
        x = x.squeeze(0) if x.dim() == 4 and x.shape[0] == 1 else x
    return x, h, w


def build_mip(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """2x2 box-average pyramid.  ``x`` is (C, H, W); returns levels 0..levels."""
    out = [x]
    cur = x.unsqueeze(0)
    for _ in range(levels):
        cur = F.avg_pool2d(cur, kernel_size=2, stride=2)
        out.append(cur.squeeze(0))
    return out


def upsample(coarse: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear enlargement back to full resolution (Section 5.2/6)."""
    single = coarse.dim() == 2
    x = coarse.view(1, 1, *coarse.shape) if single else coarse.unsqueeze(0)
    y = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return y.view(size) if single else y.squeeze(0)
