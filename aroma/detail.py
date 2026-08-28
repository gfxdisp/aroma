"""Contrast-preserving detail enhancement, Section 6.

Ambient light and the global tone curve both cost local contrast.  This stage
splits the tone-mapped image into a base and a detail layer and boosts the
detail layer by exactly the factor needed to undo that loss, capped at g_max.
"""

from __future__ import annotations

import torch

from .params import AromaParams
from .pyramid import build_mip, pad_to_multiple


def range_kernel(d: torch.Tensor, sigma: float) -> torch.Tensor:
    """The h_r of Eq. (34), sketched in Fig. 6: a triangle that decays from 1
    to a floor of 0.5.
    """
    return torch.clamp(1.0 - d.abs() / sigma, min=0.5)


def edge_stopping_upsample(
    fine: torch.Tensor, coarse: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Eq. (34): bilinear upsampling whose weights are modulated by a range
    kernel, giving a cheap bilateral filter that reuses the MIP-map.

        w_uv = h_s(x-u, y-v) . h_r(fine(x,y) - coarse(u,v))
        base = sum(w_uv . coarse(u,v)) / sum(w_uv)

    ``fine`` is (Hp, Wp), ``coarse`` is (h, w) with Hp = h * 2^n.  Omega is the
    four texels used for bilinear interpolation.

    NOTE: Eq. (34) as printed evaluates the range kernel on
    ``m(x, y) - M_t^(n)(u, v)``, i.e. the *source* tone value against the
    *tone-mapped* coarse level.  Those live in different domains, so their
    difference is not an edge measure.  We pass t(m) as ``fine``, which is the
    only reading under which h_r measures what Section 6.1 says it does.
    """
    Hp, Wp = fine.shape
    h, w = coarse.shape
    sy, sx = Hp / h, Wp / w

    dev = fine.device
    v = (torch.arange(Hp, device=dev, dtype=fine.dtype) + 0.5) / sy - 0.5
    u = (torch.arange(Wp, device=dev, dtype=fine.dtype) + 0.5) / sx - 0.5

    v0 = v.floor()
    u0 = u.floor()
    fv = (v - v0).view(-1, 1)
    fu = (u - u0).view(1, -1)

    v0i = v0.long().clamp(0, h - 1)
    v1i = (v0.long() + 1).clamp(0, h - 1)
    u0i = u0.long().clamp(0, w - 1)
    u1i = (u0.long() + 1).clamp(0, w - 1)

    num = torch.zeros_like(fine)
    den = torch.zeros_like(fine)
    for vi, wv in ((v0i, 1.0 - fv), (v1i, fv)):
        for ui, wu in ((u0i, 1.0 - fu), (u1i, fu)):
            c = coarse[vi][:, ui]                      # (Hp, Wp) gathered texels
            weight = (wv * wu) * range_kernel(fine - c, sigma)
            num = num + weight * c
            den = den + weight

    return num / den.clamp_min(1e-8)


def enhance(
    t_m: torch.Tensor,
    dtdm: torch.Tensor,
    params: AromaParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. (25)-(27) and (31)-(33).  Returns (m', gain map).

    ``t_m`` is the tone-mapped tone value t(m(x,y)) at full resolution and
    ``dtdm`` the tone-curve slope at each pixel (Eq. 30).
    """
    padded, H, W = pad_to_multiple(t_m.unsqueeze(0), 2**params.mip_level)
    padded = padded.squeeze(0)

    # Eq. (25): the base layer is the n-th MIP level of the tone-mapped values,
    # upsampled with the edge-stopping filter of Section 6.1.
    coarse = build_mip(padded.unsqueeze(0), params.mip_level)[params.mip_level][0]
    m_base = edge_stopping_upsample(padded, coarse, params.sigma_range)[:H, :W]

    m_detail = t_m - m_base  # Eq. (26)

    # Eq. (32): absolute luminance of the tone-mapped image.
    Y = params.Y_dmax * torch.exp2(t_m)

    # Eq. (31): the gain that restores the contrast lost to ambient light and
    # to tone-curve compression.
    g_u = (Y + params.Y_add ) / (Y * dtdm.clamp_min(1e-3))

    g = g_u.clamp(max=params.g_max)  # Eq. (33)

    return m_base + g * m_detail, g  # Eq. (27)
