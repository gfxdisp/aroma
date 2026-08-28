"""Colour processing: Sections 4.2 to 4.4 of the paper.

Pipeline for one frame, in order:

    display-encoded RGB -> linear relative -> absolute BT.709 (x Y_smax)
        -> native primaries (Eq. 4) -> logarithmic encoding (Eq. 5)
        -> tone values m (Eq. 8)
"""

from __future__ import annotations

import torch

# Eq. (7): the luminance weights the paper uses for the pseudo-luminance term.
# NOTE: as printed they sum to 1.0002, not 1, so a neutral grey shifts by
# ~0.02% when alpha = 0.  Kept verbatim rather than renormalised to BT.709's
# (0.2126, 0.7152, 0.0722); the error is far below one code value.
LUM_WEIGHTS = (0.213, 0.715, 0.0722)


# --------------------------------------------------------------------------
# Electro-optical transfer functions
# --------------------------------------------------------------------------
def srgb_eotf(v: torch.Tensor) -> torch.Tensor:
    """sRGB display-encoded [0,1] -> linear [0,1]. Used for still images."""
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def srgb_oetf(v: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`srgb_eotf`."""
    v = v.clamp(0.0, 1.0)
    return torch.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1.0 / 2.4) - 0.055)


def gamma_eotf(v: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """Pure power-law EOTF. Used for video (see README)."""
    return v.clamp_min(0.0) ** gamma


def gamma_oetf(v: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    return v.clamp(0.0, 1.0) ** (1.0 / gamma)


# --------------------------------------------------------------------------
# Section 4.2 - native colour space
# --------------------------------------------------------------------------
def to_native(I_bt709: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
    """Eq. (4): I_native = M_BT709-to-native . I_BT709.

    ``M`` is the 3x3 matrix built from the panel's native primaries and white
    point.  The paper requires it but a generic CLI has no way to know the
    panel, so the identity (i.e. a BT.709 display) is the default.
    """
    if M is None:
        return I_bt709
    return torch.einsum("ij,jhw->ihw", M.to(I_bt709), I_bt709)


def from_native(I_native: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
    """Inverse of :func:`to_native`."""
    if M is None:
        return I_native
    return torch.einsum("ij,jhw->ihw", torch.linalg.inv(M).to(I_native), I_native)


# --------------------------------------------------------------------------
# Section 4.3 - logarithmic encoding
# --------------------------------------------------------------------------
def log_encode(I_native: torch.Tensor, Y_smax: float, y_smin: float) -> torch.Tensor:
    """Eq. (5): l_c = log2((1 - y_smin) * I / Y_smax + y_smin).

    Returns relative log values in [log2(y_smin), 0].
    """
    rel = (1.0 - y_smin) * I_native / Y_smax + y_smin
    return torch.log2(rel.clamp_min(y_smin * 1e-3))


def log_decode(l_c: torch.Tensor, y_smin: float) -> torch.Tensor:
    """Eq. (11): i'_c = (2^l'_c - y_smin) / (1 - y_smin)."""
    return (torch.exp2(l_c) - y_smin) / (1.0 - y_smin)


# --------------------------------------------------------------------------
# Section 4.4 - tone values
# --------------------------------------------------------------------------
def tone_value(l: torch.Tensor, y_smin: float) -> torch.Tensor:
    """Eq. (8): a blend of max-RGB and pseudo-luminance.

        m = alpha * max{l} + (1 - alpha) * (w_R l_R + w_G l_G + w_B l_B)
        alpha = clip(max{l} / (y_smax - y_smin) + 1, 0, 1)

    The paper writes the denominator as "the dynamic range of the input
    signal", which in the logarithmic domain of Eq. (5) is
    ``0 - log2(y_smin)``.  alpha is then 1 for the brightest tone (max-RGB,
    which avoids clipping and keeps colours punchy) and falls to 0 at the
    darkest (luminance, which avoids the "glow" of grey tones).
    """
    l_max = l.max(dim=0).values
    w = torch.tensor(LUM_WEIGHTS, dtype=l.dtype, device=l.device).view(3, 1, 1)
    l_lum = (w * l).sum(dim=0)

    log_range = -torch.log2(torch.tensor(y_smin, dtype=l.dtype, device=l.device))
    alpha = (l_max / log_range + 1.0).clamp(0.0, 1.0)

    return alpha * l_max + (1.0 - alpha) * l_lum
