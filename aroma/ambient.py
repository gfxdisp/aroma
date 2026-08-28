"""Ambient light compensation, Section 4.6."""

from __future__ import annotations

import torch

from .params import AromaParams


def compensate(i_c: torch.Tensor, params: AromaParams) -> torch.Tensor:
    """Eq. (12): map relative values to the light the display must emit.

        I'' = [ (Y_dmax + Y_comp) i' - Y_comp ]_{Y_dmin}^{Y_dmax} + Y_amb

    Subtracting Y_comp is the partial compensation of Eq. (1): the ambient
    light fills that deficit back in, which restores contrast in dark areas.
    The clamp is required because detail enhancement (Section 6) can push
    values outside the display range.
    """
    I = (params.Y_dmax + params.Y_comp) * i_c - params.Y_comp
    return I.clamp(params.Y_dmin, params.Y_dmax) + params.Y_amb


def to_display_signal(I: torch.Tensor, params: AromaParams) -> torch.Tensor:
    """Eq. (13): rescale to [0, Y_dmax] with the ambient light removed again.

        I''' = Y_dmax (I'' - Y_amb - Y_dmin) / (Y_dmax - Y_dmin)

    This is the image handed to the rest of the display pipeline.
    """
    return params.Y_dmax * (I - params.Y_amb - params.Y_dmin) / (
        params.Y_dmax - params.Y_dmin
    )


def simulate(out: torch.Tensor, params: AromaParams) -> torch.Tensor:
    """Add the ambient light back, previewing the AR appearance (Figs. 1, 10).

    Eq. (12) adds Y_amb and Eq. (13) removes it again, so the pipeline emits a
    display drive signal rather than what a wearer sees.  ``out`` is relative to
    Y_dmax, so the ambient term is Y_amb / Y_dmax:

        out' = (out + Y_amb / Y_dmax) * Y_dmax / (Y_dmax + Y_amb)

    The second factor renormalises so the brightest possible simulated value is
    1.0 instead of clipping; note that 1.0 then means Y_dmax + Y_amb.

    NOTE: a strict simulation would also restore the display black level,
    ``out * (Y_dmax - Y_dmin) / Y_dmax + Y_dmin / Y_dmax``, which Eq. (13)
    folded out.  Only the ambient term is applied here.
    """
    return (out + params.Y_amb / params.Y_dmax) * (
        params.Y_dmax / (params.Y_dmax + params.Y_amb)
    )
