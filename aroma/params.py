"""Configuration for the AROMA operator (paper Section 4, input list)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AromaParams:
    """All parameters of the AROMA pipeline.

    Absolute luminance values are in cd/m^2.  Defaults follow ``README.md``
    where it specifies one, and the paper otherwise; parameters whose value the
    paper never states are marked below.
    """

    # --- Display and viewing conditions (Section 4) -----------------------
    Y_amb: float = 10.0
    """Ambient light at the eye, after any dimming element (Y_amb)."""

    Y_dmax: float = 100.0
    """Peak absolute display luminance (Y_dmax)."""

    Y_dmin: float = 1.0
    """Display black level (Y_dmin).  NOTE: README defaults this to 1 cd/m^2
    while the paper's example in Section 4 uses 0.1 cd/m^2.  We follow the
    README."""

    Y_smax: float = 200.0
    """Peak absolute luminance of the source content (Y_smax), Eq. (5)."""

    k_comp: float = 0.25
    """Fraction of the ambient light that is compensated for, Eq. (1).
    0.25 in the paper's experiments.  Not exposed by the README."""

    # --- Signal encoding (Section 4.3) ------------------------------------
    bits: int = 16
    """Bit depth used to derive y_smin in Eq. (6)."""

    # --- Global tone curve (Section 5) ------------------------------------
    bins: int = 32
    """Number of tone-curve segments / histogram bins (K).  The paper only
    states 32 for the on-device implementation (Section 10)."""

    lam: float = 0.0
    """Weight of the preserve-brightness term of Eq. (14)/(20).  The paper
    uses 0.01, but the term is commented out in the MATLAB reference (and
    compares y_k against x_k, which are anchored to different references), so
    it is off by default."""

    do_sensitivity: bool = True
    """Use the supra-threshold perceived-contrast objective of Eq. (20).
    When False, falls back to the physical-contrast objective of Eq. (14)
    (MATLAB's ``do_sensitivity = false``)."""

    min_slope: float = 0.0
    """Minimum tone-curve slope (monotonicity), MATLAB ``min_slope``."""

    min_slope_top: float = 0.2
    """Minimum slope enforced on the top three segments, MATLAB
    ``min_slope_top = 0.2*delta``.  Not mentioned in the paper."""

    # --- Detail enhancement (Section 6) -----------------------------------
    detail: bool = True
    mip_level: int = 5
    """n in Eq. (22)/(25): 32x downsampling."""

    g_max: float = 1.3
    """Maximum detail gain, Eq. (33).  Section 6 says "typically between 1 and
    1.6" without giving a value; Section 10 reports k_max = 1.3 on device."""

    sigma_range: float = 1.0
    """Width of the range kernel h_r of the edge-stopping upsample, Eq. (34),
    in log2 units (Section 10)."""

    # --- Colour correction (Section 7) ------------------------------------
    color_correction: bool = True

    # --- Preview ----------------------------------------------------------
    simulate_ambient: bool = False
    """Add the ambient light back into the output image, to preview how the
    result looks through the AR optics (as in Figs. 1, 8 and 10).  This is a
    diagnostic switch, not part of the paper's pipeline: with it set the output
    is no longer a display drive signal."""

    # --- Video ------------------------------------------------------------
    temporal_filter: bool = True
    """Box filter over tone-curve nodes, Eq. (24).  Width = frame rate."""

    @property
    def y_smin(self) -> float:
        """Eq. (6): the smallest representable relative source value."""
        return 1.0 / (2.0**self.bits - 1.0)

    @property
    def Y_comp(self) -> float:
        """Eq. (1): the amount of ambient light we compensate for."""
        return self.k_comp * self.Y_amb

    @property
    def Y_add(self) -> float:
        """Residual ambient light left on the display after partial
        compensation. """
        return self.Y_amb - self.Y_comp

    @property
    def nodes(self) -> int:
        """Number of tone-curve nodes: one more than the number of segments."""
        return self.bins + 1
