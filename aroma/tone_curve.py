"""The global tone curve, Section 5.

Solves the constrained optimization of Eq. (20): find node values y that keep
the *perceived* contrast of the source, given that the display has a much
smaller usable range once ambient light is added.

Where the paper and the MATLAB reference (``TC_AROMA_Peli.m``) disagree we
follow the MATLAB, since it is the authors' working code.  The three
differences are flagged inline below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

from .params import AromaParams
from .sensitivity import lum_sens


def node_positions(params: AromaParams) -> np.ndarray:
    """The x_k of Fig. 3: N nodes uniform in log2 over the source range.

    The paper says only "uniformly distributed nodes"; the range follows from
    Eq. (5), whose output spans [log2(y_smin), 0].
    """
    return np.linspace(np.log2(params.y_smin), 0.0, params.nodes)


def dest_dynamic_range(params: AromaParams) -> float:
    """Eq. (3): the dynamic range available after partial compensation.

    log2((Y_dmax + Y_comp) / (Y_dmin + Y_comp)) -- the display range is
    *widened* by subtracting Y_comp from the signal, because the ambient light
    fills the deficit back in.
    """
    return float(
        np.log2((params.Y_dmax + params.Y_comp) / (params.Y_dmin + params.Y_comp))
    )


# --------------------------------------------------------------------------
# Objective and constraints
# --------------------------------------------------------------------------
def _constraints(params: AromaParams, N: int, delta: float, dest_dr: float):
    """Assemble A y <= b, following the MATLAB reference.

    Rows 0..N-2   monotonicity:  y_k - y_{k+1} <= -min_slope
    Rows N-4..N-2 are tightened to -min_slope_top (the top three segments must
                  keep some slope, so highlights do not flatten out).
    Row  N-1      y_{N-1} <= 0        (peak clamp)
    Row  N        -y_0 <= dest_dr     (black clamp)

    NOTE: Eq. (20) writes the endpoints as *equalities* y_1 = y_min,
    y_N = y_max, and has no minimum-slope floor at all.  The MATLAB uses
    inequalities, which lets the solver pull the curve inside the display
    range when that costs less perceived contrast.
    """
    A = np.zeros((N + 1, N))
    b = np.full(N + 1, -params.min_slope)

    rows = np.arange(N - 1)
    A[rows, rows] = 1.0
    A[rows, rows + 1] = -1.0

    if N >= 5:
        b[N - 4 : N - 1] = -params.min_slope_top * delta

    A[N - 1, N - 1] = 1.0
    b[N - 1] = 0.0

    A[N, 0] = -1.0
    b[N] = dest_dr

    return A, b


def _loss(y, x, p, s_x, params: AromaParams, delta: float):
    """MATLAB ``pcd_loss_grad``: the perceived-contrast-distortion objective.

        L = sum_k p_k * ((dy_k * s_y * c_loss / delta - s_x) / w)^2

    with w = sqrt(s_y * c_loss * s_x), the geometric mean of Eq. (19).  Since
    s = 1 / c_T, this is Eq. (20) rewritten in terms of sensitivities, plus one
    extra factor:

    """
    if not params.do_sensitivity:
        # Eq. (14): physical rather than perceived contrast.
        return float(np.sum(p * (np.diff(y) / delta - 1.0) ** 2))

    y_bc = 0.5*(y[:-1] + y[1:])

    Y_tc = np.maximum(np.exp2(y_bc + np.log2(params.Y_dmax)) + params.Y_add, 1e-4)
    s_y = lum_sens(np.log2(Y_tc))
    c_loss = Y_tc / (Y_tc + params.Y_add)

    w = np.sqrt(s_y * c_loss * s_x)  # Eq. (19)

    residual = (np.diff(y) * s_y * c_loss / delta - s_x) / w
    loss = float(np.sum(p * residual**2))

    if params.lam:
        # Eq. (14)/(20) preserve-brightness term; disabled in the MATLAB.
        loss += params.lam * float(np.sum(p * (y[:-1] - x[:-1]) ** 2))
    return loss


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
def solve_tone_curve(
    p: np.ndarray,
    params: AromaParams,
    *,
    y0: np.ndarray | None = None,
    max_iter: int = 400,
) -> np.ndarray:
    """Return the N tone-curve node values y (log2, relative to Y_dmax)."""
    x = node_positions(params)
    N = x.size
    delta = float(x[1] - x[0])
    dest_dr = dest_dynamic_range(params)

    p = np.asarray(p, dtype=np.float64)
    if p.size != N - 1:
        raise ValueError(f"importance has {p.size} bins, expected {N - 1}")

    # Eq. (18): the source luminance of segment k. Note it carries no ambient
    # term -- the source is the intended appearance on a conventional display.
    x_bc = x[:-1] + 0.5 * delta
    s_x = (
        lum_sens(x_bc + np.log2(params.Y_smax))
        if params.do_sensitivity
        else np.ones(N - 1)
    )

    A, b = _constraints(params, N, delta, dest_dr)

    # Initial guess from the MATLAB reference (the paper does not give one).
    if y0 is None:
        y0 = np.linspace(-dest_dr, 0.0, N)

    result = minimize(
        _loss,
        np.asarray(y0, dtype=np.float64),
        args=(x, p, s_x, params, delta),
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda y: b - A @ y, "jac": lambda y: -A}],
        options={"maxiter": max_iter, "ftol": 1e-9},
    )
    y = np.asarray(result.x, dtype=np.float64)

    # Project back onto the feasible set: SLSQP satisfies constraints only to
    # its tolerance, and a non-monotonic curve would invert tones.
    y = np.maximum.accumulate(y)
    return np.clip(y, -dest_dr, 0.0)


# --------------------------------------------------------------------------
# Evaluating the curve
# --------------------------------------------------------------------------
@dataclass
class ToneCurve:
    """A piecewise-linear tone curve t() with uniformly spaced nodes."""

    x: np.ndarray  # node positions, log2 relative source values
    y: np.ndarray  # node values,    log2 relative display values

    @property
    def delta(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def slopes(self) -> np.ndarray:
        """Per-segment slope (dy/dx). 1 means contrast is preserved."""
        return np.diff(self.y) / self.delta

    def apply(self, m: torch.Tensor) -> torch.Tensor:
        """Eq. (9): t(m), by linear interpolation between nodes."""
        xt = torch.as_tensor(self.x, dtype=m.dtype, device=m.device)
        yt = torch.as_tensor(self.y, dtype=m.dtype, device=m.device)

        pos = (m - xt[0]) / self.delta
        i = pos.floor().long().clamp(0, xt.numel() - 2)
        frac = (pos - i).clamp(0.0, 1.0)
        return yt[i] + frac * (yt[i + 1] - yt[i])

    def derivative(self, m: torch.Tensor, slopes: np.ndarray | None = None) -> torch.Tensor:
        """dt/dm of Eq. (30), evaluated per pixel (piecewise constant)."""
        s = self.slopes if slopes is None else slopes
        st = torch.as_tensor(s, dtype=m.dtype, device=m.device)

        pos = (m - float(self.x[0])) / self.delta
        i = pos.floor().long().clamp(0, st.numel() - 1)
        return st[i]
