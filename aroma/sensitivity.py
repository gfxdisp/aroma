"""Contrast detection thresholds, Section 5.1 (Eq. 17 and 19).

The paper obtains c_T(Y) by taking the peak of the achromatic contrast
sensitivity function from castleCSF [Ashraf et al. 2024] at luminance Y and
inverting it.  It does not give a closed form.  The MATLAB reference
(``TC_AROMA_Peli.m``) instead evaluates a polynomial fit stored in
``sensitivity_lut.mat``::

    S(L) = 2 ^ polyval(pc, L),   L = log2(Y)

and that is what we use here.  The .mat also carries the tabulated ``L``/``S``
pair the polynomial was fitted to, which we use to bound extrapolation.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

_LUT_NAME = "sensitivity_lut.mat"

# Fallback used only when sensitivity_lut.mat is absent: a degree-2 fit of
# log2(peak achromatic sensitivity) against log2(luminance).  Approximate.
_FALLBACK_PC = np.array([-2.341981e-03, 5.509792e-02, 4.539264e00])
_FALLBACK_RANGE = (-6.643856, 13.287712)


def _search_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [here.parent / _LUT_NAME, here / _LUT_NAME, Path.cwd() / _LUT_NAME]


@lru_cache(maxsize=1)
def _load() -> tuple[np.ndarray, float, float]:
    """Return (polynomial coefficients, L_min, L_max)."""
    for path in _search_paths():
        if not path.is_file():
            continue
        from scipy.io import loadmat

        data = loadmat(str(path))
        pc = np.asarray(data["pc"], dtype=np.float64).ravel()
        if "L" in data:
            L = np.asarray(data["L"], dtype=np.float64).ravel()
            return pc, float(L.min()), float(L.max())
        return pc, *_FALLBACK_RANGE

    warnings.warn(
        f"{_LUT_NAME} not found (looked in {[str(p) for p in _search_paths()]}). "
        "Falling back to approximate built-in sensitivity coefficients; "
        "results will not match the reference implementation.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _FALLBACK_PC, *_FALLBACK_RANGE


def lum_sens(L):
    """Peak achromatic contrast sensitivity S at log2 luminance ``L``.

    Accepts and returns either a numpy array or a torch tensor.  ``L`` is
    clamped to the range the polynomial was fitted over: the fit is a downward
    parabola in log2 luminance, so unclamped extrapolation eventually turns
    over and produces nonsense.  (The MATLAB reference does not clamp, but
    also never evaluates far outside the fitted range.)
    """
    pc, L_min, L_max = _load()

    if isinstance(L, torch.Tensor):
        Lc = L.clamp(L_min, L_max)
        acc = torch.zeros_like(Lc)
        for coef in pc:  # Horner, matching MATLAB's polyval
            acc = acc * Lc + float(coef)
        return torch.exp2(acc)

    Lc = np.clip(np.asarray(L, dtype=np.float64), L_min, L_max)
    return np.exp2(np.polyval(pc, Lc))


def contrast_threshold(Y):
    """Eq. (17)/(19): c_T(Y), the detection threshold at luminance Y."""
    if isinstance(Y, torch.Tensor):
        return 1.0 / lum_sens(torch.log2(Y.clamp_min(1e-6)))
    return 1.0 / lum_sens(np.log2(np.maximum(np.asarray(Y, dtype=np.float64), 1e-6)))
