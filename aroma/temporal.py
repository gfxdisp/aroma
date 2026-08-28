"""Temporal filtering of the tone curve, Section 5.4.

A curve recomputed per frame flickers.  The paper compares four low-pass
filters (Fig. 5) and picks a box filter whose width equals the frame rate,
which places the first zero of its frequency response at 0.5 Hz -- below the
4-11 Hz peak of flicker sensitivity.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class BoxFilter:
    """Causal moving average over the last ``width`` tone curves, Eq. (24).

    The paper does not say what to do before ``width`` frames have been seen;
    we average over whatever is available, which converges to the steady-state
    filter within one second.
    """

    def __init__(self, width: int):
        self.width = max(1, int(width))
        self._history: deque[np.ndarray] = deque(maxlen=self.width)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        self._history.append(np.asarray(y, dtype=np.float64))
        return np.mean(np.stack(self._history, axis=0), axis=0)

    def reset(self) -> None:
        self._history.clear()
