"""The AROMA processing pipeline of Figure 2.

    input -> native colour space (4.2) -> log2 encoding (4.3) -> tone values
    (4.4) -> [tone importance (5.2/5.3) -> global tone curve (5.1) -> temporal
    filter (5.4)] -> apply tone curve -> detail enhancement (6) -> colour
    correction (7) -> exponentiate -> ambient compensation (4.6) -> output
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import ambient, color_correct, colorspace as cs, detail as detail_mod
from .importance import tone_importance
from .params import AromaParams
from .temporal import BoxFilter
from .tone_curve import ToneCurve, node_positions, solve_tone_curve


class AromaPipeline:
    """Stateful across frames (the temporal filter of Section 5.4 needs it)."""

    def __init__(
        self,
        params: AromaParams,
        device: str = "cpu",
        M_native: torch.Tensor | None = None,
        debug_dir: str | Path | None = None,
    ):
        self.params = params
        self.device = device
        self.M_native = M_native
        self.x = node_positions(params)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self._filter: BoxFilter | None = None
        self.last_curve: ToneCurve | None = None

    # ------------------------------------------------------------------
    def start_video(self, fps: float) -> None:
        """Arm the temporal filter, whose width is the frame rate (Sec. 5.4)."""
        self._filter = BoxFilter(int(round(fps))) if self.params.temporal_filter else None

    # ------------------------------------------------------------------
    def __call__(self, rgb: torch.Tensor, frame_index: int = 0) -> torch.Tensor:
        """Tone map one frame.

        ``rgb`` is (3, H, W), linear, relative to the source peak, in BT.709.
        The result is (3, H, W), linear, relative to Y_dmax.
        """
        p = self.params
        rgb = rgb.to(self.device).float().clamp_min(0.0)

        # Section 8: transparent (zero-valued) pixels are excluded from all
        # analysis and keep their value, otherwise the extreme dark-to-bright
        # transitions at their borders dominate the contrast budget.
        opaque = rgb.amax(dim=0) > 0

        # --- 4.2-4.4 -------------------------------------------------------
        I_native = cs.to_native(rgb * p.Y_smax, self.M_native)
        l_c = cs.log_encode(I_native, p.Y_smax, p.y_smin)
        m = cs.tone_value(l_c, p.y_smin)

        # --- 5.2-5.3 -------------------------------------------------------
        importance = tone_importance(
            m, bins=p.bins, y_smin=p.y_smin, level=p.mip_level, mask=opaque
        )

        # --- 5.1 -----------------------------------------------------------
        y = solve_tone_curve(importance, p)

        # --- 5.4 -----------------------------------------------------------
        if self._filter is not None:
            y = self._filter(y)

        curve = ToneCurve(self.x, y)
        self.last_curve = curve

        # --- 4.5 / 6 -------------------------------------------------------
        t_m = curve.apply(m)  # Eq. (9)

        if p.detail:
            dtdm = curve.derivative(m)  # Eq. (30)
            m_prime, gain = detail_mod.enhance(t_m, dtdm, p)
        else:
            m_prime, gain = t_m, None

        # --- 7 -------------------------------------------------------------
        if p.color_correction:
            d_t = color_correct.smooth_slopes(curve.slopes)
            s = color_correct.saturation_exponent(curve.derivative(m, slopes=d_t))
        else:
            s = torch.ones_like(m)
        l_prime = color_correct.apply(l_c, m, m_prime, s)  # Eq. (36)

        # --- 4.6 -----------------------------------------------------------
        i_prime = cs.log_decode(l_prime, p.y_smin)  # Eq. (11)
        I_out = ambient.to_display_signal(ambient.compensate(i_prime, p), p)

        out = cs.from_native(I_out, self.M_native) / p.Y_dmax

        # Zero first -- a transparent pixel emits nothing (Section 8) -- but
        # then add the ambient light, which still reaches the eye through it.
        out = torch.where(opaque.unsqueeze(0), out, torch.zeros_like(out))
        if p.simulate_ambient:
            out = ambient.simulate(out, p)
        out = out.clamp(0.0, 1.0)

        if self.debug_dir:
            self._dump(frame_index, m=m, t_m=t_m, m_prime=m_prime, gain=gain, curve=curve)
        return out

    # ------------------------------------------------------------------
    def _dump(self, index: int, *, m, t_m, m_prime, gain, curve: ToneCurve) -> None:
        from . import io as aio

        stem = self.debug_dir / f"frame{index:05d}"

        def save(name: str, img: torch.Tensor, lo: float, hi: float) -> None:
            norm = ((img - lo) / max(hi - lo, 1e-6)).clamp(0, 1)
            aio.write_image(f"{stem}_{name}.png", norm.expand(3, *norm.shape[-2:]) ** (1 / 2.2))

        lo = float(np.log2(self.params.y_smin))
        save("m", m.unsqueeze(0), lo, 0.0)
        save("t_m", t_m.unsqueeze(0), float(curve.y.min()), 0.0)
        save("m_prime", m_prime.unsqueeze(0), float(curve.y.min()), 0.0)
        save("m_detail", (m_prime - t_m).unsqueeze(0), -0.5, 0.5)
        if gain is not None:
            save("gain", gain.unsqueeze(0), 1.0, self.params.g_max)

        np.savetxt(
            f"{stem}_curve.csv",
            np.column_stack([curve.x, curve.y]),
            delimiter=",",
            header="x_log2_source,y_log2_display",
            comments="",
        )
