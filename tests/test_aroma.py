"""Unit tests: python -m unittest discover -s tests -v"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aroma import colorspace as cs
from aroma.color_correct import saturation_exponent, smooth_slopes
from aroma.importance import shape_importance, tone_importance
from aroma.params import AromaParams
from aroma.pipeline import AromaPipeline
from aroma.temporal import BoxFilter
from aroma.tone_curve import (
    ToneCurve,
    dest_dynamic_range,
    node_positions,
    solve_tone_curve,
)


class TestColorspace(unittest.TestCase):
    def test_srgb_roundtrip(self):
        v = torch.linspace(0, 1, 257)
        self.assertTrue(torch.allclose(cs.srgb_oetf(cs.srgb_eotf(v)), v, atol=1e-6))

    def test_log_encode_endpoints(self):
        """Eq. (5) maps the source peak to 0 and black to log2(y_smin)."""
        p = AromaParams()
        I = torch.tensor([[[0.0]], [[0.0]], [[p.Y_smax]]])
        l = cs.log_encode(I, p.Y_smax, p.y_smin)
        self.assertAlmostEqual(float(l[2]), 0.0, places=5)
        self.assertAlmostEqual(float(l[0]), float(np.log2(p.y_smin)), places=5)

    def test_log_decode_inverts_encode(self):
        p = AromaParams()
        I = torch.rand(3, 4, 4) * p.Y_smax
        l = cs.log_encode(I, p.Y_smax, p.y_smin)
        self.assertTrue(torch.allclose(cs.log_decode(l, p.y_smin) * p.Y_smax, I, atol=1e-3))

    def test_tone_value_alpha_endpoints(self):
        """Eq. (8): pure max-RGB at the peak, pure luminance at black."""
        p = AromaParams()
        white = cs.log_encode(torch.full((3, 1, 1), p.Y_smax), p.Y_smax, p.y_smin)
        self.assertAlmostEqual(float(cs.tone_value(white, p.y_smin)), 0.0, places=5)

        # At black alpha = 0, so m is the pseudo-luminance.  The paper's
        # weights sum to 1.0002 rather than 1, hence the loose tolerance.
        dark = cs.log_encode(torch.zeros(3, 1, 1), p.Y_smax, p.y_smin)
        self.assertAlmostEqual(
            float(cs.tone_value(dark, p.y_smin)), float(np.log2(p.y_smin)), delta=0.01
        )

    def test_tone_value_between_lum_and_max(self):
        p = AromaParams()
        I = torch.tensor([[[180.0]], [[20.0]], [[5.0]]])
        l = cs.log_encode(I, p.Y_smax, p.y_smin)
        m = float(cs.tone_value(l, p.y_smin))
        w = torch.tensor(cs.LUM_WEIGHTS).view(3, 1, 1)
        self.assertLessEqual(m, float(l.max()) + 1e-6)
        self.assertGreaterEqual(m, float((w * l).sum()) - 1e-6)


class TestSensitivity(unittest.TestCase):
    def test_monotonic_over_display_range(self):
        """Sensitivity rises with luminance across the usable range."""
        from aroma.sensitivity import contrast_threshold, lum_sens

        L = np.linspace(-6, 8, 50)
        S = lum_sens(L)
        self.assertTrue(np.all(np.diff(S) > 0))
        self.assertTrue(np.allclose(contrast_threshold(np.exp2(L)), 1.0 / S))

    def test_torch_matches_numpy(self):
        from aroma.sensitivity import lum_sens

        L = np.linspace(-5, 10, 17)
        self.assertTrue(
            np.allclose(lum_sens(L), lum_sens(torch.tensor(L)).numpy(), atol=1e-9)
        )


class TestImportance(unittest.TestCase):
    def test_shaping_renormalises(self):
        p = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01, 0.0])
        out = shape_importance(p)
        self.assertAlmostEqual(out.sum(), 1.0, places=9)
        self.assertEqual(out.size, p.size)

    def test_shaping_compresses_spread(self):
        """Eq. (23) reduces the ratio between the largest and smallest bins."""
        p = np.array([0.9] + [0.1 / 9] * 9)
        out = shape_importance(p)
        self.assertLess(out.max() / out.min(), p.max() / p.min())

    def test_flat_image_has_no_contrast_weight(self):
        """A perfectly flat image carries no local contrast at all, so the
        weighted histogram is empty and we fall back to a uniform importance.
        Eq. (22) is evaluated on mean-centred values precisely so that this
        does not dissolve into float32 rounding noise."""
        params = AromaParams()
        expected = shape_importance(np.full(params.bins, 1.0 / params.bins))
        for level in (-1.0, -6.0, -6.3):
            with self.subTest(level=level):
                p = tone_importance(
                    torch.full((64, 64), level), bins=params.bins, y_smin=params.y_smin
                )
                self.assertAlmostEqual(p.sum(), 1.0, places=9)
                self.assertTrue(np.allclose(p, expected))

    def test_importance_follows_textured_region(self):
        """Only the textured half should attract importance."""
        params = AromaParams()
        m = torch.full((64, 64), -6.0)
        m[:, 32:] = torch.linspace(-2.0, -1.0, 32).repeat(64, 1)
        p = tone_importance(m, bins=params.bins, y_smin=params.y_smin)
        lo = float(np.log2(params.y_smin))
        dark_bin = int((-6.0 - lo) / (0.0 - lo) * params.bins)
        self.assertGreater(p[dark_bin + 1 :].sum(), p[dark_bin])


class TestToneCurve(unittest.TestCase):
    def setUp(self):
        self.params = AromaParams()
        self.flat = np.full(self.params.bins, 1.0 / self.params.bins)

    def test_nodes_span_source_range(self):
        x = node_positions(self.params)
        self.assertEqual(x.size, self.params.bins + 1)
        self.assertAlmostEqual(x[0], np.log2(self.params.y_smin))
        self.assertAlmostEqual(x[-1], 0.0)

    def test_solution_is_feasible(self):
        """Monotonic and inside [y_min, y_max], across viewing conditions."""
        for Y_amb in (0.0, 1.0, 10.0, 50.0):
            with self.subTest(Y_amb=Y_amb):
                params = AromaParams(Y_amb=Y_amb)
                y = solve_tone_curve(self.flat, params)
                dr = dest_dynamic_range(params)
                self.assertTrue(np.all(np.diff(y) >= -1e-9), "not monotonic")
                self.assertGreaterEqual(y.min(), -dr - 1e-9)
                self.assertLessEqual(y.max(), 1e-9)

    def test_identity_conditions_give_unit_slope(self):
        """With no ambient light and a display matched to the source, the
        optimum is the identity: every segment keeps slope 1."""
        params = AromaParams(Y_amb=0.0, Y_smax=200.0, Y_dmax=200.0, Y_dmin=200.0 / 255.0)
        y = solve_tone_curve(self.flat, params)
        x = node_positions(params)
        slopes = np.diff(y) / (x[1] - x[0])
        self.assertTrue(
            np.allclose(slopes, 1.0, atol=0.05), f"slopes={np.round(slopes, 3)}"
        )

    def test_more_ambient_brightens_tones(self):
        """Section 5.1 / Fig. 4: "as ambient light increases, tone curves
        become steeper at lower tones ... this will also brighten images, as
        brighter tones better reproduce contrast when the display is affected
        by ambient light"."""
        mean_tone = [
            solve_tone_curve(self.flat, AromaParams(Y_amb=Y_amb)).mean()
            for Y_amb in (0.1, 1.0, 10.0, 40.0)
        ]
        self.assertTrue(
            np.all(np.diff(mean_tone) > 0), f"mean tone {np.round(mean_tone, 3)}"
        )

    def test_more_ambient_raises_emitted_luminance(self):
        """Fig. 4, solid lines: the luminance the display must emit for a given
        input rises with ambient light, because partial compensation subtracts
        Y_comp from the signal (Eq. 12)."""
        emitted = []
        for Y_amb in (0.1, 1.0, 10.0, 40.0):
            params = AromaParams(Y_amb=Y_amb, Y_dmax=200.0, Y_dmin=0.1, bits=14)
            y = solve_tone_curve(self.flat, params)
            x = node_positions(params)
            k = int(np.argmin(np.abs(x + 4.0)))  # four stops below source peak
            i_c = (np.exp2(y[k]) - params.y_smin) / (1.0 - params.y_smin)
            emitted.append((params.Y_dmax + params.Y_comp) * i_c - params.Y_comp)
        self.assertTrue(np.all(np.diff(emitted) > 0), f"emitted {np.round(emitted, 2)}")

    def test_ambient_shrinks_usable_range(self):
        """Eq. (3): partial compensation beats the raw additive contrast of
        Eq. (2), but the usable range still shrinks as ambient light grows."""
        drs = [dest_dynamic_range(AromaParams(Y_amb=a)) for a in (0.0, 10.0, 40.0)]
        self.assertTrue(np.all(np.diff(drs) < 0))

        params = AromaParams(Y_amb=40.0)
        additive = np.log2(
            (params.Y_dmax + params.Y_amb) / (params.Y_dmin + params.Y_amb)
        )
        self.assertGreater(dest_dynamic_range(params), additive)

    def test_apply_matches_nodes(self):
        x = np.linspace(-8, 0, 33)
        y = np.linspace(-4, 0, 33)
        got = ToneCurve(x, y).apply(torch.tensor(x, dtype=torch.float32))
        self.assertTrue(np.allclose(got.numpy(), y, atol=1e-5))

    def test_derivative_is_segment_slope(self):
        curve = ToneCurve(np.linspace(-8, 0, 33), np.linspace(-4, 0, 33))
        d = curve.derivative(torch.tensor([-7.9, -4.0, -0.1]))
        self.assertTrue(np.allclose(d.numpy(), 0.5, atol=1e-6))


class TestColorCorrection(unittest.TestCase):
    def test_unit_slope_is_identity(self):
        self.assertAlmostEqual(float(saturation_exponent(1.0)), 1.0, places=6)

    def test_compression_desaturates(self):
        self.assertLess(saturation_exponent(0.5), 1.0)
        self.assertGreater(saturation_exponent(2.0), 1.0)

    def test_smoothing_preserves_length_and_flattens(self):
        s = np.random.default_rng(0).uniform(0.2, 1.5, 32)
        out = smooth_slopes(s)
        self.assertEqual(out.size, s.size)
        self.assertLess(np.abs(np.diff(out)).max(), np.abs(np.diff(s)).max())


class TestTemporal(unittest.TestCase):
    def test_box_filter_converges_to_mean(self):
        f = BoxFilter(4)
        for _ in range(8):
            out = f(np.ones(3) * 2.0)
        self.assertTrue(np.allclose(out, 2.0))

    def test_box_filter_attenuates_a_step(self):
        f = BoxFilter(4)
        f(np.zeros(3))
        out = f(np.ones(3))
        self.assertTrue(np.all(out < 1.0) and np.all(out > 0.0))


class TestPipeline(unittest.TestCase):
    def test_output_range_and_shape(self):
        rgb = torch.rand(3, 96, 128)
        out = AromaPipeline(AromaParams(Y_amb=10.0))(rgb)
        self.assertEqual(out.shape, rgb.shape)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_transparent_pixels_stay_zero(self):
        """Section 8: zero-valued pixels keep their value throughout."""
        rgb = torch.rand(3, 64, 64)
        rgb[:, :16, :16] = 0.0
        out = AromaPipeline(AromaParams())(rgb)
        self.assertEqual(float(out[:, :16, :16].abs().max()), 0.0)

    def test_detail_enhancement_adds_local_contrast(self):
        """The full pipeline must retain more high-frequency energy than the
        global-only variant (the paper's 'global' vs 'global+local')."""
        x = torch.linspace(0, 40, 128)
        rgb = (0.25 + 0.05 * torch.sin(x)).expand(3, 96, 128).contiguous()

        def hf_energy(img):
            return float((img[..., 1:] - img[..., :-1]).abs().mean())

        g = AromaPipeline(AromaParams(Y_amb=10.0, detail=False))(rgb)
        f = AromaPipeline(AromaParams(Y_amb=10.0))(rgb)
        self.assertGreater(hf_energy(f), hf_energy(g))

    def test_identity_conditions_are_a_no_op(self):
        """No ambient light, display matched to the source, no detail or colour
        stage: the whole pipeline should pass the image through."""
        params = AromaParams(
            Y_amb=0.0,
            Y_smax=200.0,
            Y_dmax=200.0,
            Y_dmin=200.0 / 255.0,
            detail=False,
            color_correction=False,
        )
        rgb = torch.rand(3, 64, 64) * 0.8 + 0.1
        out = AromaPipeline(params)(rgb)
        self.assertLess(float((out - rgb).abs().mean()), 0.02)


class TestSimulateAmbient(unittest.TestCase):
    """The --simulate-ambient preview: Eq. (13) removes the ambient light from
    the drive signal, and this switch puts it back so the output shows what the
    wearer sees (Figs. 1, 8, 10)."""

    def setUp(self):
        self.rgb = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(3))

    def test_off_by_default(self):
        plain = AromaPipeline(AromaParams(Y_amb=10.0))(self.rgb)
        explicit = AromaPipeline(AromaParams(Y_amb=10.0, simulate_ambient=False))(self.rgb)
        self.assertTrue(torch.equal(plain, explicit))

    def test_black_maps_to_flat_ambient_grey(self):
        """A non-emitting pixel shows exactly the ambient light, which after
        renormalisation is Y_amb / (Y_dmax + Y_amb)."""
        params = AromaParams(Y_amb=10.0, Y_dmax=100.0, simulate_ambient=True)
        rgb = self.rgb.clone()
        rgb[:, :16, :16] = 0.0
        out = AromaPipeline(params)(rgb)

        expected = params.Y_amb / (params.Y_dmax + params.Y_amb)
        self.assertAlmostEqual(float(out[:, :16, :16].min()), expected, places=5)
        self.assertAlmostEqual(float(out[:, :16, :16].max()), expected, places=5)

    def test_nothing_clips(self):
        """Renormalising to the simulated peak means values stay inside [0, 1]
        even when the ambient light exceeds the display's own peak."""
        for Y_amb in (10.0, 50.0, 150.0):
            with self.subTest(Y_amb=Y_amb):
                params = AromaParams(Y_amb=Y_amb, Y_dmax=100.0, simulate_ambient=True)
                out = AromaPipeline(params)(self.rgb)
                self.assertGreaterEqual(float(out.min()), 0.0)
                self.assertLessEqual(float(out.max()), 1.0)

    def test_simulation_lowers_contrast(self):
        """The whole point of the preview: additive ambient light washes the
        image out, so the simulated version has less contrast."""
        params = dict(Y_amb=40.0, Y_dmax=100.0)
        plain = AromaPipeline(AromaParams(**params))(self.rgb)
        simulated = AromaPipeline(AromaParams(**params, simulate_ambient=True))(self.rgb)
        self.assertLess(float(simulated.std()), float(plain.std()))

    def test_is_an_affine_map_of_the_plain_output(self):
        """Simulation must not disturb the tone mapping itself -- it is exactly
        (out + Y_amb/Y_dmax) * Y_dmax/(Y_dmax + Y_amb) applied afterwards."""
        params = AromaParams(Y_amb=25.0, Y_dmax=100.0)
        plain = AromaPipeline(params)(self.rgb)
        simulated = AromaPipeline(
            AromaParams(Y_amb=25.0, Y_dmax=100.0, simulate_ambient=True)
        )(self.rgb)

        expected = (plain + params.Y_amb / params.Y_dmax) * (
            params.Y_dmax / (params.Y_dmax + params.Y_amb)
        )
        self.assertTrue(torch.allclose(simulated, expected, atol=1e-6))


class TestTemporalIntegration(unittest.TestCase):
    def test_filter_suppresses_curve_flicker(self):
        """Section 5.4: a curve recomputed per frame flickers; the box filter
        of Eq. (24) must damp that without blocking a real scene change."""
        rng = np.random.default_rng(0)
        base = torch.rand(3, 48, 64, generator=torch.Generator().manual_seed(1)) * 0.4
        frames = [
            (base * (1.0 + 0.15 * rng.standard_normal())).clamp(0, 1) for _ in range(16)
        ]

        jitter = {}
        for use_filter in (False, True):
            params = AromaParams(Y_amb=10.0, temporal_filter=use_filter, detail=False)
            pipe = AromaPipeline(params)
            pipe.start_video(15.0)
            curves = []
            for frame in frames:
                pipe(frame)
                curves.append(pipe.last_curve.y.copy())
            jitter[use_filter] = np.abs(np.diff(np.stack(curves), axis=0)).mean()

        self.assertLess(jitter[True], jitter[False] / 3.0)


if __name__ == "__main__":
    unittest.main()
