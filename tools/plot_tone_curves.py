"""Reproduce Figure 4 of the paper: global tone curves for several ambient
light levels, with a flat importance function and Y_dmax = 200 cd/m^2.

    python tools/plot_tone_curves.py [-o figure4.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aroma.params import AromaParams
from aroma.tone_curve import node_positions, solve_tone_curve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default="figure4.png")
    ap.add_argument("--ambient", type=float, nargs="+", default=[0.1, 1.0, 10.0, 20.0])
    ap.add_argument("--display-max", type=float, default=200.0)
    ap.add_argument("--display-min", type=float, default=0.1)
    ap.add_argument("--source-max", type=float, default=200.0)
    ap.add_argument("--bits", type=int, default=14,
                    help="source bit depth (Eq. 6); sets the plotted input range")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, Y_amb in enumerate(args.ambient):
        params = AromaParams(
            Y_amb=Y_amb,
            Y_dmax=args.display_max,
            Y_dmin=args.display_min,
            Y_smax=args.source_max,
            bits=args.bits,
        )
        x = node_positions(params)
        p = np.full(params.bins, 1.0 / params.bins)  # flat importance
        y = solve_tone_curve(p, params)

        Y_in = params.Y_smax * np.exp2(x)
        # Eq. (11) then Eq. (12), without the clamp and without +Y_amb.
        i_c = (np.exp2(y) - params.y_smin) / (1.0 - params.y_smin)
        Y_out = (params.Y_dmax + params.Y_comp) * i_c - params.Y_comp

        c = colors[i % len(colors)]
        ax.plot(Y_in, np.maximum(Y_out, 1e-3), "-", color=c, label=f"$Y_{{amb}}$={Y_amb:g}")
        ax.plot(Y_in, np.maximum(Y_out, 1e-3) + Y_amb, ":", color=c)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input (content) luminance [cd/m$^2$]")
    ax.set_ylabel("Display luminance [cd/m$^2$]")
    ax.set_xlim(0.01, args.source_max)
    ax.set_ylim(0.05, args.display_max * 1.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
