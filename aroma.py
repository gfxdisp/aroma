#!/usr/bin/env python
"""AROMA - Augmented Reality Optimized Tone Mapping.

    python aroma.py -i input.png -o output.png --ambient 10 --display-max 100

See README.md for the full option list.  Inputs are assumed to be SDR BT.709
(sRGB EOTF for images, gamma 2.2 for video).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from aroma.io import VideoWriter, is_video, probe, read_image, read_video, write_image
from aroma.params import AromaParams
from aroma.pipeline import AromaPipeline


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="aroma.py",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-i", "--input", required=True, help="input image or video")
    ap.add_argument("-o", "--output", required=True, help="output image or video")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    g = ap.add_argument_group("display and viewing conditions")
    g.add_argument("--ambient", type=float, default=10.0,
                   help="ambient luminance at the eye [cd/m^2]")
    g.add_argument("--display-max", type=float, default=100.0,
                   help="display peak luminance [cd/m^2]")
    g.add_argument("--display-min", type=float, default=0.1,
                   help="display black level [cd/m^2]")
    g.add_argument("--source-max", type=float, default=200.0,
                   help="assumed peak luminance of the source content [cd/m^2]")
    g.add_argument("--k-comp", type=float, default=0.25,
                   help="fraction of ambient light compensated for, Eq. (1)")

    g = ap.add_argument_group("algorithm (paper defaults)")
    g.add_argument("--bits", type=int, default=16,
                   help="source bit depth, sets y_smin in Eq. (6)")
    g.add_argument("--bins", type=int, default=32,
                   help="tone-curve segments / histogram bins")
    g.add_argument("--lambda", dest="lam", type=float, default=0.0,
                   help="weight of the preserve-brightness term of Eq. (20)")
    g.add_argument("--g-max", type=float, default=1.3,
                   help="maximum detail gain, Eq. (33)")
    g.add_argument("--mip-level", type=int, default=5,
                   help="MIP-map level n used for local contrast and the base layer")
    g.add_argument("--sigma-range", type=float, default=1.0,
                   help="width of the range kernel h_r, Eq. (34), in log2 units")

    g = ap.add_argument_group("stage switches")
    g.add_argument("--no-detail", action="store_true",
                   help="disable detail enhancement (the paper's 'AROMA global')")
    g.add_argument("--no-color-correction", action="store_true")
    g.add_argument("--no-sensitivity", action="store_true",
                   help="use the physical-contrast objective of Eq. (14)")
    g.add_argument("--no-temporal-filter", action="store_true",
                   help="disable the per-frame tone-curve box filter, Eq. (24)")
    g.add_argument("--simulate-ambient", action="store_true",
                   help="add the ambient light back into the output, previewing "
                        "the AR appearance instead of the display drive signal; "
                        "output is renormalised so 1.0 means display-max + ambient")
    g.add_argument("--debug-dir", help="write per-stage intermediate images here")

    return ap


def params_from_args(args: argparse.Namespace) -> AromaParams:
    return AromaParams(
        Y_amb=args.ambient,
        Y_dmax=args.display_max,
        Y_dmin=args.display_min,
        Y_smax=args.source_max,
        k_comp=args.k_comp,
        bits=args.bits,
        bins=args.bins,
        lam=args.lam,
        g_max=args.g_max,
        mip_level=args.mip_level,
        sigma_range=args.sigma_range,
        detail=not args.no_detail,
        color_correction=not args.no_color_correction,
        do_sensitivity=not args.no_sensitivity,
        temporal_filter=not args.no_temporal_filter,
        simulate_ambient=args.simulate_ambient,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params = params_from_args(args)

    if params.Y_dmin >= params.Y_dmax:
        print("error: --display-min must be below --display-max", file=sys.stderr)
        return 2

    pipeline = AromaPipeline(params, device=args.device, debug_dir=args.debug_dir)

    if is_video(args.input):
        width, height, fps, n_frames = probe(args.input)
        pipeline.start_video(fps)

        start = time.time()
        count = 0
        with VideoWriter(args.output, fps, (width, height)) as writer:
            for frame in read_video(args.input, args.device):
                writer.write(pipeline(frame, frame_index=count))
                count += 1
                if count % 10 == 0:
                    total = f"/{n_frames}" if n_frames else ""
                    print(f"\r{count}{total} frames "
                          f"({count / (time.time() - start):.1f} fps)", end="", flush=True)
        print(f"\rwrote {args.output}: {count} frames in "
              f"{time.time() - start:.1f} s        ")
    else:
        write_image(args.output, pipeline(read_image(args.input, args.device)))
        print(f"wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
