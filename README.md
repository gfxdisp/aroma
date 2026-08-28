This repository is a Python implementation of Augmented Reality Optimized Tone Mapping (AROMA), based on the SIGGRAPH paper:
AROMA : Augmented Reality Optimized Tone Mapping
SIGGRAPH Asia 2026

The tone mapping compensates for the ambient (environment) light that is transmitted through the optics of an AR headset and therefore restricts the visible contrast.

The code is the reimplementation of the method by one of the co-authors, based on the details in the paper.

# Dependencies

The code uses PyTorch for GPU acceleration, ffmpeg for video loading and writing, and imageio for loading and saving images. SciPy is used for the constrained tone-curve optimization and to read the sensitivity LUT; matplotlib is needed only by `tools/plot_tone_curves.py`. See `requirements.txt`.

`ffmpeg` and `ffprobe` are invoked as external executables and are needed only for video. They are located on `PATH`, inside the active Python environment (`$sys.prefix/Library/bin` for conda on Windows), via `imageio-ffmpeg` if installed, or via the `AROMA_FFMPEG` / `AROMA_FFPROBE` environment variables.

`sensitivity_lut.mat` must be present in the repository root. It holds the polynomial fit of peak achromatic contrast sensitivity against log2 luminance that Section 5.1 of the paper obtains from castleCSF. Without it the code falls back to approximate built-in coefficients and warns.

# Example use

```bash
python aroma.py -i <input_image.png|input_video.mp4> -o <output_image.png|output_video.mp4> --device <cpu|cuda> --ambient <luminance> --display-max <luminance> --display-min <luminance> --source-max <luminance>
```

Note that the input images are assumed to be standard dynamic range (SDR) in the BT.709 color space and default transfer function (EOTF is sRGB for images, gamma 2.2 for video).

The optional parameters are:
* `--ambient <luminance>` - the ambient light luminance in cd/m^2 that is transmitted through the AR headset and enters the eye. 10 cd/m^2 by default.
* `--display-max <luminance>` - the display peak luminance in cd/m^2. The default is 100 cd/m^2.
* `--display-min <luminance>` - the display black level in cd/m^2 (luminance emitted when displaying black pixels). The default is 1 cd/m^2.
* `--source-max <luminance>` - the assumed peak luminance of the source content. The default is 200 cd/m^2.

The method has further parameters that the paper specifies but the interface above does not expose:

* `--k-comp <fraction>` - the fraction of the ambient light that is compensated for, `k_comp` in Eq. (1). 0.25 by default.
* `--bits <n>` - the source bit depth, which fixes `y_smin` in Eq. (6) and therefore the dynamic range of the input signal. 8 by default (about 8 stops).
* `--bins <n>` - the number of tone-curve segments and histogram bins. 32 by default.
* `--lambda <weight>` - the weight of the preserve-brightness term of Eq. (20). 0 by default (the term is disabled in the reference MATLAB implementation).
* `--g-max <gain>` - the maximum detail enhancement gain, Eq. (33). 1.3 by default.
* `--mip-level <n>` - the MIP-map level used for local contrast and for the base layer, `n` in Eq. (22) and (25). 5 by default (32x downsampling).
* `--sigma-range <width>` - the width of the range kernel `h_r` of the edge-stopping upsample, Eq. (34), in log2 units. 1.0 by default.

Stages can be switched off for comparison, matching the conditions of the paper's subjective study:

* `--no-detail` - global tone curve only (the paper's *AROMA global*).
* `--no-color-correction` - skip the saturation correction of Section 7.
* `--no-sensitivity` - optimize physical rather than perceived contrast, i.e. Eq. (14) instead of Eq. (20).
* `--no-temporal-filter` - recompute the tone curve per video frame without the box filter of Eq. (24).
* `--simulate-ambient` - add the ambient light back into the output image, so that it previews what the wearer sees through the AR optics rather than the signal sent to the display. The result is renormalised so that 1.0 corresponds to `--display-max` plus `--ambient`; the preview is therefore darker and lower in contrast than the normal output, and nothing clips. Off by default.
* `--debug-dir <path>` - write per-stage intermediate images and the solved tone curve.

# Development

```bash
python -m unittest discover -s tests -v          # all tests
python -m unittest tests.test_aroma.TestToneCurve -v          # one class
python -m unittest tests.test_aroma.TestToneCurve.test_identity_conditions_give_unit_slope

python tools/plot_tone_curves.py -o figure4.png  # reproduces Figure 4 of the paper
```
