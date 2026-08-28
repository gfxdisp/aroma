"""Image and video I/O.

Per the README, inputs are SDR BT.709: the EOTF is sRGB for still images and
gamma 2.2 for video.  Video is read and written by piping raw RGB frames
through ``ffmpeg``, so no frames are ever held in memory beyond the current
one -- the pipeline is causal anyway (Section 5.4).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from . import colorspace as cs

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------
def _to_tensor(frame: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8/uint16/float -> (3, H, W) float32 in [0, 1]."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr[..., :3]

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)

    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).to(device)


def _to_uint8(rgb: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [0, 1] -> (H, W, 3) uint8."""
    arr = rgb.clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
    return np.ascontiguousarray(arr.transpose(1, 2, 0))


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
def read_image(path: str | Path, device: str = "cpu") -> torch.Tensor:
    """Read an image and linearise it.  Returns (3, H, W) relative linear."""
    import imageio.v2 as imageio

    return cs.srgb_eotf(_to_tensor(imageio.imread(str(path)), device))


def write_image(path: str | Path, rgb_linear: torch.Tensor) -> None:
    """Encode with the sRGB OETF and write."""
    import imageio.v2 as imageio

    imageio.imwrite(str(path), _to_uint8(cs.srgb_oetf(rgb_linear)))


# --------------------------------------------------------------------------
# Locating ffmpeg
# --------------------------------------------------------------------------
def _find_tool(name: str) -> str:
    """Locate ffmpeg/ffprobe, preferring an explicit override, then PATH, then
    the binaries that ship inside the active Python environment."""
    candidates = [
        os.environ.get(f"AROMA_{name.upper()}"),
        shutil.which(name),
        os.path.join(sys.prefix, "Library", "bin", f"{name}.exe"),  # conda on Windows
        os.path.join(sys.prefix, "bin", name),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand

    try:  # optional fallback
        import imageio_ffmpeg

        if name == "ffmpeg":
            return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    raise RuntimeError(
        f"{name} not found. Install it (conda install ffmpeg), put it on PATH, "
        f"or point AROMA_{name.upper()} at the executable."
    )


def probe(path: str | Path) -> tuple[int, int, float, int]:
    """Return (width, height, fps, frame count) of a video."""
    out = subprocess.run(
        [
            _find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "json", str(path),
        ],
        capture_output=True, check=True, text=True,
    )
    s = json.loads(out.stdout)["streams"][0]

    num, _, den = s["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)

    try:
        n_frames = int(s.get("nb_frames", 0))
    except (TypeError, ValueError):
        n_frames = 0

    return int(s["width"]), int(s["height"]), fps, n_frames


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------
def read_video(
    path: str | Path, device: str = "cpu", gamma: float = 2.2
) -> Iterator[torch.Tensor]:
    """Yield (3, H, W) relative-linear frames, decoded with a gamma 2.2 EOTF.

    Call :func:`probe` first if you need the frame rate (the temporal filter of
    Section 5.4 does).
    """
    width, height, _, _ = probe(path)
    frame_bytes = width * height * 3

    proc = subprocess.Popen(
        [
            _find_tool("ffmpeg"), "-v", "error", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
            yield cs.gamma_eotf(_to_tensor(frame, device), gamma)
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


class VideoWriter:
    """Encode (3, H, W) relative-linear frames with a gamma 2.2 OETF."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
        gamma: float = 2.2,
        crf: int = 17,
    ):
        width, height = size
        self.gamma = gamma
        self._proc = subprocess.Popen(
            [
                _find_tool("ffmpeg"), "-v", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
                "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", str(path),
            ],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def write(self, rgb_linear: torch.Tensor) -> None:
        frame = _to_uint8(cs.gamma_oetf(rgb_linear, self.gamma))
        self._proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        if self._proc.wait() != 0:
            err = self._proc.stderr.read().decode("utf-8", "replace").strip()
            raise RuntimeError(f"ffmpeg failed: {err}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
