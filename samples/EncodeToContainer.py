# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Encode synthetic NV12 frames with NVENC and mux directly into
an MP4/MKV/WebM container using the Python-exposed muxer.

This demonstrates the demuxer-free path:
  - CreateEncoder(...) -> elementary bitstream packets
  - CreateMuxer(..., codec/width/height/fps) -> container
  - MuxBitstream() to append packets
"""

import argparse
from pathlib import Path

import numpy as np

import PyNvVideoCodec as nvc


def encode_to_container(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
    codec: str,
    gpu_id: int,
) -> None:
    codec = codec.lower()

    # Encoder configuration: NV12 input, GPU-only path, matching tests/samples.
    config: dict[str, str] = {
        "codec": codec,
        "preset": "P3",
        "gpu_id": str(gpu_id),
        "fps": str(int(fps)),
    }

    try:
        encoder = nvc.CreateEncoder(width, height, "NV12", True, **config)
    except (nvc.PyNvVCExceptionUnsupported, nvc.PyNvVCException) as exc:
        raise SystemExit(f"NVENC encoder not usable on this system: {exc}") from exc

    muxer = nvc.CreateMuxer(
        filename=str(output_path),
        codec=codec,
        width=width,
        height=height,
        fps=fps,
    )

    frame_size = int(width * height * 3 / 2)
    y_size = width * height
    uv_size = frame_size - y_size

    for i in range(num_frames):
        # Simple synthetic NV12 pattern that changes per frame.
        y_plane = np.full(y_size, i * 10 % 256, dtype=np.uint8)
        uv_plane = np.full(uv_size, 128, dtype=np.uint8)
        frame = np.concatenate([y_plane, uv_plane])

        bitstream = encoder.Encode(frame)
        if bitstream:
            muxer.MuxBitstream(bitstream, is_key_frame=(i == 0))

    # Flush encoder and mux remaining packets.
    bitstream = encoder.EndEncode()
    if bitstream:
        muxer.MuxBitstream(bitstream, is_key_frame=False)

    muxer.Close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode synthetic NV12 frames with NVENC and mux into a container using PyNvMuxer."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output container path (extension selects container, e.g., .mp4, .mkv, .webm).",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=str,
        default="128x72",
        help="Frame size as WxH (default: 128x72).",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=float,
        default=24.0,
        help="Frame rate used for timestamps (default: 24.0).",
    )
    parser.add_argument(
        "-n",
        "--num-frames",
        type=int,
        default=16,
        help="Number of synthetic frames to encode (default: 16).",
    )
    parser.add_argument(
        "-c",
        "--codec",
        type=str,
        default="h264",
        help="Codec to use: h264, hevc, or av1 (default: h264).",
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ordinal to use (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    width_str, height_str = args.size.lower().split("x")
    width = int(width_str)
    height = int(height_str)

    if width % 2 or height % 2:
        raise SystemExit("Width and height must be even for NV12 encoding.")

    encode_to_container(
        output_path=args.output,
        width=width,
        height=height,
        fps=args.fps,
        num_frames=args.num_frames,
        codec=args.codec,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()

