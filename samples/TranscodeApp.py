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
End-to-end GPU-based transcode application using PyNvVideoCodec.

This script:
  - Decodes an input MP4 (or any supported container)
  - Re-encodes it on the GPU into another MP4/MKV container
    using the built-in SimpleTranscoder (video+audio muxing)
  - Optionally displays decoded frames on the CPU while playing
    back the source video (GPU is used for decode; display is CPU).

Example:
    python TranscodeApp.py \\
        -i input.mp4 \\
        -o output.mp4 \\
        --codec h264 \\
        --preset P3 \\
        --gpu-id 0 \\
        --display
"""

import argparse
import json
from pathlib import Path
import ctypes as C

import numpy as np

import PyNvVideoCodec as nvc


def run_transcode(input_path: Path, output_path: Path, gpu_id: int, codec: str, preset: str) -> None:
    """
    Use the library's SimpleTranscoder to perform GPU-based
    decode + encode + mux into the requested container.
    """
    # Encoder configuration. SimpleTranscoder will derive width/height/FPS.
    # Start from the default transcode config used by the samples, if present,
    # then override codec/preset from CLI so behavior matches the reference.
    config: dict[str, str] = {}
    default_cfg_path = Path("samples/SimpleDecoder/transcode_config.json")
    if default_cfg_path.is_file():
        try:
            config = json.loads(default_cfg_path.read_text())
        except Exception:
            config = {}

    config["codec"] = codec.lower()
    config["preset"] = preset.upper()

    transcoder = nvc.Transcoder(
        str(input_path),
        str(output_path),
        gpu_id,
        0,  # cuda_context (0 = let library manage)
        0,  # cuda_stream (0 = let library manage)
        **config,
    )
    transcoder.transcode_with_mux()


def remux_only(input_path: Path, output_path: Path) -> None:
    """
    Simple remux using the Python-exposed demuxer and muxer.

    This does not re-encode; it reads compressed packets from the input
    container and writes them into a new container using PyNvMuxer.
    """
    demuxer = nvc.CreateDemuxer(filename=str(input_path))
    muxer = nvc.CreateMuxer(filename=str(output_path), demuxer=demuxer)

    for packet in demuxer:
        muxer.Mux(packet)

    # Ensure container trailer is written before returning.
    muxer.Close()


def _nv12_frame_to_bgr(frame) -> np.ndarray:
    """
    Convert a DecodedFrame in NV12 layout (host memory) into a BGR image.

    Note: this assumes the decoder was created with usedevicememory=0 so that
    GetPtrToPlane(0) refers to host-accessible memory.
    """
    import cv2

    total_h, width = frame.shape  # (height * 3/2, width)
    height = int(total_h * 2 / 3)
    size = frame.framesize()

    addr = frame.GetPtrToPlane(0)
    buf = np.ctypeslib.as_array(
        C.cast(addr, C.POINTER(C.c_uint8)),
        shape=(size,),
    )

    nv12 = buf.reshape((total_h, width))
    # OpenCV expects full NV12 buffer (Y followed by interleaved UV)
    bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    return bgr


def display_video(input_path: Path, gpu_id: int, max_frames: int | None = None) -> None:
    """
    Decode video frames on the GPU and display them on the CPU.

    This path does NOT participate in the muxing/encoding; it is purely for
    preview/visualization and uses the standard CreateDemuxer/CreateDecoder API.
    """
    import cv2

    demuxer = nvc.CreateDemuxer(filename=str(input_path))
    decoder = nvc.CreateDecoder(
        gpuid=gpu_id,
        codec=demuxer.GetNvCodecId(),
        cudacontext=0,
        cudastream=0,
        usedevicememory=0,  # host-accessible memory for convenient display
        latency=nvc.DisplayDecodeLatencyType.NATIVE,
    )

    frames_shown = 0
    window_name = "PyNvVideoCodec Preview"

    for packet in demuxer:
        for frame in decoder.Decode(packet):
            try:
                bgr = _nv12_frame_to_bgr(frame)
            except Exception:
                # If conversion fails for any reason, stop preview but do not crash the app.
                cv2.destroyAllWindows()
                return

            cv2.imshow(window_name, bgr)
            # Close on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                cv2.destroyAllWindows()
                return

            frames_shown += 1
            if max_frames is not None and frames_shown >= max_frames:
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-based MP4/MKV transcode using PyNvVideoCodec, with optional preview."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input video file (e.g., .mp4, .mkv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output container (extension determines container, e.g., .mp4 or .mkv).",
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ordinal to use (default: 0).",
    )
    parser.add_argument(
        "-c",
        "--codec",
        type=str,
        default="h264",
        help="Output codec (h264, hevc, av1). Default: h264.",
    )
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        default="P3",
        help="Encoder preset (P1-P7). Default: P3.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="If set, also decode and display frames on the CPU.",
    )
    parser.add_argument(
        "--remux-only",
        action="store_true",
        help="If set, only remux the input container using PyNvMuxer (no re-encode).",
    )
    parser.add_argument(
        "--max-display-frames",
        type=int,
        default=0,
        help="Maximum number of frames to display (0 = all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file does not exist: {args.input}")

    # Either perform a full GPU transcode or a simple remux using the
    # new Python-visible muxer.
    if args.remux_only:
        remux_only(args.input, args.output)
    else:
        run_transcode(args.input, args.output, args.gpu_id, args.codec, args.preset)

    # Optional preview (decode-only path, not required for transcode).
    if args.display:
        max_frames = args.max_display_frames if args.max_display_frames > 0 else None
        display_video(args.input, args.gpu_id, max_frames=max_frames)


if __name__ == "__main__":
    main()
