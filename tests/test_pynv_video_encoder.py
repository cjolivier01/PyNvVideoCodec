import subprocess
from pathlib import Path

import pytest
import torch

import PyNvVideoCodec as nvc
from PyNvVideoCodec import PyNvVideoEncoder


def _ensure_ffmpeg() -> str:
    from shutil import which

    path = which("ffmpeg")
    if path is None:
        pytest.skip("ffmpeg is required for encoder test")
    return path


def _ffprobe_stream_info(path: Path, ffmpeg: str) -> list[str]:
    """Run ffprobe and return lines describing the first streams."""
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt",
        "-of",
        "default=nw=1",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()


def _skip_if_codec_unsupported(codec: str) -> dict:
    """Return encoder caps or skip if NVENC does not support the requested codec."""
    try:
        caps = nvc.GetEncoderCaps(codec=codec)
    except (nvc.PyNvVCException, nvc.PyNvVCExceptionUnsupported):
        pytest.skip(f"Codec {codec} not supported by NVENC on this system")
    if not caps or caps.get("num_encoder_engines", 0) <= 0:
        pytest.skip(f"Codec {codec} not supported by NVENC on this system")
    return caps


def _encode_and_check(tmp_path: Path, codec: str) -> None:
    """Helper to encode a short CUDA tensor clip with the given codec and assert container properties."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    caps = _skip_if_codec_unsupported(codec)
    ffmpeg = _ensure_ffmpeg()

    # Respect encoder min width/height from caps, and ensure they are even for NV12.
    min_w = int(caps.get("width_min", 16))
    min_h = int(caps.get("height_min", 16))
    # Start from a modest baseline and clamp to supported range.
    width = max(min_w, 128)
    height = max(min_h, 72)
    if width % 2:
        width += 1
    if height % 2:
        height += 1

    fps = 24.0
    out_path = tmp_path / f"tensor_encoded_{codec}.mp4"

    encoder = PyNvVideoEncoder(
        output_path=out_path,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
        preset="P3",
        gpu_id=0,
    )

    try:
        encoder.open()
    except (nvc.PyNvVCException, nvc.PyNvVCExceptionUnsupported) as exc:
        pytest.skip(f"NVENC encoder not usable on this system: {exc}")

    # Two batches with different memory layouts, all on CUDA:
    frames_nhwc = torch.randint(
        0,
        256,
        (4, height, width, 3),
        dtype=torch.uint8,
        device="cuda",
    )
    frames_nchw = torch.randint(
        0,
        256,
        (4, 3, height, width),
        dtype=torch.uint8,
        device="cuda",
    )

    try:
        encoder.write(frames_nhwc)
        encoder.write(frames_nchw)
        encoder.close()
    except BrokenPipeError:
        # ffmpeg could not mux this elementary stream (commonly AV1 raw
        # streams on older builds). Treat as a skip for this codec.
        if codec == "av1":
            pytest.skip("ffmpeg cannot mux raw AV1 streams on this system")
        raise

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Verify container and stream properties via ffprobe.
    lines = _ffprobe_stream_info(out_path, ffmpeg)

    assert any(line == "codec_type=video" for line in lines)
    assert any(line == f"codec_name={codec}" for line in lines)
    assert any(line == f"width={width}" for line in lines)
    assert any(line == f"height={height}" for line in lines)

    # Ensure we produced a yuv420p-like format (NV12 input).
    pix_fmt_lines = [l for l in lines if l.startswith("pix_fmt=")]
    # Some FFmpeg builds may label as yuv420p or yuvj420p; accept both.
    assert any(
        l in ("pix_fmt=yuv420p", "pix_fmt=yuvj420p") for l in pix_fmt_lines
    ), f"Unexpected pix_fmt lines: {pix_fmt_lines}"


def test_pynv_video_encoder_h264_cuda(tmp_path):
    """Encode random CUDA BGR frames into an H.264 MP4 container."""
    _encode_and_check(tmp_path, "h264")


def test_pynv_video_encoder_hevc_cuda(tmp_path):
    """Encode random CUDA BGR frames into an HEVC MP4 container."""
    _encode_and_check(tmp_path, "hevc")


def test_pynv_video_encoder_av1_cuda(tmp_path):
    """Encode random CUDA BGR frames into an AV1 MP4 container."""
    _encode_and_check(tmp_path, "av1")
