import subprocess
from pathlib import Path

import pytest


def _ensure_ffmpeg() -> str:
    """Return ffmpeg executable path or skip if not available."""
    from shutil import which

    path = which("ffmpeg")
    if path is None:
        pytest.skip("ffmpeg is required for GPU decode test")
    return path


def _generate_test_video(tmp_path: Path) -> Path:
    """Generate a small H.264 MP4 clip for decoding tests."""
    ffmpeg = _ensure_ffmpeg()
    out = tmp_path / "pynv_test.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=128x72:rate=24",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        pytest.skip(f"ffmpeg failed to generate test clip: {exc}")
    return out


def test_gpu_decode_single_frame(tmp_path):
    """End-to-end GPU-backed decode of at least one frame."""
    import PyNvVideoCodec as nvc

    video_path = _generate_test_video(tmp_path)

    demuxer = nvc.CreateDemuxer(filename=str(video_path))

    decoder = nvc.CreateDecoder(
        gpuid=0,
        codec=demuxer.GetNvCodecId(),
        cudacontext=0,
        cudastream=0,
        usedevicememory=1,
        latency=nvc.DisplayDecodeLatencyType.NATIVE,
    )

    decoded_frames = 0
    for packet in demuxer:
        for _frame in decoder.Decode(packet):
            decoded_frames += 1
            if decoded_frames >= 1:
                break
        if decoded_frames >= 1:
            break

    assert decoded_frames >= 1
