import subprocess
from pathlib import Path

import pytest


def _ensure_ffmpeg() -> str:
    """Return ffmpeg executable path or skip if not available."""
    from shutil import which

    path = which("ffmpeg")
    if path is None:
        pytest.skip("ffmpeg is required for muxer remux test")
    return path


def _generate_test_video(tmp_path: Path) -> Path:
    """Generate a small H.264 MP4 clip for muxer tests."""
    ffmpeg = _ensure_ffmpeg()
    out = tmp_path / "pynv_muxer_input.mp4"
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


def _ffprobe_streams(path: Path) -> list[str]:
    ffmpeg = _ensure_ffmpeg()
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type,codec_name,width,height",
        "-of",
        "default=nw=1",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()


def test_muxer_remux_video_only(tmp_path):
    """Use PyNvDemuxer + PyNvMuxer to remux a short MP4 clip."""
    import PyNvVideoCodec as nvc

    input_path = _generate_test_video(tmp_path)
    output_path = tmp_path / "pynv_muxer_output.mp4"

    demuxer = nvc.CreateDemuxer(filename=str(input_path))
    muxer = nvc.CreateMuxer(filename=str(output_path), demuxer=demuxer)

    # Remux all video packets from the input into the new container.
    for packet in demuxer:
        muxer.Mux(packet)

    # Finalize the container to ensure trailer is written.
    muxer.Close()

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Verify ffprobe can read the muxed container and basic stream properties.
    lines = _ffprobe_streams(output_path)

    assert any(line == "codec_type=video" for line in lines)
    assert any(line.startswith("codec_name=") for line in lines)
    assert any(line.startswith("width=") for line in lines)
    assert any(line.startswith("height=") for line in lines)

