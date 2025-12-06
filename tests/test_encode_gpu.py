import json
import subprocess
from pathlib import Path

import numpy as np
import pytest


def _ensure_ffmpeg() -> str:
    from shutil import which

    path = which("ffmpeg")
    if path is None:
        pytest.skip("ffmpeg is required for GPU encode test")
    return path


def _generate_synthetic_nv12(tmp_path: Path, width: int, height: int, frames: int) -> Path:
    """Create a small synthetic NV12 YUV file on disk."""
    frame_size = int(width * height * 3 / 2)
    y_size = width * height
    uv_size = frame_size - y_size

    data = bytearray()
    for i in range(frames):
        # Simple gradient pattern that changes slightly per frame
        y_plane = np.full(y_size, i * 10 % 256, dtype=np.uint8)
        uv_plane = np.full(uv_size, 128, dtype=np.uint8)
        data.extend(y_plane.tobytes())
        data.extend(uv_plane.tobytes())

    path = tmp_path / "synthetic_nv12.yuv"
    path.write_bytes(data)
    return path


def _ffprobe_streams(path: Path) -> None:
    ffmpeg = _ensure_ffmpeg()
    cmd = [
        ffmpeg.replace("ffmpeg", "ffprobe"),
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
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_gpu_encode_h264_from_cpu(tmp_path):
    """Encode a few NV12 frames on GPU and verify ffmpeg can read the output."""
    import PyNvVideoCodec as nvc

    width, height, num_frames = 128, 72, 4
    raw_path = _generate_synthetic_nv12(tmp_path, width, height, num_frames)
    enc_path = tmp_path / "out.h264"

    # Load the same default encoder configuration used by the samples.
    config_path = Path("samples") / "encode_config.json"
    if not config_path.exists():
        pytest.skip("encode_config.json not found; cannot run encode test")

    config = json.loads(config_path.read_text())
    # Match sample behavior: uppercase preset, lower-case codec, explicit gpu_id.
    config["preset"] = config.get("preset", "P3").upper()
    config["codec"] = "h264"
    config["gpu_id"] = 0

    # Use the same high-level API as the samples: CPU input buffer, GPU encode.
    try:
        encoder = nvc.CreateEncoder(width, height, "NV12", True, **config)
    except (nvc.PyNvVCExceptionUnsupported, nvc.PyNvVCException) as exc:
        pytest.skip(f"NVENC encoder not usable on this system: {exc}")

    frame_size = int(width * height * 3 / 2)
    with raw_path.open("rb") as f, enc_path.open("wb") as out:
        for _ in range(num_frames):
            buf = f.read(frame_size)
            if len(buf) != frame_size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8)
            bitstream = encoder.Encode(frame)
            if bitstream:
                out.write(bytearray(bitstream))

        # Flush remaining frames
        bitstream = encoder.EndEncode()
        if bitstream:
            out.write(bytearray(bitstream))

    assert enc_path.exists()
    assert enc_path.stat().st_size > 0

    # Verify ffmpeg can parse the encoded elementary stream.
    _ffprobe_streams(enc_path)
