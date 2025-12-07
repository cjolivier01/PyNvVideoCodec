import subprocess
from pathlib import Path

import numpy as np
import pytest


def _ensure_ffmpeg() -> str:
    """Return ffmpeg executable path or skip if not available."""
    from shutil import which

    path = which("ffmpeg")
    if path is None:
        pytest.skip("ffmpeg is required for muxer encode-container test")
    return path


def _ffprobe_stream_info(path: Path, ffmpeg: str) -> list[str]:
    """Run ffprobe and return lines describing the first video stream."""
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


def _decode_frame_count(path: Path) -> int:
    """Decode the video with PyNvVideoCodec and return the number of frames."""
    import PyNvVideoCodec as nvc

    try:
        demuxer = nvc.CreateDemuxer(filename=str(path))
    except (nvc.PyNvVCException, nvc.PyNvVCExceptionUnsupported) as exc:
        pytest.skip(f"Decoder not usable for container {path}: {exc}")

    try:
        decoder = nvc.CreateDecoder(
            gpuid=0,
            codec=demuxer.GetNvCodecId(),
            cudacontext=0,
            cudastream=0,
            usedevicememory=1,
            latency=nvc.DisplayDecodeLatencyType.NATIVE,
        )
    except (nvc.PyNvVCException, nvc.PyNvVCExceptionUnsupported) as exc:
        pytest.skip(f"Decoder not usable for stream {path}: {exc}")

    decoded_frames = 0
    try:
        for packet in demuxer:
            for _frame in decoder.Decode(packet):
                decoded_frames += 1
    except (nvc.PyNvVCException, nvc.PyNvVCExceptionUnsupported) as exc:
        pytest.skip(f"Decoder failed while reading {path}: {exc}")

    return decoded_frames


def _skip_if_codec_unsupported(codec: str) -> dict:
    """Return encoder caps or skip if NVENC does not support the requested codec."""
    import PyNvVideoCodec as nvc

    try:
        caps = nvc.GetEncoderCaps(codec=codec)
    except (nvc.PyNvVCException, nvc.PyNvVCExceptionUnsupported):
        pytest.skip(f"Codec {codec} not supported by NVENC on this system")
    if not caps or caps.get("num_encoder_engines", 0) <= 0:
        pytest.skip(f"Codec {codec} not supported by NVENC on this system")
    return caps


def test_muxer_encode_to_container_h264(tmp_path: Path):
    """Encode synthetic NV12 frames, mux into a container via PyNvMuxer, and validate with ffprobe."""
    import PyNvVideoCodec as nvc

    codec = "h264"
    caps = _skip_if_codec_unsupported(codec)
    ffmpeg = _ensure_ffmpeg()

    # Respect encoder min width/height from caps, and ensure they are even for NV12.
    min_w = int(caps.get("width_min", 16))
    min_h = int(caps.get("height_min", 16))
    width = max(min_w, 128)
    height = max(min_h, 72)
    if width % 2:
        width += 1
    if height % 2:
        height += 1

    fps = 24.0
    num_frames = 4
    out_path = tmp_path / "muxer_encoded_h264.mp4"

    config: dict[str, str] = {
        "codec": codec,
        "preset": "P3",
        "gpu_id": "0",
        "fps": str(int(fps)),
    }

    # Create encoder that accepts NV12 CPU input buffers.
    try:
        encoder = nvc.CreateEncoder(width, height, "NV12", True, **config)
    except (nvc.PyNvVCExceptionUnsupported, nvc.PyNvVCException) as exc:
        pytest.skip(f"NVENC encoder not usable on this system: {exc}")

    muxer = nvc.CreateMuxer(
        filename=str(out_path),
        codec=codec,
        width=width,
        height=height,
        fps=fps,
    )

    frame_size = int(width * height * 3 / 2)
    y_size = width * height
    uv_size = frame_size - y_size

    for i in range(num_frames):
        # Simple gradient pattern in NV12: luma varies per frame, chroma fixed.
        y_plane = np.full(y_size, i * 10 % 256, dtype=np.uint8)
        uv_plane = np.full(uv_size, 128, dtype=np.uint8)
        frame = np.concatenate([y_plane, uv_plane])

        bitstream = encoder.Encode(frame)
        if bitstream:
            muxer.MuxBitstream(bitstream, is_key_frame=(i == 0))

    # Flush encoder and mux any remaining packets.
    bitstream = encoder.EndEncode()
    if bitstream:
        muxer.MuxBitstream(bitstream, is_key_frame=False)

    muxer.Close()

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Validate container and stream properties via ffprobe.
    lines = _ffprobe_stream_info(out_path, ffmpeg)

    assert any(line == "codec_type=video" for line in lines)
    assert any(line == f"codec_name={codec}" for line in lines)
    assert any(line == f"width={width}" for line in lines)
    assert any(line == f"height={height}" for line in lines)

    # Ensure no frames were dropped: decoded frame count matches encoded frames.
    decoded_frames = _decode_frame_count(out_path)
    assert decoded_frames == num_frames


def test_muxer_encode_to_container_hevc(tmp_path: Path):
    """Same as H.264 test but using HEVC codec."""
    import PyNvVideoCodec as nvc

    codec = "hevc"
    caps = _skip_if_codec_unsupported(codec)
    ffmpeg = _ensure_ffmpeg()

    min_w = int(caps.get("width_min", 16))
    min_h = int(caps.get("height_min", 16))
    width = max(min_w, 128)
    height = max(min_h, 72)
    if width % 2:
        width += 1
    if height % 2:
        height += 1

    fps = 24.0
    num_frames = 4
    out_path = tmp_path / "muxer_encoded_hevc.mp4"

    config: dict[str, str] = {
        "codec": codec,
        "preset": "P3",
        "gpu_id": "0",
        "fps": str(int(fps)),
    }

    try:
        encoder = nvc.CreateEncoder(width, height, "NV12", True, **config)
    except (nvc.PyNvVCExceptionUnsupported, nvc.PyNvVCException) as exc:
        pytest.skip(f"NVENC encoder not usable on this system: {exc}")

    muxer = nvc.CreateMuxer(
        filename=str(out_path),
        codec=codec,
        width=width,
        height=height,
        fps=fps,
    )

    frame_size = int(width * height * 3 / 2)
    y_size = width * height
    uv_size = frame_size - y_size

    for i in range(num_frames):
        y_plane = np.full(y_size, i * 10 % 256, dtype=np.uint8)
        uv_plane = np.full(uv_size, 128, dtype=np.uint8)
        frame = np.concatenate([y_plane, uv_plane])

        bitstream = encoder.Encode(frame)
        if bitstream:
            muxer.MuxBitstream(bitstream, is_key_frame=(i == 0))

    bitstream = encoder.EndEncode()
    if bitstream:
        muxer.MuxBitstream(bitstream, is_key_frame=False)

    muxer.Close()

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    lines = _ffprobe_stream_info(out_path, ffmpeg)

    assert any(line == "codec_type=video" for line in lines)
    assert any(line == f"codec_name={codec}" for line in lines)
    assert any(line == f"width={width}" for line in lines)
    assert any(line == f"height={height}" for line in lines)

    decoded_frames = _decode_frame_count(out_path)
    assert decoded_frames == num_frames


def test_muxer_encode_to_container_av1(tmp_path: Path):
    """Same as H.264 test but using AV1 codec."""
    import PyNvVideoCodec as nvc

    codec = "av1"
    caps = _skip_if_codec_unsupported(codec)
    ffmpeg = _ensure_ffmpeg()

    min_w = int(caps.get("width_min", 16))
    min_h = int(caps.get("height_min", 16))
    width = max(min_w, 128)
    height = max(min_h, 72)
    if width % 2:
        width += 1
    if height % 2:
        height += 1

    fps = 24.0
    num_frames = 4
    out_path = tmp_path / "muxer_encoded_av1.mp4"

    config: dict[str, str] = {
        "codec": codec,
        "preset": "P3",
        "gpu_id": "0",
        "fps": str(int(fps)),
    }

    try:
        encoder = nvc.CreateEncoder(width, height, "NV12", True, **config)
    except (nvc.PyNvVCExceptionUnsupported, nvc.PyNvVCException) as exc:
        pytest.skip(f"NVENC encoder not usable on this system: {exc}")

    muxer = nvc.CreateMuxer(
        filename=str(out_path),
        codec=codec,
        width=width,
        height=height,
        fps=fps,
    )

    frame_size = int(width * height * 3 / 2)
    y_size = width * height
    uv_size = frame_size - y_size

    for i in range(num_frames):
        y_plane = np.full(y_size, i * 10 % 256, dtype=np.uint8)
        uv_plane = np.full(uv_size, 128, dtype=np.uint8)
        frame = np.concatenate([y_plane, uv_plane])

        try:
            bitstream = encoder.Encode(frame)
        except BrokenPipeError:
            # Encoder could not produce a usable elementary AV1 stream.
            pytest.skip("NVENC could not produce a muxable AV1 bitstream on this system")
        if bitstream:
            muxer.MuxBitstream(bitstream, is_key_frame=(i == 0))

    try:
        bitstream = encoder.EndEncode()
    except BrokenPipeError:
        pytest.skip("NVENC could not flush a muxable AV1 bitstream on this system")
    if bitstream:
        muxer.MuxBitstream(bitstream, is_key_frame=False)

    muxer.Close()

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    lines = _ffprobe_stream_info(out_path, ffmpeg)

    assert any(line == "codec_type=video" for line in lines)
    assert any(line == f"codec_name={codec}" for line in lines)
    assert any(line == f"width={width}" for line in lines)
    assert any(line == f"height={height}" for line in lines)

    decoded_frames = _decode_frame_count(out_path)
    assert decoded_frames == num_frames
