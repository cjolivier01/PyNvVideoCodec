import os
import subprocess
from pathlib import Path
from typing import Optional, Union

import torch

import PyNvVideoCodec as nvc


class _DLPackFrame:
    """
    Lightweight wrapper exposing a torch CUDA tensor via the DLPack protocol.

    This exists to adapt to PyNvEncoder's expectation of calling
    __dlpack__(consumer_stream) with a positional argument, while newer
    PyTorch versions require keyword-only parameters. We ignore the
    consumer stream and delegate to tensor.__dlpack__().
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def cuda(self):
        # PyNvEncoder's Encode() checks for a .cuda() method and calls it;
        # returning self keeps everything on the GPU while still allowing
        # the encoder to discover the DLPack interface on this wrapper.
        return self

    def __dlpack__(self, *args, **kwargs):
        # Ignore consumer_stream argument; rely on PyTorch defaults.
        return self._tensor.__dlpack__()

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


class PyNvVideoEncoder:
    """
    High-level GPU-only video encoder for PyNvVideoCodec.

    - Accepts BGR torch.Tensors on CUDA (single frame or batch).
    - Converts BGR -> NV12 entirely on the GPU with PyTorch.
    - Feeds NV12 frames to PyNvVideoCodec's NVENC bindings using CUDA memory.
    - Streams the elementary bitstream to ffmpeg for container muxing
      (MP4/MKV/etc., based on the output file extension).

    No raw frame data is ever moved back to the CPU; only compressed
    bitstream bytes are written on the CPU side.

    Example:
        enc = PyNvVideoEncoder(
            output_path=\"out.mp4\",
            width=1920,
            height=1080,
            fps=30.0,
            codec=\"h264\",
            preset=\"P3\",
            gpu_id=0,
        )
        enc.open()
        enc.write(batch_of_bgr_frames)  # torch.Tensor on CUDA
        enc.close()
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        width: int,
        height: int,
        fps: float = 30.0,
        codec: str = "h264",
        preset: str = "P3",
        gpu_id: int = 0,
        cuda_context: Optional[int] = None,
        cuda_stream: Optional[int] = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = codec.lower()
        self.preset = preset.upper()
        self.gpu_id = int(gpu_id)
        self.cuda_context = cuda_context
        self.cuda_stream = cuda_stream

        if self.width % 2 or self.height % 2:
            raise ValueError("Width and height must be even for NV12 (yuv420) encoding.")

        self._encoder: Optional[nvc.PyNvEncoder] = None  # type: ignore[assignment]
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._opened = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Initialize NVENC encoder and ffmpeg remuxer."""
        if self._opened:
            return

        self._encoder = self._create_encoder()
        self._ffmpeg_proc = self._spawn_ffmpeg()
        self._opened = True

    def write(self, frames: torch.Tensor) -> None:
        """
        Encode one or more BGR frames and append them to the output stream.

        Args:
            frames: torch.Tensor with shape:
                - [H, W, 3]
                - [N, H, W, 3]
                - [3, H, W]
                - [N, 3, H, W]
              Values are interpreted as 8-bit BGR (0–255) or floats in
              [0, 1] or [0, 255]. All processing stays on CUDA.
        """
        if not self._opened:
            raise RuntimeError("Encoder is not open. Call open() before write().")

        if self._encoder is None or self._ffmpeg_proc is None or self._ffmpeg_proc.stdin is None:
            raise RuntimeError("Encoder is not properly initialized.")

        batch = self._normalize_frames(frames)

        for frame in batch:
            nv12 = self._bgr_to_nv12(frame)
            # nv12 is a 2D CUDA tensor with shape [H*3/2, W], uint8.
            bitstream = self._encoder.Encode(nv12)  # type: ignore[union-attr]
            if bitstream:
                self._ffmpeg_proc.stdin.write(bytearray(bitstream))

    def close(self) -> None:
        """Flush pending frames, finalize container, and release resources."""
        if not self._opened:
            return

        if self._encoder is not None and self._ffmpeg_proc is not None and self._ffmpeg_proc.stdin is not None:
            # Flush encoder
            bitstream = self._encoder.EndEncode()  # type: ignore[union-attr]
            if bitstream:
                self._ffmpeg_proc.stdin.write(bytearray(bitstream))

            # Close ffmpeg stdin and wait for it to finish writing the container
            self._ffmpeg_proc.stdin.close()
            self._ffmpeg_proc.wait()

        self._encoder = None
        self._ffmpeg_proc = None
        self._opened = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_encoder(self) -> nvc.PyNvEncoder:  # type: ignore[override]
        """
        Create a PyNvEncoder configured for NV12 input and the requested codec.

        The encoder expects CUDA-accessible NV12 surfaces; we provide them
        via torch tensors implementing __dlpack__/__cuda_array_interface__.
        """
        config: dict[str, str] = {
            "codec": self.codec,
            "preset": self.preset,
            "fps": str(int(self.fps)),
            "gpu_id": str(self.gpu_id),
        }

        if self.cuda_context is not None:
            config["cudacontext"] = int(self.cuda_context)
        if self.cuda_stream is not None:
            config["cudastream"] = int(self.cuda_stream)

        # NV12 is efficient for NVENC and maps to yuv420p in the container.
        # Set usecpuinputbuffer=False to keep frames on CUDA.
        return nvc.CreateEncoder(self.width, self.height, "NV12", False, **config)

    def _spawn_ffmpeg(self) -> subprocess.Popen:
        """
        Launch ffmpeg to remux an elementary bitstream into a container.

        No re-encoding is done; ffmpeg simply copies the video stream into
        the requested container format based on the output file extension.
        """
        from shutil import which
        import signal
        import ctypes

        ffmpeg = which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for container muxing but was not found in PATH.")

        if self.codec == "h264":
            stream_format = "h264"
        elif self.codec == "hevc":
            stream_format = "hevc"
        elif self.codec == "av1":
            stream_format = "av1"
        else:
            raise ValueError(f"Unsupported codec for muxing: {self.codec}")

        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts",
            "-r",
            str(self.fps),
            "-f",
            stream_format,
            "-i",
            "pipe:0",
            "-c",
            "copy",
            str(self.output_path),
        ]

        kwargs = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if os.name == "posix":

            def _set_pdeathsig() -> None:
                try:
                    libc = ctypes.CDLL("libc.so.6", use_errno=True)
                    PR_SET_PDEATHSIG = 1
                    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
                except Exception:
                    # Best-effort only; ignore failures to keep encoding working.
                    pass

            kwargs["preexec_fn"] = _set_pdeathsig

        proc = subprocess.Popen(cmd, **kwargs)
        return proc

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to a CUDA tensor with shape [N, H, W, 3] and dtype uint8 (BGR).
        """
        if not isinstance(frames, torch.Tensor):
            raise TypeError("frames must be a torch.Tensor")

        # Move to the requested GPU and drop gradients; no CPU round-trip.
        device = torch.device(f"cuda:{self.gpu_id}")
        frames = frames.to(device=device, non_blocking=True).detach()

        if frames.ndim == 3:
            # HWC or CHW
            if frames.shape[0] == 3 and frames.shape[-1] != 3:
                # CHW -> HWC
                frames = frames.permute(1, 2, 0)
            frames = frames.unsqueeze(0)
        elif frames.ndim == 4:
            # NHWC or NCHW
            if frames.shape[1] == 3 and frames.shape[-1] != 3:
                # NCHW -> NHWC
                frames = frames.permute(0, 2, 3, 1)
        else:
            raise ValueError("frames must have 3 or 4 dimensions")

        if frames.shape[-1] != 3:
            raise ValueError("Last dimension must be 3 (BGR channels)")

        n, h, w, c = frames.shape
        if h != self.height or w != self.width:
            raise ValueError(f"Expected frames of shape (*, {self.height}, {self.width}, 3), got {tuple(frames.shape)}")

        # Normalize dtype to uint8 in [0, 255]
        if frames.dtype.is_floating_point:
            max_val = frames.max()
            if float(max_val) <= 1.0:
                frames = frames * 255.0
            frames = frames.clamp(0, 255).to(torch.uint8)
        elif frames.dtype != torch.uint8:
            frames = frames.to(torch.uint8)

        return frames.contiguous()

    def _bgr_to_nv12(self, frame: torch.Tensor) -> _DLPackFrame:
        """
        Convert a single BGR frame (H, W, 3) in uint8 (CUDA) to NV12 layout:
        Y plane (H x W) followed by interleaved UV (H/2 x W), all on GPU.
        """
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("Expected BGR frame with shape (H, W, 3)")

        h, w, _ = frame.shape
        if h != self.height or w != self.width:
            raise ValueError(f"Frame size mismatch: expected {self.width}x{self.height}, got {w}x{h}")

        frame = frame.contiguous()

        # Split BGR channels
        b = frame[..., 0].to(dtype=torch.float32)
        g = frame[..., 1].to(dtype=torch.float32)
        r = frame[..., 2].to(dtype=torch.float32)

        # BT.601-like conversion from BGR to YUV
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.169 * r - 0.331 * g + 0.5 * b + 128.0
        v = 0.5 * r - 0.419 * g - 0.081 * b + 128.0

        y = y.clamp(0.0, 255.0).to(torch.uint8)
        u = u.clamp(0.0, 255.0)
        v = v.clamp(0.0, 255.0)

        # 4:2:0 chroma subsampling (average over 2x2 blocks)
        u = u.view(self.height // 2, 2, self.width // 2, 2).mean(dim=(1, 3))
        v = v.view(self.height // 2, 2, self.width // 2, 2).mean(dim=(1, 3))

        u = u.clamp(0.0, 255.0).to(torch.uint8)
        v = v.clamp(0.0, 255.0).to(torch.uint8)

        # Interleave U and V to form the chroma plane (H/2 x W)
        uv = torch.empty((self.height // 2, self.width), dtype=torch.uint8, device=frame.device)
        uv[:, 0::2] = u
        uv[:, 1::2] = v

        # Stack Y and UV along the height dimension: (H + H/2, W) == (1.5 * H, W)
        nv12 = torch.cat([y, uv], dim=0)
        # Wrap in a DLPack adapter so PyNvEncoder can consume it via its
        # __dlpack__ path without bringing data back to CPU.
        return _DLPackFrame(nv12.contiguous())
