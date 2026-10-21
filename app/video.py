"""Upload-only NVIDIA video transcoding; no shell or caller-supplied FFmpeg flags."""

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Annotated, Literal

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.auth import require_api_key
from app.config import PROJECT_ROOT, Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["video"], dependencies=[Depends(require_api_key)])
Codec = Literal["h264", "hevc"]
Preset = Literal["p1", "p2", "p3", "p4", "p5", "p6", "p7"]


class TemporaryVideoResponse(FileResponse):
    """Keep the output alive while sending; also clean up if sending fails."""

    def __init__(self, path: Path, temporary: tempfile.TemporaryDirectory):
        super().__init__(path, media_type="video/mp4", filename="transcoded.mp4")
        self.temporary = temporary

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(self.temporary.cleanup)


def build_command(
    settings: Settings, source: Path, target: Path, *, codec: Codec,
    preset: Preset, quality: int, height: int | None,
) -> list[str]:
    executable = settings.ffmpeg_path
    # Bare names use PATH; relative executable paths use the project root.
    if not Path(executable).is_absolute() and ("/" in executable or "\\" in executable):
        executable = str(PROJECT_ROOT / executable)
    # Decode, convert to 8-bit 4:2:0 and optionally scale on the selected GPU.
    scale = "scale_cuda=format=yuv420p"
    if height is not None:
        scale = f"scale_cuda=w=-2:h={height}:format=yuv420p"
    return [
        executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        # Only self-contained media containers; deny network/playlist inputs.
        "-protocol_whitelist", "file,pipe",
        "-format_whitelist", "mov,matroska,webm,avi,mpegts,mpeg,flv,ogg",
        "-hwaccel", "cuda", "-hwaccel_device", str(settings.ffmpeg_gpu_index),
        "-hwaccel_output_format", "cuda", "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
        "-vf", scale, "-c:v", f"{codec}_nvenc", "-gpu", str(settings.ffmpeg_gpu_index),
        "-preset", preset, "-rc", "vbr", "-cq", str(quality), "-b:v", "0",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-map_metadata", "-1", "-map_chapters", "-1", "-movflags", "+faststart",
        # FFmpeg may exceed -fs slightly; reject the result if it hits the cap.
        "-fs", str(settings.ffmpeg_max_output_mb * 1024 * 1024),
        *(["-tag:v", "hvc1"] if codec == "hevc" else []), str(target),
    ]


async def run_ffmpeg(command: list[str], settings: Settings) -> None:
    # Store diagnostics on disk rather than accumulating unbounded stderr in RAM.
    with tempfile.TemporaryFile() as errors:
        try:
            with anyio.fail_after(settings.ffmpeg_timeout_seconds):
                result = await anyio.run_process(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=errors, check=False,
                )
        except TimeoutError:
            # anyio.run_process kills and reaps the child on cancellation.
            logger.warning("Video transcode exceeded FFmpeg timeout")
            raise HTTPException(504, "Video transcode exceeded FFMPEG_TIMEOUT_SECONDS") from None
        except OSError:
            logger.exception("Cannot start FFmpeg; check FFMPEG_PATH")
            raise HTTPException(503, "Cannot start FFmpeg; check FFMPEG_PATH and server logs") from None
        if result.returncode != 0:
            errors.seek(0, 2)
            errors.seek(max(0, errors.tell() - 6000))
            logger.error("FFmpeg exited with %s: %s", result.returncode, errors.read().decode(errors="replace"))
            raise HTTPException(
                422, "Video transcode failed; check input, NVIDIA driver, FFmpeg CUDA/NVENC support and server logs",
            )


@router.post(
    "/v1/video/transcode", response_class=FileResponse,
    responses={200: {"content": {"video/mp4": {"schema": {"type": "string", "format": "binary"}}}}},
    description="上传视频，使用 NVIDIA CUDA/NVENC 转码并返回 MP4 文件；可选等比例缩放。",
)
async def transcode_video(
    request: Request,
    file: Annotated[UploadFile, File(description="自包含视频文件，如 mp4/mov/mkv/webm/avi/ts/flv/ogv")],
    codec: Annotated[Codec, Form(description="输出视频编码；h264 兼容性较好，hevc 为 H.265")] = "h264",
    preset: Annotated[Preset, Form(description="p1 最快，p7 更重视压缩质量；默认 p4")] = "p4",
    quality: Annotated[int, Form(ge=1, le=51, description="CQ 质量值，越低画质越好、通常体积越大")] = 23,
    height: Annotated[int | None, Form(ge=144, le=4320, multiple_of=2, description="目标高度，偶数；留空保留原始尺寸，宽度自动计算")] = None,
) -> FileResponse:
    settings = get_settings()
    temporary = None
    transferred = False
    acquired = False
    slots = getattr(request.app.state, "video_slots", None)
    started = time.monotonic()
    try:
        if not settings.ffmpeg_enabled:
            raise HTTPException(503, "Video transcoding is disabled by FFMPEG_ENABLED")
        if slots is None:
            raise HTTPException(503, "Service not ready")
        if file.size is not None and file.size > settings.ffmpeg_max_upload_mb * 1024 * 1024:
            raise HTTPException(413, "Video file exceeds FFMPEG_MAX_UPLOAD_MB")
        try:
            with anyio.fail_after(settings.ffmpeg_queue_timeout_seconds):
                await slots.acquire()
                acquired = True
        except TimeoutError:
            raise HTTPException(503, "Video capacity exhausted; retry later", headers={"Retry-After": "5"}) from None

        parent = settings.ffmpeg_temp_dir
        if parent is not None:
            if not parent.is_absolute():
                parent = PROJECT_ROOT / parent
            parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="asr-fast-video-", dir=parent)
        source, target = Path(temporary.name) / "input.video", Path(temporary.name) / "output.mp4"
        size = 0
        async with await anyio.open_file(source, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.ffmpeg_max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, "Video file exceeds FFMPEG_MAX_UPLOAD_MB")
                await output.write(chunk)
        if not size:
            raise HTTPException(400, "Video file is empty")

        logger.info(
            "Video transcode started: bytes=%s; codec=%s; preset=%s; height=%s; gpu=%s",
            size, codec, preset, height, settings.ffmpeg_gpu_index,
        )
        await run_ffmpeg(build_command(
            settings, source, target, codec=codec, preset=preset, quality=quality, height=height,
        ), settings)
        if not target.is_file() or not target.stat().st_size:
            raise HTTPException(422, "FFmpeg produced no video")
        if target.stat().st_size >= settings.ffmpeg_max_output_mb * 1024 * 1024:
            raise HTTPException(413, "Video output reached FFMPEG_MAX_OUTPUT_MB; output discarded")
        logger.info("Video transcode complete: codec=%s; elapsed=%.1fs", codec, time.monotonic() - started)
        response = TemporaryVideoResponse(target, temporary)
        transferred = True
        return response
    except HTTPException:
        raise
    except Exception:
        logger.exception("Video transcode failed")
        raise HTTPException(500, "Video transcode failed; check server logs") from None
    finally:
        if acquired:
            slots.release()
        with anyio.CancelScope(shield=True):
            try:
                await file.close()
            finally:
                if temporary is not None and not transferred:
                    await anyio.to_thread.run_sync(temporary.cleanup)
