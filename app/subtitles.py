"""Add a validated SRT track to video, or burn captions with NVENC encoding."""

import json
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Literal

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from app.auth import require_api_key
from app.config import Settings, get_settings
from app.video import (
    Codec, Preset, TemporaryVideoResponse, create_video_directory,
    ffmpeg_executable, run_ffmpeg, validate_video_output,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["video"], dependencies=[Depends(require_api_key)])
VIDEO_FORMATS = "mov,matroska,webm,avi,mpegts,mpeg,flv,ogg"
STAMP = r"(\d{2,3}):([0-5]\d):([0-5]\d)[,.](\d{3})"
TIMING = re.compile(rf"^{STAMP}\s+-->\s+{STAMP}$")


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: str


def parse_srt(content: bytes) -> list[Cue]:
    try:
        text = content.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").strip()
    except UnicodeDecodeError:
        raise HTTPException(422, "Subtitles must be UTF-8 encoded SRT") from None
    if not text:
        raise HTTPException(400, "Subtitle file is empty")
    if "\x00" in text:
        raise HTTPException(422, "Invalid NUL character in subtitles")
    cues = []
    for index, block in enumerate(re.split(r"\n(?:[ \t]*\n)+", text), 1):
        lines = block.splitlines()
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        timing = TIMING.fullmatch(lines[0].strip()) if lines else None
        if timing is None or len(lines) < 2 or not "\n".join(lines[1:]).strip():
            raise HTTPException(422, f"Invalid SRT cue {index}; expected timestamps and caption text")
        values = [int(n) for n in timing.groups()]
        start = ((values[0] * 60 + values[1]) * 60 + values[2]) * 1000 + values[3]
        end = ((values[4] * 60 + values[5]) * 60 + values[6]) * 1000 + values[7]
        if end <= start:
            raise HTTPException(422, f"SRT cue {index} must end after its start")
        if cues and start < cues[-1].start_ms:
            raise HTTPException(422, "SRT cues must be ordered by start time")
        cues.append(Cue(start, end, "\n".join(lines[1:]).strip()))
    return cues


def srt_timestamp(milliseconds: int) -> str:
    seconds, millis = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_srt(path: Path, cues: list[Cue], duration: Decimal) -> None:
    end_ms = int(duration * 1000)
    clipped = [Cue(c.start_ms, min(c.end_ms, end_ms), c.text) for c in cues if c.start_ms < end_ms]
    if not clipped:
        raise HTTPException(422, "No subtitle cues fall within the video duration")
    blocks = [f"{index}\n{srt_timestamp(c.start_ms)} --> {srt_timestamp(c.end_ms)}\n{c.text}\n" for index, c in enumerate(clipped, 1)]
    path.write_text("\n".join(blocks), encoding="utf-8")


def ffprobe_executable(settings: Settings) -> str:
    if settings.ffprobe_path:
        return ffmpeg_executable(settings.model_copy(update={"ffmpeg_path": settings.ffprobe_path}))
    ffmpeg = Path(ffmpeg_executable(settings))
    name = "ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe"
    return str(ffmpeg.with_name(name))


async def video_duration(source: Path, settings: Settings) -> Decimal:
    command = [
        ffprobe_executable(settings), "-v", "error", "-protocol_whitelist", "file,pipe",
        "-format_whitelist", VIDEO_FORMATS, "-select_streams", "v:0",
        "-show_entries", "stream=duration:format=duration", "-of", "json", str(source),
    ]
    with tempfile.TemporaryFile() as errors:
        try:
            with anyio.fail_after(settings.ffprobe_timeout_seconds):
                result = await anyio.run_process(command, stdin=subprocess.DEVNULL, stderr=errors, check=False)
        except TimeoutError:
            raise HTTPException(504, "Video probe exceeded FFPROBE_TIMEOUT_SECONDS") from None
        except OSError:
            logger.exception("Cannot start FFprobe")
            raise HTTPException(503, "Cannot start FFprobe; install ffprobe alongside ffmpeg or set FFPROBE_PATH") from None
        if result.returncode != 0:
            errors.seek(0, 2)
            errors.seek(max(0, errors.tell() - 6000))
            logger.error("FFprobe failed: %s", errors.read().decode(errors="replace"))
            raise HTTPException(422, "Cannot inspect video; check input and server logs")
    try:
        info = json.loads(result.stdout)
        if (
            not isinstance(info, dict)
            or not isinstance(info.get("streams"), list)
            or not info["streams"]
            or not isinstance(info["streams"][0], dict)
        ):
            raise ValueError("No video stream")
        format_info = info.get("format")
        candidates = (
            info["streams"][0].get("duration"),
            format_info.get("duration") if isinstance(format_info, dict) else None,
        )
        for value in candidates:
            try:
                duration = Decimal(str(value))
            except InvalidOperation:
                continue
            if duration.is_finite() and duration > 0:
                return duration
    except (ValueError, TypeError, KeyError, IndexError):
        pass
    raise HTTPException(422, "Cannot determine a finite video duration")


def subtitle_command(
    settings: Settings, source: Path, target: Path, duration: Decimal, *, subtitle_mode: Literal["soft", "burn"],
    codec: Codec, preset: Preset, quality: int, fps: int, font_size: int,
    audio_codec: Literal["copy", "aac"], subtitle_language: str,
) -> list[str]:
    # The process runs in its private temp directory. Filter paths contain no
    # Windows drive letters, user filenames or other filtergraph syntax.
    command = [
        ffmpeg_executable(settings), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-protocol_whitelist", "file,pipe", "-format_whitelist", VIDEO_FORMATS, "-i", str(source),
    ]
    if subtitle_mode == "soft":
        command += [
            "-protocol_whitelist", "file,pipe", "-f", "srt", "-i", "captions.srt",
            "-map", "0:v:0", "-map", "0:a:0?", "-map", "1:s:0",
            "-c:v", "copy", "-c:s", "mov_text", "-disposition:s:0", "default",
            "-metadata:s:s:0", f"language={subtitle_language}",
        ]
    else:
        style = f"FontName={settings.ffmpeg_subtitle_font_name},FontSize={font_size},Outline=2,Shadow=0,Alignment=2,MarginV=20"
        command += [
            "-map", "0:v:0", "-map", "0:a:0?", "-sn",
            # Duplicate low-fps frames BEFORE drawing text so short captions
            # still appear at their own timestamps on a 1-fps slideshow.
            "-vf", f"fps={fps},subtitles=filename=captions.srt:force_style='{style}',format=yuv420p",
            "-c:v", f"{codec}_nvenc", "-gpu", str(settings.ffmpeg_gpu_index),
            "-preset", preset, "-rc", "vbr", "-cq", str(quality), "-b:v", "0",
            *(["-tag:v", "hvc1"] if codec == "hevc" else []),
        ]
    # Do not use -shortest: a short subtitle track must not truncate the video.
    return command + [
        "-c:a", audio_codec, *(["-b:a", "192k", "-ac", "2"] if audio_codec == "aac" else []),
        "-dn", "-t", f"{duration:.6f}", "-movflags", "+faststart",
        "-fs", str(settings.ffmpeg_max_output_mb * 1024 * 1024), str(target),
    ]


@router.post(
    "/v1/video/subtitles", response_class=FileResponse,
    responses={200: {"content": {"video/mp4": {"schema": {"type": "string", "format": "binary"}}}}},
    description="上传视频和 UTF-8 SRT。soft 添加可开关字幕并复制视频流；burn 烧录字幕并以 NVENC 重新编码。subtitle_mode 可通过 URL 查询参数或表单传入，URL 优先；均未指定时为 soft。其余选项通过表单传入。",
)
async def add_subtitles(
    request: Request,
    file: Annotated[UploadFile, File(description="视频文件")],
    subtitles: Annotated[UploadFile, File(description="UTF-8 编码的 SRT 字幕文件")],
    subtitle_mode: Annotated[Literal["soft", "burn"], Form(description="字幕模式；URL 中的同名参数优先")] = "soft",
    codec: Annotated[Codec, Form(description="仅 burn 模式使用")] = "h264",
    preset: Annotated[Preset, Form(description="仅 burn 模式使用")] = "p4",
    quality: Annotated[int, Form(ge=1, le=51, description="仅 burn 模式使用")] = 23,
    fps: Annotated[int, Form(ge=1, le=60, description="仅 burn 模式使用；先补帧再渲染字幕")] = 24,
    font_size: Annotated[int, Form(ge=8, le=96, description="仅 burn 模式使用；ASS 字号")] = 24,
    audio_codec: Annotated[Literal["copy", "aac"], Form()] = "copy",
    subtitle_language: Annotated[str, Form(pattern=r"^[a-z]{3}$", description="soft 模式 ISO 639-2 三字母语言代码")] = "zho",
    subtitle_mode_query: Annotated[Literal["soft", "burn"] | None, Query(alias="subtitle_mode", description="字幕模式；填写时优先于表单，省略时使用表单值（默认 soft）")] = None,
) -> FileResponse:
    if subtitle_mode_query is not None:
        subtitle_mode = subtitle_mode_query
    settings = get_settings()
    temporary = None
    transferred = False
    acquired = False
    slots = getattr(request.app.state, "video_slots", None)
    started = time.monotonic()
    try:
        if not settings.ffmpeg_enabled:
            raise HTTPException(503, "Video processing is disabled by FFMPEG_ENABLED")
        if slots is None:
            raise HTTPException(503, "Service not ready")
        limit = settings.ffmpeg_subtitle_max_upload_kb * 1024
        content = await subtitles.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(413, "Subtitles exceed FFMPEG_SUBTITLE_MAX_UPLOAD_KB")
        cues = parse_srt(content)
        total = len(content)
        if total + (file.size or 0) > settings.ffmpeg_max_upload_mb * 1024 * 1024:
            raise HTTPException(413, "Video and subtitles exceed FFMPEG_MAX_UPLOAD_MB")
        try:
            with anyio.fail_after(settings.ffmpeg_queue_timeout_seconds):
                await slots.acquire()
                acquired = True
        except TimeoutError:
            raise HTTPException(503, "Video capacity exhausted; retry later", headers={"Retry-After": "5"}) from None
        temporary = create_video_directory(settings)
        root = Path(temporary.name)
        source, target = root / "input.video", root / "subtitled.mp4"
        size = 0
        async with await anyio.open_file(source, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                total += len(chunk)
                if total > settings.ffmpeg_max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, "Video and subtitles exceed FFMPEG_MAX_UPLOAD_MB")
                await output.write(chunk)
        if not size:
            raise HTTPException(400, "Video file is empty")
        execution_settings = settings.model_copy(update={"ffmpeg_timeout_seconds": settings.ffmpeg_subtitle_timeout_seconds + 1})
        try:
            with anyio.fail_after(settings.ffmpeg_subtitle_timeout_seconds):
                duration = await video_duration(source, settings)
                write_srt(root / "captions.srt", cues, duration)
                logger.info("Add subtitles: mode=%s; duration=%ss; cues=%s", subtitle_mode, duration, len(cues))
                await run_ffmpeg(subtitle_command(
                    execution_settings, source, target, duration, subtitle_mode=subtitle_mode,
                    codec=codec, preset=preset, quality=quality, fps=fps, font_size=font_size,
                    audio_codec=audio_codec, subtitle_language=subtitle_language,
                ), execution_settings, cwd=root)
        except TimeoutError:
            raise HTTPException(504, "Subtitles processing exceeded FFMPEG_SUBTITLE_TIMEOUT_SECONDS") from None
        validate_video_output(target, settings)
        logger.info("Subtitles complete: mode=%s; elapsed=%.1fs", subtitle_mode, time.monotonic() - started)
        response = TemporaryVideoResponse(target, temporary, filename="subtitled.mp4")
        transferred = True
        return response
    except HTTPException:
        raise
    except Exception:
        logger.exception("Adding subtitles failed")
        raise HTTPException(500, "Adding subtitles failed; check server logs") from None
    finally:
        if acquired:
            slots.release()
        with anyio.CancelScope(shield=True):
            try:
                await file.close()
                await subtitles.close()
            finally:
                if temporary is not None and not transferred:
                    await anyio.to_thread.run_sync(temporary.cleanup)
