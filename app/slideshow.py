"""Create an NVIDIA-encoded slideshow from uploaded photos, audio and ffconcat."""

import logging
import shlex
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.auth import require_api_key
from app.config import Settings, get_settings
from app.video import (
    Codec, Preset, TemporaryVideoResponse, create_video_directory,
    ffmpeg_executable, run_ffmpeg, validate_video_output,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["video"], dependencies=[Depends(require_api_key)])
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_TIMELINE_BYTES = 256 * 1024


@dataclass(frozen=True)
class Slide:
    image: str
    duration: Decimal


def image_name(filename: str) -> str:
    # Paths supplied in concat.txt refer to client files, never server paths.
    name = PureWindowsPath(filename).name
    if not name or name in {".", ".."} or any(c in name for c in "\r\n\x00"):
        raise HTTPException(422, "Invalid image filename in timeline or upload")
    return name


def parse_timeline(content: bytes, settings: Settings) -> list[Slide]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, "Timeline must be a UTF-8 ffconcat text file") from None
    slides = []
    pending = None
    header_seen = False
    for line_number, line in enumerate(text.splitlines(), 1):
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError:
            raise HTTPException(422, f"Invalid quoting in timeline at line {line_number}") from None
        if not tokens:
            continue
        if tokens == ["ffconcat", "version", "1.0"] and not header_seen and not slides and pending is None:
            header_seen = True
            continue
        if len(tokens) == 2 and tokens[0] == "file":
            if pending is not None:
                raise HTTPException(422, f"Missing duration before line {line_number}")
            pending = image_name(tokens[1])
        elif len(tokens) == 2 and tokens[0] == "duration" and pending is not None:
            try:
                duration = Decimal(tokens[1])
            except InvalidOperation:
                raise HTTPException(422, f"Invalid duration at line {line_number}") from None
            if not duration.is_finite() or duration <= 0 or duration > Decimal(str(settings.ffmpeg_slideshow_max_duration_seconds)):
                raise HTTPException(422, f"Duration must be positive, finite and within the slideshow limit at line {line_number}")
            slides.append(Slide(pending, duration))
            pending = None
            if len(slides) > settings.ffmpeg_slideshow_max_images:
                raise HTTPException(413, "Timeline exceeds FFMPEG_SLIDESHOW_MAX_IMAGES")
        else:
            raise HTTPException(422, f"Unsupported or incomplete timeline directive at line {line_number}; use file and duration only")
    # Conventional ffconcat files may repeat the final image without a duration.
    if pending is not None and (not slides or pending.casefold() != slides[-1].image.casefold()):
        raise HTTPException(422, "Final image is missing its duration")
    if not slides:
        raise HTTPException(422, "Timeline must contain at least one image and duration")
    if sum(s.duration for s in slides) > Decimal(str(settings.ffmpeg_slideshow_max_duration_seconds)):
        raise HTTPException(413, "Timeline exceeds FFMPEG_SLIDESHOW_MAX_DURATION_SECONDS")
    return slides


def write_manifest(target: Path, slides: list[Slide], names: dict[str, str]) -> None:
    lines = ["ffconcat version 1.0"]
    for slide in slides:
        # names contains only server-generated relative filenames.
        lines.extend([f"file '{names[slide.image.casefold()]}'", f"duration {slide.duration:.6f}"])
    # A final timestamped frame is needed to preserve the last still's duration.
    lines.append(f"file '{names[slides[-1].image.casefold()]}'")
    target.write_text("\n".join(lines) + "\n", encoding="ascii")


def normalize_command(settings: Settings, source: Path, target: Path, width: int, height: int) -> list[str]:
    # Normalize each unique photo once, rather than rescaling every output frame.
    return [
        ffmpeg_executable(settings), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-protocol_whitelist", "file,pipe", "-format_whitelist", "image2,jpeg_pipe,png_pipe,webp_pipe",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=rgb24",
        "-frames:v", "1", "-c:v", "png", "-threads", "1", "-compression_level", "1", "-update", "1", str(target),
    ]


def render_command(
    settings: Settings, manifest: Path, audio: Path, target: Path, *, duration: Decimal,
    codec: Codec, preset: Preset, quality: int, fps: int, audio_codec: Literal["copy", "aac"],
) -> list[str]:
    return [
        ffmpeg_executable(settings), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-protocol_whitelist", "file,pipe", "-f", "concat", "-safe", "1", "-i", str(manifest),
        "-protocol_whitelist", "file,pipe", "-format_whitelist", "mov,aac,mp3,wav,flac,ogg,matroska,webm",
        "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-sn", "-dn",
        # Preserve the concat timestamps. An input -r would erase the durations.
        "-vf", f"format=yuv420p,fps={fps}",
        "-c:v", f"{codec}_nvenc", "-gpu", str(settings.ffmpeg_gpu_index),
        "-preset", preset, "-rc", "vbr", "-cq", str(quality), "-b:v", "0",
        "-c:a", audio_codec, *(["-b:a", "192k", "-ac", "2"] if audio_codec == "aac" else []),
        "-shortest", "-t", f"{duration:.6f}", "-map_metadata", "-1", "-map_chapters", "-1",
        "-movflags", "+faststart", "-fs", str(settings.ffmpeg_max_output_mb * 1024 * 1024),
        *(["-tag:v", "hvc1"] if codec == "hevc" else []), str(target),
    ]


@router.post(
    "/v1/video/slideshow", response_class=FileResponse,
    responses={200: {"content": {"video/mp4": {"schema": {"type": "string", "format": "binary"}}}}},
    description="上传图片、音频及 ffconcat 时间线，直接用 NVIDIA NVENC 编码为 MP4；音频和图片时间线任一结束即停止。",
)
async def slideshow(
    request: Request,
    images: Annotated[list[UploadFile], File(description="重复 images 字段上传 JPG/PNG/WebP；按文件名匹配时间线")],
    audio: Annotated[UploadFile, File(description="配音/录音，如 AAC 编码的 m4a")],
    timeline: Annotated[UploadFile, File(description="UTF-8 concat.txt；仅支持 file 和 duration，时长单位秒")],
    codec: Annotated[Codec, Form()] = "h264",
    preset: Annotated[Preset, Form()] = "p7",
    quality: Annotated[int, Form(ge=1, le=51)] = 23,
    width: Annotated[int, Form(ge=144, le=3840, multiple_of=2)] = 1920,
    height: Annotated[int, Form(ge=144, le=3840, multiple_of=2)] = 1080,
    fps: Annotated[int, Form(ge=1, le=60)] = 24,
    audio_codec: Annotated[Literal["copy", "aac"], Form(description="copy 保留音频编码，需兼容 MP4；其他输入建议 aac")] = "copy",
) -> FileResponse:
    settings = get_settings()
    temporary = None
    transferred = False
    acquired = False
    slots = getattr(request.app.state, "video_slots", None)
    try:
        if not settings.ffmpeg_enabled:
            raise HTTPException(503, "Video processing is disabled by FFMPEG_ENABLED")
        if slots is None:
            raise HTTPException(503, "Service not ready")
        if not images or len(images) > settings.ffmpeg_slideshow_max_images:
            raise HTTPException(413, "Images exceed FFMPEG_SLIDESHOW_MAX_IMAGES")
        if width * height > 3840 * 2160:
            raise HTTPException(422, "Output canvas must not exceed 3840 * 2160 pixels")
        content = await timeline.read(MAX_TIMELINE_BYTES + 1)
        if len(content) > MAX_TIMELINE_BYTES:
            raise HTTPException(413, "Timeline file exceeds 256 KiB")
        slides = parse_timeline(content, settings)
        if any(s.duration < Decimal(1) / fps for s in slides):
            raise HTTPException(422, "Each image duration must be at least one output frame (1/fps seconds)")
        uploads = {}
        for image in images:
            name = image_name(image.filename or "")
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                raise HTTPException(422, f"Unsupported image type: {name}; use JPG, PNG or WebP")
            if name.casefold() in uploads:
                raise HTTPException(422, f"Duplicate uploaded image filename: {name}")
            uploads[name.casefold()] = image
        missing = sorted({s.image for s in slides if s.image.casefold() not in uploads})
        if missing:
            raise HTTPException(422, {"message": "Timeline images not uploaded", "missing": missing})
        if set(uploads) != {s.image.casefold() for s in slides}:
            raise HTTPException(422, "Some uploaded images are not referenced by the timeline")
        total_bytes = len(content)
        if total_bytes + sum(f.size or 0 for f in [*images, audio]) > settings.ffmpeg_max_upload_mb * 1024 * 1024:
            raise HTTPException(413, "Combined slideshow uploads exceed FFMPEG_MAX_UPLOAD_MB")
        try:
            with anyio.fail_after(settings.ffmpeg_queue_timeout_seconds):
                await slots.acquire()
                acquired = True
        except TimeoutError:
            raise HTTPException(503, "Video capacity exhausted; retry later", headers={"Retry-After": "5"}) from None
        temporary = create_video_directory(settings)
        root = Path(temporary.name)

        async def save(upload: UploadFile, path: Path):
            nonlocal total_bytes
            size = 0
            async with await anyio.open_file(path, "wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > settings.ffmpeg_max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, "Combined slideshow uploads exceed FFMPEG_MAX_UPLOAD_MB")
                    await output.write(chunk)
            if not size:
                raise HTTPException(400, "Slideshow image or audio file is empty")

        sources, normalized = {}, {}
        for index, (name, image) in enumerate(uploads.items()):
            sources[name] = root / f"upload-{index:04d}{Path(image_name(image.filename)).suffix.lower()}"
            normalized[name] = f"slide-{index:04d}.png"
            await save(image, sources[name])
        audio_path = root / "audio.input"
        await save(audio, audio_path)
        manifest, target = root / "concat.ffconcat", root / "slideshow.mp4"
        write_manifest(manifest, slides, normalized)
        duration = sum(s.duration for s in slides)
        started = time.monotonic()
        # One deadline covers all normalization processes plus final encoding.
        render_settings = settings.model_copy(update={"ffmpeg_timeout_seconds": settings.ffmpeg_slideshow_timeout_seconds + 1})
        try:
            with anyio.fail_after(settings.ffmpeg_slideshow_timeout_seconds):
                for index, (name, source) in enumerate(sources.items(), 1):
                    logger.info("Slideshow: normalize photo %s/%s", index, len(sources))
                    await run_ffmpeg(normalize_command(render_settings, source, root / normalized[name], width, height), render_settings)
                logger.info("Slideshow: NVENC encoding; timeline=%ss; fps=%s; gpu=%s", duration, fps, settings.ffmpeg_gpu_index)
                await run_ffmpeg(render_command(
                    render_settings, manifest, audio_path, target, duration=duration,
                    codec=codec, preset=preset, quality=quality, fps=fps, audio_codec=audio_codec,
                ), render_settings)
        except TimeoutError:
            raise HTTPException(504, "Slideshow exceeded FFMPEG_SLIDESHOW_TIMEOUT_SECONDS") from None
        validate_video_output(target, settings)
        logger.info("Slideshow complete after %.1fs", time.monotonic() - started)
        response = TemporaryVideoResponse(target, temporary, filename="slideshow.mp4")
        transferred = True
        return response
    except HTTPException:
        raise
    except Exception:
        logger.exception("Slideshow creation failed")
        raise HTTPException(500, "Slideshow creation failed; check server logs") from None
    finally:
        if acquired:
            slots.release()
        with anyio.CancelScope(shield=True):
            try:
                for upload in [*images, audio, timeline]:
                    await upload.close()
            finally:
                if temporary is not None and not transferred:
                    await anyio.to_thread.run_sync(temporary.cleanup)
