import logging
import secrets
import time
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated, Literal

import anyio
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.asr import AudioTooLong, InvalidAudio, UnsupportedLanguage, WhisperService
from app.config import DOTENV_MODELSCOPE_ENDPOINT, ENV_FILE, get_settings
from app.docs import setup_docs
from app.logging_config import configure_logging
from app.schemas import Transcription

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


async def report_startup(service: WhisperService, started: float) -> None:
    while True:
        await anyio.sleep(15)
        logger.info(
            "Startup still in progress: stage=%s; elapsed=%.0fs",
            service.startup_stage,
            time.monotonic() - started,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.app_log_level)
    service = WhisperService(settings)
    app.state.service = service
    app.state.slots = anyio.Semaphore(settings.app_max_concurrent_requests)
    app.state.ready = False
    started = time.monotonic()
    try:
        logger.info("Configuration file: %s; exists=%s", ENV_FILE, ENV_FILE.is_file())
        logger.info("Model download source: ModelScope; endpoint=%s", settings.modelscope_endpoint)
        if (
            DOTENV_MODELSCOPE_ENDPOINT is not None
            and DOTENV_MODELSCOPE_ENDPOINT.strip().rstrip("/") != settings.modelscope_endpoint.strip().rstrip("/")
        ):
            logger.warning("MODELSCOPE_ENDPOINT in .env is overridden by the process environment")
        logger.info("Loading model %s on %s", settings.whisper_model, settings.whisper_device)
        try:
            async with anyio.create_task_group() as startup_tasks:
                startup_tasks.start_soon(report_startup, service, started)
                try:
                    await anyio.to_thread.run_sync(service.load)
                finally:
                    startup_tasks.cancel_scope.cancel()
        except Exception:
            logger.exception("Startup failed at stage: %s", service.startup_stage)
            raise
        app.state.ready = True
        logger.info("Model ready after %.1fs; HTTP startup can complete", time.monotonic() - started)
        yield
    finally:
        app.state.ready = False
        service.close()


app = FastAPI(
    title="Faster Whisper ASR", version="0.1.0", lifespan=lifespan,
    docs_url=None, redoc_url=None,
)
setup_docs(app)


async def require_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> None:
    expected = get_settings().app_api_key.get_secret_value()
    if expected and (
        credentials is None
        or not secrets.compare_digest(credentials.credentials.encode(), expected.encode())
    ):
        raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})


@app.get("/health", tags=["health"])
async def health(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(503, "Model not ready")
    return {"status": "ok"}


@app.post(
    "/v1/audio/transcriptions",
    response_model=Transcription,
    dependencies=[Depends(require_api_key)],
    tags=["ASR"],
    description="上传音频，返回文本、语言及时间戳。translate 将语音翻译为英文。",
)
async def transcribe(
    request: Request,
    file: Annotated[UploadFile, File(description="音频文件，如 wav/mp3/m4a/flac/ogg/webm")],
    language: Annotated[str | None, Form(description="语言代码，如 zh/en；留空使用环境默认或自动检测")] = None,
    task: Annotated[Literal["transcribe", "translate"], Form()] = "transcribe",
    beam_size: Annotated[int | None, Form(ge=1, le=20)] = None,
    temperature: Annotated[float, Form(ge=0, le=1)] = 0,
    initial_prompt: Annotated[str | None, Form(max_length=2000)] = None,
    vad_filter: Annotated[bool | None, Form()] = None,
    word_timestamps: Annotated[bool | None, Form()] = None,
) -> Transcription:
    settings = get_settings()
    try:
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(503, "Model not ready")
        # Multipart is already spooled by Starlette; this bounds the audio file,
        # not the full HTTP request. Set a total body limit at the reverse proxy.
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.app_max_upload_mb * 1024 * 1024:
                raise HTTPException(413, "Audio file exceeds APP_MAX_UPLOAD_MB")
        if size == 0:
            raise HTTPException(400, "Audio file is empty")
        await file.seek(0)

        slots = request.app.state.slots
        try:
            with anyio.fail_after(settings.app_queue_timeout_seconds):
                await slots.acquire()
        except TimeoutError:
            raise HTTPException(503, "ASR capacity exhausted; retry later", headers={"Retry-After": "5"}) from None

        try:
            # Keep the slot and upload alive until native inference finishes.
            with anyio.CancelScope(shield=True):
                return await anyio.to_thread.run_sync(partial(
                    request.app.state.service.transcribe,
                    file.file,
                    language=(language or settings.whisper_language).strip() or None,
                    task=task,
                    beam_size=beam_size if beam_size is not None else settings.whisper_beam_size,
                    temperature=temperature,
                    initial_prompt=initial_prompt,
                    vad_filter=vad_filter if vad_filter is not None else settings.whisper_vad_filter,
                    word_timestamps=word_timestamps if word_timestamps is not None else settings.whisper_word_timestamps,
                ))
        finally:
            slots.release()
    except (InvalidAudio, UnsupportedLanguage) as exc:
        raise HTTPException(422, str(exc)) from exc
    except AudioTooLong as exc:
        raise HTTPException(413, str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Transcription failed")
        raise HTTPException(500, "Transcription failed; check server logs") from None
    finally:
        with anyio.CancelScope(shield=True):
            await file.close()
