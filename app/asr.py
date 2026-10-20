"""Blocking model operations; callers must dispatch them to a worker thread."""

import logging
import os
from importlib.metadata import version
from pathlib import Path
from typing import BinaryIO

from app.config import PROJECT_ROOT, Settings
from app.model_download import download_model
from app.schemas import Segment, Transcription, Word

logger = logging.getLogger(__name__)


class InvalidAudio(ValueError):
    pass


class AudioTooLong(ValueError):
    pass


class UnsupportedLanguage(ValueError):
    pass


class WhisperService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self._dll_handles = []
        self.startup_stage = "not started"

    def _stage(self, name: str) -> None:
        self.startup_stage = name
        logger.info("Startup stage: %s", name)

    def load(self) -> None:
        self._stage("configure Windows DLL directories")
        if os.name == "nt":
            directories = [p.strip() for p in self.settings.cuda_dll_dirs.split(";") if p.strip()]
            # Register DLL search directories before importing native libraries.
            for directory in directories:
                path = Path(directory).expanduser().resolve()
                if not path.is_dir():
                    raise RuntimeError(f"CUDA_DLL_DIRS directory does not exist: {path}")
                self._dll_handles.append(os.add_dll_directory(str(path)))
                logger.info("Registered DLL directory: %s", path)
            if directories:
                os.environ["PATH"] = os.pathsep.join(
                    [str(Path(p).expanduser().resolve()) for p in directories]
                    + [os.environ.get("PATH", "")]
                )

        self._stage("import CTranslate2")
        import ctranslate2

        logger.info("CTranslate2 version: %s", ctranslate2.__version__)
        self._stage("import PyAV (FFmpeg native libraries)")
        logger.info("Installed PyAV version: %s", version("av"))
        import av

        logger.info("PyAV import completed: %s", av.__version__)
        self._stage("import faster-whisper")
        from faster_whisper import WhisperModel

        settings = self.settings
        if settings.whisper_device == "cuda":
            self._stage("detect CUDA devices")
            device_count = ctranslate2.get_cuda_device_count()
            logger.info("CUDA devices detected: %s; selected index: %s", device_count, settings.whisper_device_index)
            if device_count <= settings.whisper_device_index:
                raise RuntimeError("Configured CUDA device unavailable; check NVIDIA driver and DLL paths")
            self._stage("check CUDA compute types")
            supported = ctranslate2.get_supported_compute_types("cuda", settings.whisper_device_index)
            logger.info("Supported CUDA compute types: %s", sorted(supported))
            if settings.whisper_compute_type not in {"auto", "default"} | supported:
                raise RuntimeError(f"Unsupported CUDA compute type; supported types: {sorted(supported)}")

        self._stage("resolve local model or download from ModelScope")
        model_path = download_model(
            settings.whisper_model,
            endpoint=settings.modelscope_endpoint,
            revision=settings.modelscope_revision,
            cache_dir=str(settings.whisper_download_root),
            local_files_only=settings.whisper_local_files_only,
            token=settings.modelscope_api_token.get_secret_value() or None,
            project_root=PROJECT_ROOT,
        )
        logger.info("Model files available: %s", model_path)

        self._stage("load model into device memory")
        self.model = WhisperModel(
            model_path,
            device=settings.whisper_device,
            device_index=settings.whisper_device_index,
            compute_type=settings.whisper_compute_type,
            cpu_threads=settings.whisper_cpu_threads,
            num_workers=settings.app_max_concurrent_requests,
            # Only load the validated local snapshot, including its tokenizer.
            local_files_only=True,
        )
        language = settings.whisper_language.strip() or None
        if language and language not in self.model.supported_languages:
            raise RuntimeError(f"WHISPER_LANGUAGE is unsupported by the configured model: {language}")
        self._stage("model ready")

    def close(self) -> None:
        # Release native model references before unregistering DLL search paths.
        self.model = None
        for handle in self._dll_handles:
            handle.close()
        self._dll_handles.clear()

    def transcribe(
        self,
        audio_file: BinaryIO,
        *,
        language: str | None,
        task: str,
        beam_size: int,
        temperature: float,
        initial_prompt: str | None,
        vad_filter: bool,
        word_timestamps: bool,
    ) -> Transcription:
        from av.error import FFmpegError
        from faster_whisper.audio import decode_audio

        if self.model is None:
            raise RuntimeError("Model is not loaded")
        if language and language not in self.model.supported_languages:
            raise UnsupportedLanguage(f"Unsupported language for this model: {language}")

        try:
            audio_file.seek(0)
            audio = decode_audio(audio_file, sampling_rate=16000)
        except (FFmpegError, ValueError, EOFError) as exc:
            raise InvalidAudio("Cannot decode audio; upload a supported, non-corrupt audio file") from exc
        if audio.size == 0:
            raise InvalidAudio("Audio contains no samples")
        if audio.size / 16000 > self.settings.app_max_audio_seconds:
            raise AudioTooLong("Decoded audio exceeds APP_MAX_AUDIO_SECONDS")

        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            temperature=temperature,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
            vad_parameters={"min_silence_duration_ms": self.settings.whisper_vad_min_silence_ms},
            word_timestamps=word_timestamps,
        )
        # Inference is lazy: exhaust the generator within the same worker thread.
        result = [
            Segment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=[
                    Word(start=w.start, end=w.end, word=w.word, probability=w.probability)
                    for w in segment.words
                ] if segment.words is not None else None,
            )
            for segment in segments
        ]
        return Transcription(
            text="".join(segment.text for segment in result).strip(),
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            duration_after_vad=info.duration_after_vad,
            segments=result,
        )
