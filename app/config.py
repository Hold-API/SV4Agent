"""Load .env before importing native inference or ModelScope libraries."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
# Windows editors may save UTF-8 with a BOM. Accept both forms.
DOTENV_MODELSCOPE_ENDPOINT = dotenv_values(ENV_FILE, encoding="utf-8-sig").get("MODELSCOPE_ENDPOINT")
load_dotenv(ENV_FILE, override=False, encoding="utf-8-sig")

# ModelScope SDK timeout constants are read on import.
for name, value in {
    "MODELSCOPE_ENDPOINT": "https://modelscope.cn",
    "MODELSCOPE_API_TIMEOUT": "30",
    "MODELSCOPE_DOWNLOAD_TIMEOUT": "120",
}.items():
    os.environ.setdefault(name, value)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    app_api_key: SecretStr = SecretStr("")
    app_max_upload_mb: int = Field(default=100, ge=1)
    app_max_audio_seconds: float = Field(default=1800, gt=0)
    app_max_concurrent_requests: int = Field(default=1, ge=1, le=16)
    app_queue_timeout_seconds: float = Field(default=30, gt=0)

    ffmpeg_enabled: bool = True
    ffmpeg_path: str = Field(default="ffmpeg", min_length=1)
    ffmpeg_gpu_index: int = Field(default=0, ge=0)
    ffmpeg_max_upload_mb: int = Field(default=1024, ge=1)
    ffmpeg_max_output_mb: int = Field(default=2048, ge=1)
    ffmpeg_timeout_seconds: float = Field(default=600, gt=0)
    ffmpeg_max_concurrent_requests: int = Field(default=1, ge=1, le=8)
    ffmpeg_queue_timeout_seconds: float = Field(default=30, gt=0)
    ffmpeg_temp_dir: Path | None = None
    ffmpeg_slideshow_max_images: int = Field(default=100, ge=1, le=500)
    ffmpeg_slideshow_max_duration_seconds: float = Field(default=86400, gt=0, allow_inf_nan=False)
    ffmpeg_slideshow_timeout_seconds: float = Field(default=3600, gt=0)

    modelscope_endpoint: str = Field(default="https://modelscope.cn", min_length=1)
    modelscope_api_token: SecretStr = SecretStr("")
    modelscope_revision: str = Field(default="master", min_length=1)

    whisper_model: str = Field(default="Systran/faster-whisper-large-v3", min_length=1)
    whisper_download_root: Path = PROJECT_ROOT / "models" / "modelscope"
    whisper_local_files_only: bool = False
    whisper_device: Literal["cuda", "cpu"] = "cuda"
    whisper_device_index: int = Field(default=0, ge=0)
    whisper_compute_type: Literal[
        "float16", "float32", "int8", "int8_float16", "int8_float32", "default", "auto"
    ] = "float16"
    whisper_cpu_threads: int = Field(default=4, ge=1)
    whisper_beam_size: int = Field(default=5, ge=1, le=20)
    whisper_language: str = ""
    whisper_vad_filter: bool = True
    whisper_vad_min_silence_ms: int = Field(default=500, ge=0)
    whisper_word_timestamps: bool = False
    # Windows paths separated by semicolons, independent of os.pathsep.
    cuda_dll_dirs: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
