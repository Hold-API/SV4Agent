"""Load .env before importing CTranslate2 or Hugging Face libraries."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Hub constants are read on import. These must be set even without a .env file.
for name, value in {
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HOME": str(PROJECT_ROOT / ".cache" / "huggingface"),
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_ETAG_TIMEOUT": "30",
    "HF_HUB_DOWNLOAD_TIMEOUT": "120",
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

    whisper_model: str = Field(default="large-v3", min_length=1)
    whisper_download_root: Path = PROJECT_ROOT / "models"
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
