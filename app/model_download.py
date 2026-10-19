"""Resolve a local CTranslate2 model or download it from ModelScope."""

import logging
import re
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

MODEL_REPOSITORIES = {
    name: f"Systran/faster-whisper-{name}"
    for name in (
        "tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
        "large-v1", "large-v2", "large-v3",
    )
}
MODEL_REPOSITORIES.update({
    "large": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "pengzhendong/faster-whisper-large-v3-turbo",
    "turbo": "pengzhendong/faster-whisper-large-v3-turbo",
})


def validate_model_directory(directory: str | Path) -> str:
    """Require local tokenizer files so faster-whisper cannot fetch them from HF."""
    path = Path(directory).expanduser().resolve()
    required = ("config.json", "model.bin", "tokenizer.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete CTranslate2 model directory {path}; missing: {', '.join(missing)}")
    vocabularies = [path / name for name in ("vocabulary.json", "vocabulary.txt") if (path / name).is_file()]
    if not vocabularies:
        raise ValueError(f"Incomplete CTranslate2 model directory {path}; missing vocabulary.json or vocabulary.txt")
    if any(file.stat().st_size == 0 for file in [*(path / name for name in required), *vocabularies]):
        raise ValueError(f"CTranslate2 model directory contains an empty required file: {path}")
    with (path / "model.bin").open("rb") as weights:
        if weights.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"model.bin is a Git LFS pointer, not downloaded model weights: {path}")
    return str(path)


def download_model(
    model: str,
    *,
    endpoint: str,
    cache_dir: str,
    local_files_only: bool,
    token: str | None = None,
    revision: str = "master",
    project_root: str | Path | None = None,
) -> str:
    root = Path(project_root).resolve() if project_root is not None else Path.cwd()
    candidate = Path(model).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_dir():
        logger.info("Using local model directory: %s", candidate.resolve())
        return validate_model_directory(candidate)
    if (
        Path(model).is_absolute() or PureWindowsPath(model).drive
        or model.startswith((".", "~")) or "\\" in model
    ):
        raise FileNotFoundError(f"Local model directory does not exist: {candidate}")

    endpoint = endpoint.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("MODELSCOPE_ENDPOINT must be an HTTP(S) URL, e.g. https://modelscope.cn")
    repo_id = model if "/" in model else MODEL_REPOSITORIES.get(model)
    if not repo_id or not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", repo_id):
        raise ValueError(f"Unknown model: {model}; use a supported alias or a ModelScope owner/repository ID")
    cache = Path(cache_dir).expanduser()
    if not cache.is_absolute():
        cache = root / cache

    logger.info(
        "ModelScope model=%s; endpoint=%s; revision=%s; cache=%s; local_files_only=%s",
        repo_id, endpoint, revision, cache.resolve(), local_files_only,
    )
    from modelscope import snapshot_download

    # Errors propagate without a fallback to Hugging Face. Tokens are passed
    # explicitly and never logged.
    model_path = snapshot_download(
        model_id=repo_id,
        endpoint=endpoint,
        revision=revision,
        cache_dir=str(cache.resolve()),
        local_files_only=local_files_only,
        token=token,
        allow_patterns=[
            "config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*",
        ],
    )
    return validate_model_directory(model_path)
