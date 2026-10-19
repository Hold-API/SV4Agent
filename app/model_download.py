"""Download model files with an explicit Hub endpoint, before loading the model.

Aliases and file patterns follow faster-whisper 1.2.1:
https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/utils.py
"""

from urllib.parse import urlsplit


MODEL_REPOSITORIES = {
    name: f"Systran/faster-whisper-{name}"
    for name in (
        "tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
        "large-v1", "large-v2", "large-v3", "distil-large-v2", "distil-medium.en",
        "distil-small.en", "distil-large-v3",
    )
}
MODEL_REPOSITORIES.update({
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
})


def download_model(
    model: str,
    *,
    endpoint: str,
    cache_dir: str,
    local_files_only: bool,
    token: str | None = None,
) -> str:
    endpoint = endpoint.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("HF_ENDPOINT must be an HTTP(S) URL, e.g. https://hf-mirror.com")
    repo_id = model if "/" in model else MODEL_REPOSITORIES.get(model)
    if not repo_id:
        raise ValueError(f"Unknown model alias: {model}; use a supported alias or a full HF repository ID")

    from huggingface_hub import snapshot_download

    # Explicit endpoint applies to both repository metadata and file downloads,
    # even if another module imported huggingface_hub before this application.
    # No fallback to the official Hub is performed by this application.
    return snapshot_download(
        repo_id=repo_id,
        endpoint=endpoint,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
        allow_patterns=[
            "config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*",
        ],
    )
