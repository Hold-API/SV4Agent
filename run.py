"""Start with `python run.py`; configuration is loaded from project-root .env."""

from app.config import get_settings


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    # One process owns one GPU model; concurrency is controlled inside the app.
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.app_log_level,
        workers=1,
    )
