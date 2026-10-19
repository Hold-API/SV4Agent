import logging


def configure_logging(level: str) -> None:
    """Uvicorn configures its own loggers, so configure our namespace explicitly."""
    logger = logging.getLogger("app")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        ))
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
