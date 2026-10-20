"""Serve API documentation assets locally, including behind a path prefix."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles


def setup_docs(app: FastAPI) -> None:
    app.mount(
        "/static/docs",
        StaticFiles(directory=Path(__file__).resolve().parent / "static" / "docs"),
        name="docs-assets",
    )

    @app.get("/docs", include_in_schema=False)
    async def swagger_docs(request: Request):
        root = request.scope.get("root_path", "").rstrip("/")
        return get_swagger_ui_html(
            openapi_url=f"{root}{app.openapi_url}",
            title=f"{app.title} - Swagger UI",
            swagger_js_url=f"{root}/static/docs/swagger-ui-bundle.js",
            swagger_css_url=f"{root}/static/docs/swagger-ui.css",
            swagger_favicon_url=f"{root}/static/docs/favicon.png",
            oauth2_redirect_url=f"{root}/docs/oauth2-redirect",
            swagger_ui_parameters={"validatorUrl": None},
        )

    @app.get("/docs/oauth2-redirect", include_in_schema=False)
    async def swagger_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    @app.get("/redoc", include_in_schema=False)
    async def redoc_docs(request: Request):
        root = request.scope.get("root_path", "").rstrip("/")
        return get_redoc_html(
            openapi_url=f"{root}{app.openapi_url}",
            title=f"{app.title} - ReDoc",
            redoc_js_url=f"{root}/static/docs/redoc.standalone.js",
            redoc_favicon_url=f"{root}/static/docs/favicon.png",
            with_google_fonts=False,
        )
