import hashlib
import json
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from app.main import app


class AssetLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value and ((tag == "script" and key == "src") or (tag == "link" and key == "href")):
                self.urls.append(value)


async def get(path, root_path=""):
    # Exercise the real application over ASGI without running model lifespan.
    events = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(event):
        events.append(event)

    full_path = root_path + path
    await app({
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": full_path, "raw_path": full_path.encode(), "query_string": b"",
        "root_path": root_path, "headers": [(b"host", b"testserver")],
        "server": ("testserver", 80), "client": ("127.0.0.1", 12345),
    }, receive, send)
    start = next(event for event in events if event["type"] == "http.response.start")
    body = b"".join(event.get("body", b"") for event in events if event["type"] == "http.response.body")
    return start["status"], dict(start["headers"]), body


class DocumentationTests(unittest.IsolatedAsyncioTestCase):
    async def test_docs_only_reference_local_assets_that_are_served(self):
        for page in ("/docs", "/redoc"):
            with self.subTest(page=page):
                status, _, body = await get(page)
                self.assertEqual(status, 200)
                links = AssetLinks()
                links.feed(body.decode())
                self.assertTrue(links.urls)
                for url in links.urls:
                    self.assertFalse(urlsplit(url).netloc, url)
                    asset_status, headers, content = await get(url)
                    self.assertEqual(asset_status, 200, url)
                    self.assertTrue(content, url)
                    self.assertIn(b"content-type", headers)
                self.assertNotIn(b"fonts.googleapis.com", body)
                self.assertNotIn(b"cdn.jsdelivr.net", body)

    async def test_swagger_uses_local_schema_and_disables_remote_validator(self):
        _, _, body = await get("/docs")
        self.assertIn(b"url: '/openapi.json'", body)
        self.assertIn(b'"validatorUrl": null', body)
        status, _, schema = await get("/openapi.json")
        self.assertEqual(status, 200)
        paths = json.loads(schema)["paths"]
        self.assertIn("/v1/audio/transcriptions", paths)
        self.assertNotIn("/docs", paths)

    async def test_proxy_prefix_is_used_for_assets_and_schema(self):
        for page in ("/docs", "/redoc"):
            with self.subTest(page=page):
                status, _, body = await get(page, root_path="/asr")
                self.assertEqual(status, 200)
                self.assertIn(b"/asr/openapi.json", body)
                links = AssetLinks()
                links.feed(body.decode())
                for url in links.urls:
                    self.assertTrue(url.startswith("/asr/static/docs/"), url)
                    asset_status, _, _ = await get(url.removeprefix("/asr"), root_path="/asr")
                    self.assertEqual(asset_status, 200, url)

    async def test_oauth_redirect_does_not_need_external_assets(self):
        status, _, body = await get("/docs/oauth2-redirect")
        self.assertEqual(status, 200)
        self.assertIn(b"window.opener", body)
        links = AssetLinks()
        links.feed(body.decode())
        self.assertEqual(links.urls, [])

    async def test_bundled_assets_match_recorded_checksums(self):
        root = Path(__file__).resolve().parent.parent / "app" / "static" / "docs"
        manifest = json.loads((root / "manifest.json").read_text())
        for package in manifest.values():
            for filename, expected in package["files"].items():
                with self.subTest(filename=filename):
                    self.assertEqual(hashlib.sha256((root / filename).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
