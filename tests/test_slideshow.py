import asyncio
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
from fastapi import HTTPException

from app.config import Settings
from app.main import app
from app.slideshow import Slide, parse_timeline, write_manifest
from test_docs import get
from test_video import post_multipart

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "concat_20261012.txt"
TIMELINE = b"ffconcat version 1.0\nfile 'C:\\photos\\first.jpg'\nduration 5.000\nfile 'second.png'\nduration 8.000\n"


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None)

    def test_user_example_is_preserved(self):
        slides = parse_timeline(EXAMPLE.read_bytes(), self.settings)
        self.assertEqual(len(slides), 14)
        self.assertEqual(slides[0], Slide("IMG_20261012_084558.jpg", Decimal("1224.000")))
        self.assertEqual(slides[-1].duration, Decimal("1906.000"))
        self.assertEqual(sum(s.duration for s in slides), Decimal("5164.000"))

    def test_bom_comments_and_optional_final_duplicate(self):
        slides = parse_timeline(b"\xef\xbb\xbf# comment\n" + TIMELINE + b"file 'second.png'\n", self.settings)
        self.assertEqual(len(slides), 2)
        self.assertEqual(slides[0].image, "first.jpg")
        self.assertEqual(slides[1].duration, Decimal(8))

    def test_invalid_or_unsupported_timeline_is_rejected(self):
        cases = [
            b"file\nduration 1224\n",
            b"file 'first.jpg'\nfile 'second.jpg'\nduration 2\n",
            b"file 'first.jpg'\n",
            b"file 'first.jpg'\nduration 0\n",
            b"file 'first.jpg'\nduration -1\n",
            b"file 'first.jpg'\nduration nan\n",
            b"file 'first.jpg'\nduration infinity\n",
            b"file 'first.jpg'\nduration 1\noption protocol_whitelist http\n",
            b"file 'unterminated\nduration 1\n",
            b"\xff\xfe",
        ]
        for content in cases:
            with self.subTest(content=content):
                with self.assertRaises(HTTPException) as error:
                    parse_timeline(content, self.settings)
                self.assertEqual(error.exception.status_code, 422)

    def test_count_and_aggregate_duration_limits(self):
        self.settings.ffmpeg_slideshow_max_images = 1
        with self.assertRaises(HTTPException) as error:
            parse_timeline(TIMELINE, self.settings)
        self.assertEqual(error.exception.status_code, 413)
        self.settings.ffmpeg_slideshow_max_images = 100
        self.settings.ffmpeg_slideshow_max_duration_seconds = 10
        with self.assertRaises(HTTPException) as error:
            parse_timeline(TIMELINE, self.settings)
        self.assertEqual(error.exception.status_code, 413)

    def test_manifest_preserves_order_and_last_duration_without_client_paths(self):
        slides = parse_timeline(TIMELINE, self.settings)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "concat.ffconcat"
            write_manifest(manifest, slides, {"first.jpg": "slide-0001.png", "second.png": "slide-0000.png"})
            self.assertEqual(manifest.read_text(), (
                "ffconcat version 1.0\nfile 'slide-0001.png'\nduration 5.000000\n"
                "file 'slide-0000.png'\nduration 8.000000\nfile 'slide-0000.png'\n"
            ))


class SlideshowRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = Settings(_env_file=None, ffmpeg_temp_dir=self.root, app_api_key="")
        self.settings_patch = patch("app.slideshow.get_settings", return_value=self.settings)
        self.auth_patch = patch("app.auth.get_settings", return_value=self.settings)
        self.settings_patch.start()
        self.auth_patch.start()
        self.previous_slots = getattr(app.state, "video_slots", None)
        app.state.video_slots = anyio.Semaphore(1)
        self.commands = []
        self.manifest = None

        async def fake_ffmpeg(command, settings):
            self.commands.append(command)
            source = Path(command[command.index("-i") + 1])
            self.assertTrue(source.is_file())
            if "concat" in command:
                self.manifest = source.read_text()
                for line in self.manifest.splitlines():
                    if line.startswith("file "):
                        self.assertTrue((source.parent / line[6:-1]).is_file())
            Path(command[-1]).write_bytes(b"rendered data")

        self.process = AsyncMock(side_effect=fake_ffmpeg)
        self.process_patch = patch("app.slideshow.run_ffmpeg", self.process)
        self.process_patch.start()

    async def asyncTearDown(self):
        self.process_patch.stop()
        self.auth_patch.stop()
        self.settings_patch.stop()
        if self.previous_slots is None:
            del app.state.video_slots
        else:
            app.state.video_slots = self.previous_slots
        self.temporary.cleanup()

    async def upload(self, timeline=TIMELINE, images=None, fields=None, **kwargs):
        files = [("timeline", "concat.txt", timeline), ("audio", "recording.m4a", b"audio data")]
        files.extend(("images", name, content) for name, content in (
            images if images is not None else [("first.jpg", b"jpg data"), ("second.png", b"png data")]
        ))
        return await post_multipart("/v1/video/slideshow", files, fields, **kwargs)

    async def test_upload_order_does_not_change_timeline_and_mp4_is_returned(self):
        status, headers, body = await self.upload(images=[("SECOND.PNG", b"png"), ("first.jpg", b"jpg")])
        self.assertEqual(status, 200)
        self.assertEqual(body, b"rendered data")
        self.assertEqual(headers[b"content-type"], b"video/mp4")
        self.assertIn(b"slideshow.mp4", headers[b"content-disposition"])
        self.assertIn("file 'slide-0001.png'\nduration 5.000000", self.manifest)
        self.assertTrue(self.manifest.endswith("file 'slide-0000.png'\n"))
        command = self.commands[-1]
        self.assertIn("h264_nvenc", command)
        self.assertEqual(command[command.index("-preset") + 1], "p7")
        self.assertEqual(command[command.index("-c:a") + 1], "copy")
        self.assertEqual(command[command.index("-t") + 1], "13.000000")
        self.assertIn("-shortest", command)
        self.assertNotIn("-r", command)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(app.state.video_slots.value, 1)

    async def test_one_photo_repeated_in_timeline_is_normalized_only_once(self):
        status, _, _ = await self.upload(
            timeline=TIMELINE.replace(b"second.png", b"FIRST.JPG"),
            images=[("first.jpg", b"jpg")], fields={"audio_codec": "aac", "codec": "hevc", "fps": 30},
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(self.commands), 2)
        self.assertIn("hevc_nvenc", self.commands[-1])
        self.assertIn("format=yuv420p,fps=30", self.commands[-1])
        self.assertEqual(self.commands[-1][self.commands[-1].index("-c:a") + 1], "aac")

    async def test_missing_extra_and_duplicate_photos_rejected(self):
        cases = [
            [("first.jpg", b"jpg")],
            [("first.jpg", b"jpg"), ("second.png", b"png"), ("extra.jpg", b"jpg")],
            [("first.jpg", b"jpg"), ("FIRST.JPG", b"jpg"), ("second.png", b"png")],
        ]
        for images in cases:
            with self.subTest(images=images):
                status, _, body = await self.upload(images=images)
                self.assertEqual(status, 422, body)
        self.process.assert_not_awaited()
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_combined_upload_limit_and_empty_images(self):
        self.settings.ffmpeg_max_upload_mb = 1
        status, _, _ = await self.upload(images=[("first.jpg", b"x" * 600000), ("second.png", b"x" * 600000)])
        self.assertEqual(status, 413)
        status, _, _ = await self.upload(images=[("first.jpg", b""), ("second.png", b"png")])
        self.assertEqual(status, 400)
        self.process.assert_not_awaited()
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_invalid_options_and_subframe_duration_rejected(self):
        for fields in ({"width": 1919}, {"fps": 0}, {"audio_codec": "shell"}, {"width": 3840, "height": 3840}):
            status, _, _ = await self.upload(fields=fields)
            self.assertEqual(status, 422)
        status, _, _ = await self.upload(timeline=TIMELINE.replace(b"5.000", b"0.001"))
        self.assertEqual(status, 422)
        self.process.assert_not_awaited()

    async def test_auth_and_shared_video_queue(self):
        self.settings.app_api_key = type(self.settings.app_api_key)("secret")
        self.assertEqual((await self.upload())[0], 401)
        self.settings.ffmpeg_queue_timeout_seconds = 0.01
        await app.state.video_slots.acquire()
        try:
            status, headers, _ = await self.upload(headers=[(b"authorization", b"Bearer secret")])
            self.assertEqual(status, 503)
            self.assertEqual(headers[b"retry-after"], b"5")
            self.assertEqual(app.state.video_slots.value, 0)
        finally:
            app.state.video_slots.release()
        self.process.assert_not_awaited()

    async def test_ffmpeg_failure_cleans_up_and_releases_shared_slot(self):
        self.process.side_effect = HTTPException(422, "invalid image")
        self.assertEqual((await self.upload())[0], 422)
        self.assertEqual(app.state.video_slots.value, 1)
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_whole_render_timeout_and_external_cancellation_clean_up(self):
        started = asyncio.Event()

        async def wait(command, settings):
            Path(command[-1]).write_bytes(b"partial")
            started.set()
            await anyio.sleep_forever()

        self.process.side_effect = wait
        self.settings.ffmpeg_slideshow_timeout_seconds = 0.01
        status, _, body = await self.upload()
        self.assertEqual(status, 504)
        self.assertIn(b"FFMPEG_SLIDESHOW_TIMEOUT_SECONDS", body)
        self.settings.ffmpeg_slideshow_timeout_seconds = 60
        started.clear()
        task = asyncio.create_task(self.upload())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(app.state.video_slots.value, 1)
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_response_send_failure_and_output_limit_clean_up(self):
        with self.assertRaises(ConnectionError):
            await self.upload(fail_sending=True)
        self.assertEqual(list(self.root.iterdir()), [])
        self.settings.ffmpeg_max_output_mb = 1

        async def too_large(command, settings):
            with Path(command[-1]).open("wb") as output:
                output.truncate(1024 * 1024)

        self.process.side_effect = too_large
        self.assertEqual((await self.upload())[0], 413)
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_docs_expose_multipart_uploads_and_binary_response(self):
        status, _, body = await get("/openapi.json")
        self.assertEqual(status, 200)
        schema = json.loads(body)
        operation = schema["paths"]["/v1/video/slideshow"]["post"]
        self.assertIn("video/mp4", operation["responses"]["200"]["content"])
        reference = operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
        fields = schema["components"]["schemas"][reference.rsplit("/", 1)[1]]["properties"]
        self.assertEqual(fields["images"]["type"], "array")
        for field in (fields["audio"], fields["timeline"], fields["images"]["items"]):
            # FastAPI versions use either OpenAPI binary format or contentMediaType.
            self.assertTrue(field.get("format") == "binary" or field.get("contentMediaType") == "application/octet-stream")


if __name__ == "__main__":
    unittest.main()
