import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
from fastapi import HTTPException

from app.config import Settings
from app.main import app
from app.video import run_ffmpeg


async def upload(content=b"video data", fields=None, *, headers=(), fail_sending=False):
    boundary = b"video-test-boundary"
    parts = [
        b"--" + boundary + b'\r\nContent-Disposition: form-data; name="file"; filename="input.mp4"'
        b"\r\nContent-Type: video/mp4\r\n\r\n" + content + b"\r\n",
    ]
    for name, value in (fields or {}).items():
        parts.append(b"--" + boundary + f'\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    body = b"".join(parts) + b"--" + boundary + b"--\r\n"
    events = []
    consumed = False

    async def receive():
        nonlocal consumed
        if not consumed:
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}
        await anyio.sleep_forever()

    async def send(event):
        if fail_sending and event["type"] == "http.response.body":
            raise ConnectionError("client disconnected")
        events.append(event)

    await app({
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/v1/video/transcode", "raw_path": b"/v1/video/transcode",
        "query_string": b"", "root_path": "",
        "headers": [(b"host", b"testserver"),
                    (b"content-type", b"multipart/form-data; boundary=" + boundary), *headers],
        "server": ("testserver", 80), "client": ("127.0.0.1", 12345),
    }, receive, send)
    start = next(e for e in events if e["type"] == "http.response.start")
    return start["status"], dict(start["headers"]), b"".join(e.get("body", b"") for e in events if e["type"] == "http.response.body")


class VideoRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = Settings(_env_file=None, ffmpeg_temp_dir=self.root, app_api_key="")
        self.settings_patch = patch("app.video.get_settings", return_value=self.settings)
        self.auth_patch = patch("app.auth.get_settings", return_value=self.settings)
        self.settings_patch.start()
        self.auth_patch.start()
        self.previous_slots = getattr(app.state, "video_slots", None)
        app.state.video_slots = anyio.Semaphore(1)
        self.commands = []

        async def fake_ffmpeg(command, settings):
            self.commands.append(command)
            self.assertEqual(Path(command[command.index("-i") + 1]).read_bytes(), b"video data")
            Path(command[-1]).write_bytes(b"mp4 response")

        self.process = AsyncMock(side_effect=fake_ffmpeg)
        self.process_patch = patch("app.video.run_ffmpeg", self.process)
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

    async def test_streams_mp4_and_cleans_up_after_response(self):
        status, headers, body = await upload(fields={"codec": "hevc", "height": 720})
        self.assertEqual(status, 200)
        self.assertEqual(headers[b"content-type"], b"video/mp4")
        self.assertIn(b"attachment", headers[b"content-disposition"])
        self.assertEqual(body, b"mp4 response")
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertIn("hevc_nvenc", self.commands[0])
        self.assertIn("scale_cuda=w=-2:h=720:format=yuv420p", self.commands[0])
        self.assertIn("0:a:0?", self.commands[0])
        self.assertEqual(app.state.video_slots.value, 1)

    async def test_api_key_required_before_execution(self):
        self.settings.app_api_key = type(self.settings.app_api_key)("secret")
        status, _, _ = await upload()
        self.assertEqual(status, 401)
        self.process.assert_not_awaited()
        status, _, _ = await upload(headers=[(b"authorization", b"Bearer secret")])
        self.assertEqual(status, 200)

    async def test_invalid_parameters_rejected_without_starting_ffmpeg(self):
        for fields in ({"codec": "libx264"}, {"preset": "shell"}, {"height": 721}, {"quality": 52}):
            with self.subTest(fields=fields):
                status, _, _ = await upload(fields=fields)
                self.assertEqual(status, 422)
        self.process.assert_not_awaited()

    async def test_empty_and_oversized_uploads_do_not_execute(self):
        self.settings.ffmpeg_max_upload_mb = 1
        for content, expected in ((b"", 400), (b"x" * (1024 * 1024 + 1), 413)):
            status, _, _ = await upload(content)
            self.assertEqual(status, expected)
        self.process.assert_not_awaited()
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(app.state.video_slots.value, 1)

    async def test_disabled_and_busy_return_503(self):
        self.settings.ffmpeg_enabled = False
        self.assertEqual((await upload())[0], 503)
        self.settings.ffmpeg_enabled = True
        self.settings.ffmpeg_queue_timeout_seconds = 0.01
        await app.state.video_slots.acquire()
        try:
            status, headers, _ = await upload()
            self.assertEqual(status, 503)
            self.assertEqual(headers[b"retry-after"], b"5")
            self.assertEqual(app.state.video_slots.value, 0)
        finally:
            app.state.video_slots.release()
        self.process.assert_not_awaited()

    async def test_process_error_and_output_cap_discard_partial_files(self):
        async def fail(command, settings):
            Path(command[-1]).write_bytes(b"partial")
            raise HTTPException(422, "failed")

        self.process.side_effect = fail
        self.assertEqual((await upload())[0], 422)
        self.assertEqual(list(self.root.iterdir()), [])

        async def too_large(command, settings):
            with Path(command[-1]).open("wb") as output:
                output.truncate(1024 * 1024)

        self.settings.ffmpeg_max_output_mb = 1
        self.process.side_effect = too_large
        self.assertEqual((await upload())[0], 413)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(app.state.video_slots.value, 1)

    async def test_send_failure_still_removes_output(self):
        with self.assertRaises(ConnectionError):
            await upload(fail_sending=True)
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_cancellation_cleans_up_files_and_releases_slot(self):
        started = asyncio.Event()

        async def wait(command, settings):
            Path(command[-1]).write_bytes(b"partial")
            started.set()
            await anyio.sleep_forever()

        self.process.side_effect = wait
        task = asyncio.create_task(upload())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(app.state.video_slots.value, 1)


class ProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_child_exit_and_missing_executable(self):
        settings = Settings(_env_file=None)
        await run_ffmpeg([sys.executable, "-c", "pass"], settings)
        with self.assertLogs("app.video", level="ERROR") as logs:
            with self.assertRaises(HTTPException) as error:
                await run_ffmpeg([sys.executable, "-c", "import sys; sys.stderr.write('bad input'); sys.exit(1)"], settings)
        self.assertEqual(error.exception.status_code, 422)
        self.assertIn("bad input", "".join(logs.output))
        with self.assertLogs("app.video", level="ERROR"):
            with self.assertRaises(HTTPException) as error:
                await run_ffmpeg(["asr-fast-nonexistent-ffmpeg-executable"], settings)
        self.assertEqual(error.exception.status_code, 503)

    @unittest.skipIf(os.name == "nt", "POSIX PID liveness assertion")
    async def test_timeout_kills_and_reaps_real_child(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "pid"
            settings = Settings(_env_file=None, ffmpeg_timeout_seconds=0.5)
            script = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
            with self.assertLogs("app.video", level="WARNING"):
                with self.assertRaises(HTTPException) as error:
                    await run_ffmpeg([sys.executable, "-c", script, str(marker)], settings)
            self.assertEqual(error.exception.status_code, 504)
            pid = int(marker.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
