import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
from fastapi import HTTPException

from app.config import Settings
from app.main import app
from app.subtitles import Cue, ffprobe_executable, parse_srt, video_duration, write_srt
from app.video import run_ffmpeg
from test_docs import get
from test_video import post_multipart

SRT = "1\n00:00:00,000 --> 00:00:02,500\n你好，世界。\n\n2\n00:00:03,000 --> 00:00:05,000\n第二行字幕\n另一行文字\n".encode()


class SubtitleTextTests(unittest.TestCase):
    def test_utf8_bom_crlf_and_multiple_blank_lines(self):
        data = b"\xef\xbb\xbf" + SRT.replace(b"\n\n", b"\n\n\n").replace(b"\n", b"\r\n")
        cues = parse_srt(data)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0], Cue(0, 2500, "你好，世界。"))
        self.assertEqual(cues[1].text, "第二行字幕\n另一行文字")

    def test_invalid_srt_is_rejected(self):
        for data in (b"not SRT", b"\xff", SRT + b"\x00", b"1\n00:00:05,000 --> 00:00:02,000\ntext", b"1\n00:61:00,000 --> 01:02:00,000\ntext", SRT + b"\n3\n00:00:01,000 --> 00:00:02,000\ntext"):
            with self.subTest(data=data):
                with self.assertRaises(HTTPException) as error:
                    parse_srt(data)
                self.assertEqual(error.exception.status_code, 422)
        with self.assertRaises(HTTPException) as error:
            parse_srt(b"  \n")
        self.assertEqual(error.exception.status_code, 400)

    def test_cues_clipped_to_video_duration_and_all_outside_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.srt"
            write_srt(path, [Cue(0, 1000, "first"), Cue(8000, 12000, "last"), Cue(15000, 17000, "outside")], Decimal(10))
            self.assertEqual(parse_srt(path.read_bytes()), [Cue(0, 1000, "first"), Cue(8000, 10000, "last")])
            with self.assertRaises(HTTPException) as error:
                write_srt(path, [Cue(15000, 17000, "outside")], Decimal(10))
            self.assertEqual(error.exception.status_code, 422)

    def test_ffprobe_path_defaults_to_ffmpeg_sibling_or_path(self):
        settings = Settings(_env_file=None, ffmpeg_path="ffmpeg", ffprobe_path="")
        self.assertEqual(ffprobe_executable(settings), "ffprobe")
        settings.ffmpeg_path = str(Path(tempfile.gettempdir()) / "tools with spaces" / "ffmpeg.exe")
        self.assertEqual(Path(ffprobe_executable(settings)), Path(settings.ffmpeg_path).with_name("ffprobe.exe"))
        settings.ffprobe_path = str(Path(tempfile.gettempdir()) / "custom-probe")
        self.assertEqual(ffprobe_executable(settings), settings.ffprobe_path)


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_video_stream_duration_then_container_fallback(self):
        for info, expected in (
            ({"streams": [{"duration": "10.125"}], "format": {"duration": "11"}}, Decimal("10.125")),
            ({"streams": [{"duration": "N/A"}], "format": {"duration": "11.5"}}, Decimal("11.5")),
        ):
            result = subprocess.CompletedProcess([], 0, json.dumps(info).encode())
            with patch("app.subtitles.anyio.run_process", AsyncMock(return_value=result)):
                self.assertEqual(await video_duration(Path("video.mp4"), Settings(_env_file=None)), expected)

    async def test_missing_video_or_duration_rejected(self):
        for info in ({"streams": []}, {"streams": [{}]}, {"streams": [{"duration": "NaN"}]}, []):
            with patch("app.subtitles.anyio.run_process", AsyncMock(return_value=subprocess.CompletedProcess([], 0, json.dumps(info).encode()))):
                with self.assertRaises(HTTPException) as error:
                    await video_duration(Path("video.mp4"), Settings(_env_file=None))
                self.assertEqual(error.exception.status_code, 422)

    async def test_missing_probe_and_probe_timeout(self):
        with self.assertLogs("app.subtitles", level="ERROR"):
            with self.assertRaises(HTTPException) as error:
                await video_duration(Path("video.mp4"), Settings(_env_file=None, ffprobe_path="sv4agent-nonexistent-ffprobe"))
        self.assertEqual(error.exception.status_code, 503)

        async def wait(*args, **kwargs):
            await anyio.sleep_forever()

        with patch("app.subtitles.anyio.run_process", AsyncMock(side_effect=wait)):
            with self.assertRaises(HTTPException) as error:
                await video_duration(Path("video.mp4"), Settings(_env_file=None, ffprobe_timeout_seconds=0.01))
        self.assertEqual(error.exception.status_code, 504)

    async def test_real_process_resolves_relative_files_in_private_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="subtitle cwd ") as directory:
            root = Path(directory)
            (root / "captions.srt").write_bytes(SRT)
            script = "from pathlib import Path; assert Path('captions.srt').read_bytes(); Path('done').write_text('ok')"
            await run_ffmpeg([sys.executable, "-c", script], Settings(_env_file=None), cwd=root)
            self.assertEqual((root / "done").read_text(), "ok")


class SubtitleRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = Settings(_env_file=None, ffmpeg_temp_dir=self.root, app_api_key="")
        self.patches = [patch("app.subtitles.get_settings", return_value=self.settings), patch("app.auth.get_settings", return_value=self.settings)]
        for item in self.patches:
            item.start()
        self.previous_slots = getattr(app.state, "video_slots", None)
        app.state.video_slots = anyio.Semaphore(1)
        self.commands = []

        async def process(command, settings, *, cwd):
            self.commands.append(command)
            self.assertEqual((cwd / "input.video").read_bytes(), b"video")
            self.assertEqual(parse_srt((cwd / "captions.srt").read_bytes()), parse_srt(SRT))
            Path(command[-1]).write_bytes(b"mp4 data")

        self.process = AsyncMock(side_effect=process)
        self.probe = AsyncMock(return_value=Decimal(10))
        self.patches.extend([patch("app.subtitles.run_ffmpeg", self.process), patch("app.subtitles.video_duration", self.probe)])
        for item in self.patches[2:]:
            item.start()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()
        if self.previous_slots is None:
            del app.state.video_slots
        else:
            app.state.video_slots = self.previous_slots
        self.temporary.cleanup()

    async def upload(self, video=b"video", subtitles=SRT, fields=None, **kwargs):
        return await post_multipart("/v1/video/subtitles", [("file", "video.mp4", video), ("subtitles", "captions.srt", subtitles)], fields, **kwargs)

    async def test_default_soft_mode_copies_video_without_gpu_or_fps_filter(self):
        status, headers, body = await self.upload()
        self.assertEqual(status, 200)
        self.assertEqual(body, b"mp4 data")
        self.assertEqual(headers[b"content-type"], b"video/mp4")
        self.assertIn(b"subtitled.mp4", headers[b"content-disposition"])
        command = self.commands[-1]
        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertEqual(command[command.index("-c:a") + 1], "copy")
        self.assertEqual(command[command.index("-c:s") + 1], "mov_text")
        self.assertEqual(command[command.index("-t") + 1], "10.000000")
        self.assertNotIn("-shortest", command)
        self.assertNotIn("-vf", command)
        self.assertNotIn("-gpu", command)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(app.state.video_slots.value, 1)

    async def test_burn_upsamples_before_drawing_and_uses_nvenc(self):
        status, _, _ = await self.upload(fields={"subtitle_mode": "burn", "codec": "hevc", "audio_codec": "aac", "font_size": 30})
        self.assertEqual(status, 200)
        command = self.commands[-1]
        self.assertIn("hevc_nvenc", command)
        filters = command[command.index("-vf") + 1]
        self.assertTrue(filters.startswith("fps=24,subtitles=filename=captions.srt:"))
        self.assertIn("FontSize=30", filters)
        self.assertNotIn(str(self.root), filters)
        self.assertEqual(command[command.index("-c:a") + 1], "aac")

    async def test_query_mode_selects_encoding_and_overrides_form(self):
        for query_mode, form_mode in (("burn", None), ("soft", None), ("burn", "soft"), ("soft", "burn"), ("burn", "burn")):
            with self.subTest(query_mode=query_mode, form_mode=form_mode):
                fields = {"subtitle_mode": form_mode} if form_mode is not None else None
                status, _, _ = await self.upload(fields=fields, query_string=f"subtitle_mode={query_mode}".encode())
                self.assertEqual(status, 200)
                command = self.commands[-1]
                self.assertEqual(command[command.index("-c:v") + 1], "h264_nvenc" if query_mode == "burn" else "copy")
                self.assertEqual("-vf" in command, query_mode == "burn")
                self.assertEqual("-c:s" in command, query_mode == "soft")

    async def test_invalid_query_mode_rejected_before_processing(self):
        for query_string in (b"subtitle_mode=other", b"subtitle_mode=", b"subtitle_mode=BURN"):
            with self.subTest(query_string=query_string):
                self.assertEqual((await self.upload(query_string=query_string))[0], 422)
        self.process.assert_not_awaited()
        self.probe.assert_not_awaited()
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(app.state.video_slots.value, 1)

    async def test_invalid_options_or_srt_do_not_start_processing(self):
        for fields in ({"subtitle_mode": "other"}, {"fps": 0}, {"font_size": 200}, {"subtitle_language": "zh;cmd"}):
            self.assertEqual((await self.upload(fields=fields))[0], 422)
        self.assertEqual((await self.upload(subtitles=b"invalid"))[0], 422)
        self.assertEqual((await self.upload(subtitles=b""))[0], 400)
        self.process.assert_not_awaited()
        self.probe.assert_not_awaited()

    async def test_upload_limits_and_empty_video(self):
        self.settings.ffmpeg_subtitle_max_upload_kb = 1
        self.assertEqual((await self.upload(subtitles=b"x" * 1025))[0], 413)
        self.settings.ffmpeg_max_upload_mb = 1
        self.assertEqual((await self.upload(video=b"x" * 1024 * 1024))[0], 413)
        self.assertEqual((await self.upload(video=b""))[0], 400)
        self.process.assert_not_awaited()
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(app.state.video_slots.value, 1)

    async def test_auth_disabled_and_shared_queue(self):
        self.settings.app_api_key = type(self.settings.app_api_key)("secret")
        self.assertEqual((await self.upload())[0], 401)
        headers = [(b"authorization", b"Bearer secret")]
        self.settings.ffmpeg_enabled = False
        self.assertEqual((await self.upload(headers=headers))[0], 503)
        self.settings.ffmpeg_enabled = True
        self.settings.ffmpeg_queue_timeout_seconds = 0.01
        await app.state.video_slots.acquire()
        try:
            status, response_headers, _ = await self.upload(headers=headers)
            self.assertEqual(status, 503)
            self.assertEqual(response_headers[b"retry-after"], b"5")
            self.assertEqual(app.state.video_slots.value, 0)
        finally:
            app.state.video_slots.release()
        self.process.assert_not_awaited()

    async def test_probe_and_ffmpeg_errors_clean_up(self):
        self.probe.side_effect = HTTPException(422, "bad video")
        self.assertEqual((await self.upload())[0], 422)
        self.process.assert_not_awaited()
        self.probe.side_effect = None
        self.process.side_effect = HTTPException(422, "missing libass")
        self.assertEqual((await self.upload(fields={"subtitle_mode": "burn"}))[0], 422)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(app.state.video_slots.value, 1)

    async def test_timeout_cancellation_output_cap_and_send_failure_clean_up(self):
        with self.assertRaises(ConnectionError):
            await self.upload(fail_sending=True)
        started = asyncio.Event()

        async def wait(command, settings, *, cwd):
            Path(command[-1]).write_bytes(b"partial")
            started.set()
            await anyio.sleep_forever()

        self.process.side_effect = wait
        self.settings.ffmpeg_subtitle_timeout_seconds = 0.01
        status, _, body = await self.upload()
        self.assertEqual(status, 504)
        self.assertIn(b"FFMPEG_SUBTITLE_TIMEOUT_SECONDS", body)
        self.settings.ffmpeg_subtitle_timeout_seconds = 60
        started.clear()
        task = asyncio.create_task(self.upload())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        async def too_large(command, settings, *, cwd):
            with Path(command[-1]).open("wb") as target:
                target.truncate(1024 * 1024)

        self.process.side_effect = too_large
        self.settings.ffmpeg_max_output_mb = 1
        self.assertEqual((await self.upload())[0], 413)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(app.state.video_slots.value, 1)

    async def test_docs_show_new_endpoint_and_soft_default(self):
        status, _, body = await get("/openapi.json")
        self.assertEqual(status, 200)
        schema = json.loads(body)
        operation = schema["paths"]["/v1/video/subtitles"]["post"]
        self.assertIn("video/mp4", operation["responses"]["200"]["content"])
        query_mode = next(p for p in operation["parameters"] if p["name"] == "subtitle_mode" and p["in"] == "query")
        self.assertFalse(query_mode.get("required", False))
        self.assertIn({"enum": ["soft", "burn"], "type": "string"}, query_mode["schema"]["anyOf"])
        ref = operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
        model = schema["components"]["schemas"][ref.rsplit("/", 1)[1]]
        self.assertEqual(model["properties"]["subtitle_mode"]["default"], "soft")
        self.assertEqual(set(model["required"]), {"file", "subtitles"})


if __name__ == "__main__":
    unittest.main()
