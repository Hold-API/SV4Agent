import sys
import types
import unittest
from unittest.mock import Mock, patch

from app.model_download import download_model


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.snapshot_download = Mock(return_value="cached-model")
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = self.snapshot_download
        # Simulate a Hub library that was imported using a different endpoint.
        hub.constants = types.SimpleNamespace(ENDPOINT="https://huggingface.co")
        self.modules = patch.dict(sys.modules, {"huggingface_hub": hub})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def download(self, **overrides):
        options = dict(
            model="large-v3", endpoint="https://hf-mirror.com/",
            cache_dir="models", local_files_only=False,
        )
        options.update(overrides)
        return download_model(**options)

    def test_explicit_endpoint_overrides_import_time_default(self):
        self.assertEqual(self.download(), "cached-model")
        kwargs = self.snapshot_download.call_args.kwargs
        self.assertEqual(kwargs["endpoint"], "https://hf-mirror.com")
        self.assertEqual(kwargs["repo_id"], "Systran/faster-whisper-large-v3")
        self.assertEqual(kwargs["cache_dir"], "models")

    def test_custom_repository_and_authentication_are_forwarded(self):
        self.download(model="team/custom-ct2", token="test-only-token", local_files_only=True)
        kwargs = self.snapshot_download.call_args.kwargs
        self.assertEqual(kwargs["repo_id"], "team/custom-ct2")
        self.assertEqual(kwargs["token"], "test-only-token")
        self.assertTrue(kwargs["local_files_only"])

    def test_turbo_alias_uses_the_correct_repository(self):
        self.download(model="turbo")
        self.assertEqual(
            self.snapshot_download.call_args.kwargs["repo_id"],
            "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        )

    def test_no_fallback_when_mirror_download_fails(self):
        self.snapshot_download.side_effect = RuntimeError("mirror unavailable")
        with self.assertRaisesRegex(RuntimeError, "mirror unavailable"):
            self.download()
        self.assertEqual(self.snapshot_download.call_count, 1)

    def test_invalid_endpoint_or_model_fails_before_network_call(self):
        for overrides in ({"endpoint": "hf-mirror.com"}, {"model": "unknown-model"}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.download(**overrides)
        self.snapshot_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
