import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.model_download import download_model


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.model_dir = self.root / "downloaded"
        self.model_dir.mkdir()
        for name, content in {
            "config.json": b"{}", "model.bin": b"mock weights",
            "tokenizer.json": b"{}", "vocabulary.json": b"[]",
        }.items():
            (self.model_dir / name).write_bytes(content)
        self.snapshot_download = Mock(return_value=str(self.model_dir))
        modelscope = types.ModuleType("modelscope")
        modelscope.snapshot_download = self.snapshot_download
        self.hf_download = Mock(side_effect=AssertionError("Hugging Face must not be used"))
        hf = types.ModuleType("huggingface_hub")
        hf.snapshot_download = self.hf_download
        self.modules = patch.dict(sys.modules, {"modelscope": modelscope, "huggingface_hub": hf})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()
        self.temporary.cleanup()

    def download(self, **overrides):
        options = dict(
            model="large-v3", endpoint="https://modelscope.cn/",
            cache_dir="models/modelscope", local_files_only=False, project_root=self.root,
        )
        options.update(overrides)
        return download_model(**options)

    def test_large_v3_alias_uses_requested_systran_modelscope_repository(self):
        self.assertEqual(self.download(), str(self.model_dir))
        kwargs = self.snapshot_download.call_args.kwargs
        self.assertEqual(kwargs["endpoint"], "https://modelscope.cn")
        self.assertEqual(kwargs["model_id"], "Systran/faster-whisper-large-v3")
        self.assertEqual(kwargs["revision"], "master")
        self.assertEqual(kwargs["cache_dir"], str(self.root / "models" / "modelscope"))
        self.hf_download.assert_not_called()

    def test_explicit_systran_repository_is_preserved(self):
        self.download(model="Systran/faster-whisper-large-v3")
        self.assertEqual(self.snapshot_download.call_args.kwargs["model_id"], "Systran/faster-whisper-large-v3")

    def test_custom_repository_and_authentication_are_forwarded(self):
        self.download(model="team/custom-ct2", token="test-only-token", revision="v1", local_files_only=True)
        kwargs = self.snapshot_download.call_args.kwargs
        self.assertEqual(kwargs["model_id"], "team/custom-ct2")
        self.assertEqual(kwargs["token"], "test-only-token")
        self.assertEqual(kwargs["revision"], "v1")
        self.assertTrue(kwargs["local_files_only"])

    def test_turbo_alias_uses_the_correct_repository(self):
        self.download(model="turbo")
        self.assertEqual(
            self.snapshot_download.call_args.kwargs["model_id"],
            "pengzhendong/faster-whisper-large-v3-turbo",
        )

    def test_no_hf_fallback_when_modelscope_download_fails(self):
        self.snapshot_download.side_effect = RuntimeError("ModelScope unavailable")
        with self.assertRaisesRegex(RuntimeError, "ModelScope unavailable"):
            self.download()
        self.assertEqual(self.snapshot_download.call_count, 1)
        self.hf_download.assert_not_called()

    def test_invalid_endpoint_or_model_fails_before_network_call(self):
        for overrides in ({"endpoint": "modelscope.cn"}, {"model": "unknown-model"}, {"model": "https://modelscope.cn/a/b"}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.download(**overrides)
        self.snapshot_download.assert_not_called()

    def test_existing_local_directory_does_not_use_download_sdk(self):
        result = self.download(model=str(self.model_dir), local_files_only=True)
        self.assertEqual(result, str(self.model_dir))
        self.snapshot_download.assert_not_called()

    def test_relative_local_path_is_resolved_against_project_root(self):
        result = self.download(model="./downloaded", local_files_only=True)
        self.assertEqual(result, str(self.model_dir))
        self.snapshot_download.assert_not_called()

    def test_missing_explicit_local_path_does_not_become_a_repository(self):
        with self.assertRaisesRegex(FileNotFoundError, "Local model directory does not exist"):
            self.download(model="./models/missing")
        self.snapshot_download.assert_not_called()

    def test_missing_tokenizer_is_rejected_before_inference(self):
        (self.model_dir / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing: tokenizer.json"):
            self.download(model=str(self.model_dir))
        self.snapshot_download.assert_not_called()

    def test_incomplete_download_is_rejected(self):
        (self.model_dir / "config.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing: config.json"):
            self.download()
        self.hf_download.assert_not_called()

    def test_git_lfs_pointer_is_not_accepted_as_weights(self):
        (self.model_dir / "model.bin").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            self.download(model=str(self.model_dir))

    def test_empty_weights_are_not_accepted(self):
        (self.model_dir / "model.bin").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty required file"):
            self.download(model=str(self.model_dir))

    def test_text_vocabulary_is_supported(self):
        (self.model_dir / "vocabulary.json").unlink()
        (self.model_dir / "vocabulary.txt").write_text("test\n")
        self.assertEqual(self.download(), str(self.model_dir))

    def test_missing_vocabulary_is_rejected(self):
        (self.model_dir / "vocabulary.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing vocabulary"):
            self.download(model=str(self.model_dir))

    def test_empty_vocabulary_is_rejected(self):
        (self.model_dir / "vocabulary.json").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty required file"):
            self.download(model=str(self.model_dir))

    def test_token_is_not_logged(self):
        with self.assertLogs("app.model_download", level="INFO") as captured:
            self.download(token="test-only-token")
        self.assertNotIn("test-only-token", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
