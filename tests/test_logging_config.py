import io
import logging
import unittest
from contextlib import redirect_stderr

from app.logging_config import configure_logging


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("app")
        self.previous = (self.logger.handlers[:], self.logger.level, self.logger.propagate)
        self.logger.handlers = []

    def tearDown(self):
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers, self.logger.level, self.logger.propagate = self.previous

    def test_startup_info_is_visible_without_root_logger_configuration(self):
        output = io.StringIO()
        with redirect_stderr(output):
            configure_logging("info")
            logging.getLogger("app.startup_test").info("Loading model")
        self.assertIn("INFO [app.startup_test] Loading model", output.getvalue())

    def test_reconfiguration_does_not_duplicate_logs(self):
        output = io.StringIO()
        with redirect_stderr(output):
            configure_logging("info")
            configure_logging("debug")
            logging.getLogger("app.startup_test").debug("Model ready")
        self.assertEqual(output.getvalue().count("Model ready"), 1)


if __name__ == "__main__":
    unittest.main()
