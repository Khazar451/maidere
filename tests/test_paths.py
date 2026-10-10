"""Paths are anchored at the checkout or the data dir, never at the working directory."""

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from maidere import __version__
from maidere.api.app import setup_logging
from maidere.core.config import settings
from maidere.paths import (
    agents_dir,
    package_dir,
    repo_root,
    resolve_config_path,
    resolve_db_path,
    runtime_root,
    skills_dir,
    static_dir,
)


class TestPaths(unittest.TestCase):
    def test_package_layout(self):
        self.assertTrue((package_dir() / "api" / "app.py").is_file())
        self.assertTrue((static_dir() / "index.html").is_file())
        self.assertTrue((static_dir() / "app.js").is_file())
        self.assertTrue((skills_dir() / "deep-research" / "SKILL.md").is_file())
        self.assertTrue((agents_dir() / "researcher.md").is_file())
        self.assertIsNotNone(repo_root())
        self.assertTrue(__version__)

    def test_relative_paths_ignore_cwd(self):
        anchor = runtime_root()
        original = os.getcwd()
        try:
            os.chdir("/")
            resolved = resolve_config_path("db/maidere.db")
            self.assertEqual(resolved, (anchor / "db" / "maidere.db").resolve())
            self.assertNotEqual(resolved, Path("/db/maidere.db"))
        finally:
            os.chdir(original)

    def test_absolute_paths_pass_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "custom.db"
            self.assertEqual(resolve_config_path(str(target)), target)
            self.assertEqual(resolve_db_path(str(target)), str(target))
            self.assertTrue(target.parent.is_dir())

    def test_memory_database_is_unchanged(self):
        self.assertEqual(resolve_db_path(":memory:"), ":memory:")

    def test_runtime_root_override(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"MAIDERE_RUNTIME_ROOT": tmp}):
            self.assertEqual(runtime_root(), Path(tmp).resolve())
            self.assertEqual(resolve_config_path("logs/x.jsonl"), Path(tmp).resolve() / "logs" / "x.jsonl")


class TestFileLogging(unittest.TestCase):
    def test_setup_logging_writes_json_and_rotates_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "nested" / "maidere.jsonl"
            with patch.object(settings, "log_path", str(log_file)), patch.object(settings, "log_level", "INFO"):
                setup_logging()
                logging.getLogger("maidere.test").info("phase3-log-probe", extra={})
                # structlog bound logger goes through the same handlers.
                import structlog

                structlog.get_logger("maidere.test").info("phase3-structlog-probe", marker="ok")
            self.assertTrue(log_file.is_file(), "log file was not created")
            parsed = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(parsed)
            self.assertTrue(any(item.get("marker") == "ok" for item in parsed), parsed[-1])
            # Console handler exists alongside the file handler.
            self.assertGreaterEqual(len(logging.getLogger().handlers), 2)
