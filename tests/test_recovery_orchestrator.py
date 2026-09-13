import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from train.recovery_orchestrator import process_queue


class RecoveryOrchestratorTests(unittest.TestCase):
    def _queue(self, directory: str, status: str = "pending") -> Path:
        path = Path(directory) / "queue.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "status",
                    "command",
                    "working_dir",
                    "session_name",
                    "resume_override",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "id": "first",
                    "status": status,
                    "command": "python -m train.train_encoder_seq",
                    "working_dir": directory,
                    "session_name": "train-first",
                    "resume_override": "checkpoint_load_path=auto",
                }
            )
        return path

    def test_pending_job_is_started_and_marked_running(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._queue(directory)
            with (
                mock.patch("train.recovery_orchestrator._session_exists", return_value=False),
                mock.patch("train.recovery_orchestrator._start_session") as start,
            ):
                self.assertEqual(process_queue(path), "first")

            start.assert_called_once()
            self.assertFalse(start.call_args.kwargs["resume"])
            with path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.DictReader(handle))["status"], "running")

    def test_running_job_without_session_is_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._queue(directory, status="running")
            with (
                mock.patch("train.recovery_orchestrator._session_exists", return_value=False),
                mock.patch("train.recovery_orchestrator._start_session") as start,
            ):
                process_queue(path)

            self.assertTrue(start.call_args.kwargs["resume"])

    def test_completed_job_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._queue(directory, status="completed")
            with mock.patch("train.recovery_orchestrator._start_session") as start:
                self.assertIsNone(process_queue(path))
            start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
