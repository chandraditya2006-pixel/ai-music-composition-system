"""
test_retention.py
=================
Automated test suite for MIDI file retention management.
Uses isolated temporary directories to guarantee that existing project
files and existing generated MIDI files are never modified or deleted.
"""

import os
import time
import tempfile
import unittest
from unittest.mock import patch
import threading

from app import (
    app,
    cleanup_generated_files,
    acquire_active_download,
    release_active_download,
    get_active_downloads,
    is_download_active
)


class TestRetentionManagement(unittest.TestCase):

    def setUp(self):
        # Create an isolated temporary directory for every test
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = self.test_dir.name

    def tearDown(self):
        # Clean up the isolated temporary directory
        self.test_dir.cleanup()

    def _create_mock_file(self, filename, content=b"MThd_mock_midi", mtime_offset=0):
        """Helper to create a file with a specific modification timestamp."""
        filepath = os.path.join(self.dir_path, filename)
        with open(filepath, "wb") as f:
            f.write(content)
        target_mtime = time.time() + mtime_offset
        os.utime(filepath, (target_mtime, target_mtime))
        return filepath

    # -----------------------------------------------------------------
    # Test 1: Files under the retention limit are preserved
    # -----------------------------------------------------------------
    def test_files_under_retention_limit_preserved(self):
        """When total files <= max_files, no files should be deleted."""
        for i in range(5):
            self._create_mock_file(f"track_{i}.mid", mtime_offset=i * 10)

        deleted = cleanup_generated_files(directory=self.dir_path, max_files=10)
        self.assertEqual(deleted, 0)

        remaining = [f for f in os.listdir(self.dir_path) if f.endswith(".mid")]
        self.assertEqual(len(remaining), 5)

    # -----------------------------------------------------------------
    # Test 2: Files over the limit are trimmed to configured maximum
    # -----------------------------------------------------------------
    def test_files_over_limit_trimmed_to_max(self):
        """When total files > max_files, excess oldest files must be deleted."""
        for i in range(10):
            self._create_mock_file(f"song_{i:02d}.mid", mtime_offset=i * 5)

        deleted = cleanup_generated_files(directory=self.dir_path, max_files=4)
        self.assertEqual(deleted, 6)

        remaining = [f for f in os.listdir(self.dir_path) if f.endswith(".mid")]
        self.assertEqual(len(remaining), 4)

    # -----------------------------------------------------------------
    # Test 3: The newest files are strictly retained
    # -----------------------------------------------------------------
    def test_newest_files_retained(self):
        """Cleanup must delete the oldest files and preserve the most recent."""
        # Create files from oldest (t=0) to newest (t=40)
        files = []
        for i in range(5):
            files.append(f"version_{i}.mid")
            self._create_mock_file(f"version_{i}.mid", mtime_offset=i * 10)

        # Retain top 2 newest
        deleted = cleanup_generated_files(directory=self.dir_path, max_files=2)
        self.assertEqual(deleted, 3)

        remaining = set(os.listdir(self.dir_path))
        # Expect version_3.mid and version_4.mid to survive
        self.assertIn("version_3.mid", remaining)
        self.assertIn("version_4.mid", remaining)
        # Oldest should be gone
        self.assertNotIn("version_0.mid", remaining)
        self.assertNotIn("version_1.mid", remaining)
        self.assertNotIn("version_2.mid", remaining)

    # -----------------------------------------------------------------
    # Test 4: Excluded/active files are protected from deletion
    # -----------------------------------------------------------------
    def test_excluded_file_never_deleted(self):
        """Active/newly generated file specified in exclude_filenames is protected."""
        self._create_mock_file("old_active.mid", mtime_offset=-100)
        self._create_mock_file("other_1.mid", mtime_offset=10)
        self._create_mock_file("other_2.mid", mtime_offset=20)

        # Retain only 1 file, but protect the oldest
        deleted = cleanup_generated_files(
            directory=self.dir_path,
            max_files=1,
            exclude_filenames={"old_active.mid"}
        )
        remaining = set(os.listdir(self.dir_path))
        self.assertIn("old_active.mid", remaining)
        self.assertIn("other_2.mid", remaining)
        self.assertNotIn("other_1.mid", remaining)

    # -----------------------------------------------------------------
    # Test 5: Non-MIDI files are completely untouched
    # -----------------------------------------------------------------
    def test_non_midi_files_untouched(self):
        """Files like .txt, .json, .py, or subdirectories must never be touched."""
        # Create non-MIDI files
        self._create_mock_file("metadata.json", content=b"{}", mtime_offset=-500)
        self._create_mock_file("readme.txt", content=b"docs", mtime_offset=-400)
        self._create_mock_file("model.keras", content=b"weights", mtime_offset=-300)

        # Create subfolder
        subdir = os.path.join(self.dir_path, "subfolder.mid")
        os.makedirs(subdir, exist_ok=True)

        # Create 5 MIDI files
        for i in range(5):
            self._create_mock_file(f"tune_{i}.mid", mtime_offset=i * 10)

        # Retain only 2 MIDI files
        cleanup_generated_files(directory=self.dir_path, max_files=2)

        # Assert all non-MIDI files and directories still exist
        self.assertTrue(os.path.isfile(os.path.join(self.dir_path, "metadata.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.dir_path, "readme.txt")))
        self.assertTrue(os.path.isfile(os.path.join(self.dir_path, "model.keras")))
        self.assertTrue(os.path.isdir(subdir))

        # Check MIDI files pruned correctly to 2
        midi_files = [f for f in os.listdir(self.dir_path) if f.endswith(".mid") and os.path.isfile(os.path.join(self.dir_path, f))]
        self.assertEqual(len(midi_files), 2)

    # -----------------------------------------------------------------
    # Test 6: Symlinks are not followed or deleted
    # -----------------------------------------------------------------
    def test_symlinks_are_not_followed(self):
        """Symlinks pointing to outside files must be ignored."""
        outside_file = os.path.join(self.dir_path, "..", "external_target.mid")
        try:
            with open(outside_file, "wb") as f:
                f.write(b"external")

            symlink_path = os.path.join(self.dir_path, "symlink_file.mid")
            try:
                os.symlink(outside_file, symlink_path)
            except (OSError, NotImplementedError):
                # On Windows without developer mode/admin privilege, symlinks might not be permitted
                self.skipTest("Symlinks not supported in current environment.")

            # Create 3 regular files
            for i in range(3):
                self._create_mock_file(f"regular_{i}.mid", mtime_offset=i * 10)

            # Max files = 1
            cleanup_generated_files(directory=self.dir_path, max_files=1)

            # Symlink must still exist and outside file must exist
            self.assertTrue(os.path.exists(symlink_path))
            self.assertTrue(os.path.exists(outside_file))

        finally:
            if os.path.exists(outside_file):
                try:
                    os.remove(outside_file)
                except OSError:
                    pass

    # -----------------------------------------------------------------
    # Test 7: Cleanup error does not fail generation request
    # -----------------------------------------------------------------
    def test_cleanup_failure_does_not_fail_generation(self):
        """If cleanup raises an unexpected OSError, the API must still return 201."""
        client = app.test_client()

        with patch("app.cleanup_generated_files", side_effect=OSError("Disk access error")):
            response = client.post("/api/generate", json={"num_notes": 10, "temperature": 0.8})
            self.assertEqual(response.status_code, 201)
            data = response.get_json()
            self.assertEqual(data.get("status"), "success")
            self.assertTrue(data.get("filename", "").endswith(".mid"))

    # -----------------------------------------------------------------
    # Test 8: Health endpoint exposes configured max_generated_files
    # -----------------------------------------------------------------
    def test_health_endpoint_exposes_retention_limit(self):
        """GET /api/health should include max_generated_files setting."""
        client = app.test_client()
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("max_generated_files", data)
        self.assertEqual(data["max_generated_files"], 100)

    # -----------------------------------------------------------------
    # Test 9: Active in-flight download is shielded from cleanup
    # -----------------------------------------------------------------
    def test_active_download_protected_from_cleanup(self):
        """A file actively marked as being downloaded must never be deleted by cleanup."""
        # Create 5 files with oldest being actively served
        self._create_mock_file("old_downloading.mid", mtime_offset=-200)
        self._create_mock_file("file_1.mid", mtime_offset=10)
        self._create_mock_file("file_2.mid", mtime_offset=20)
        self._create_mock_file("file_3.mid", mtime_offset=30)

        # Mark oldest file as active in-flight download
        acquire_active_download("old_downloading.mid")
        try:
            self.assertTrue(is_download_active("old_downloading.mid"))

            # Clean up with max_files=2 (would delete oldest 2 if unprotected)
            deleted = cleanup_generated_files(directory=self.dir_path, max_files=2)

            remaining = set(os.listdir(self.dir_path))
            # old_downloading.mid MUST be preserved because it is active
            self.assertIn("old_downloading.mid", remaining)
            # file_3.mid is the newest, so it is preserved
            self.assertIn("file_3.mid", remaining)
            # Unprotected older files should be pruned
            self.assertNotIn("file_1.mid", remaining)
        finally:
            release_active_download("old_downloading.mid")

        self.assertFalse(is_download_active("old_downloading.mid"))

    # -----------------------------------------------------------------
    # Test 10: Concurrent downloads reference counting
    # -----------------------------------------------------------------
    def test_concurrent_downloads_refcounting(self):
        """Multiple concurrent downloads on the same file increment and decrement safely."""
        target = "shared_stream.mid"
        self._create_mock_file(target, mtime_offset=-500)
        self._create_mock_file("newer_1.mid", mtime_offset=10)
        self._create_mock_file("newer_2.mid", mtime_offset=20)

        # First reader begins
        acquire_active_download(target)
        # Second reader begins
        acquire_active_download(target)
        self.assertTrue(is_download_active(target))

        # First reader finishes, second still reading
        release_active_download(target)
        self.assertTrue(is_download_active(target), "File should remain active while second reader is running")

        # Cleanup runs while second reader is running
        cleanup_generated_files(directory=self.dir_path, max_files=1)
        self.assertTrue(os.path.isfile(os.path.join(self.dir_path, target)))

        # Second reader finishes
        release_active_download(target)
        self.assertFalse(is_download_active(target))

    # -----------------------------------------------------------------
    # Test 11: Download route registers and unregisters via call_on_close
    # -----------------------------------------------------------------
    def test_download_route_lifecycle_refcount(self):
        """Verifies that GET /api/download tracks active downloads during streaming and cleans up after close."""
        client = app.test_client()

        # Temporarily mock GENERATED_DIR to point to self.dir_path
        with patch("app.GENERATED_DIR", self.dir_path):
            test_file = "stream_test.mid"
            self._create_mock_file(test_file, content=b"MThd_stream_content")

            response = client.get(f"/api/download/{test_file}", buffered=False)
            self.assertEqual(response.status_code, 200)

            # While response is open (simulating ongoing streaming), file is active
            self.assertTrue(is_download_active(test_file))

            # Simulate client closing stream
            response.close()

            # Reference count must now be 0
            self.assertFalse(is_download_active(test_file))

    # -----------------------------------------------------------------
    # Test 12: Concurrent downloads and cleanup without deadlocks
    # -----------------------------------------------------------------
    def test_concurrent_downloads_and_cleanup_no_deadlock(self):
        """Stress-tests multi-threaded concurrent download acquisitions and cleanup runs."""
        for i in range(20):
            self._create_mock_file(f"stress_{i}.mid", mtime_offset=i)

        errors = []

        def reader_task(filename):
            try:
                for _ in range(25):
                    acquire_active_download(filename)
                    time.sleep(0.001)
                    release_active_download(filename)
            except Exception as e:
                errors.append(e)

        def cleanup_task():
            try:
                for _ in range(10):
                    cleanup_generated_files(directory=self.dir_path, max_files=10)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=reader_task, args=(f"stress_{i}.mid",))
            threads.append(t)
        for _ in range(3):
            t = threading.Thread(target=cleanup_task)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            self.assertFalse(t.is_alive(), "Thread deadlock detected!")

        self.assertEqual(len(errors), 0, f"Errors occurred during concurrency test: {errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
