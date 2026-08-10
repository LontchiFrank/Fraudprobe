"""REVIEW.md item 6: MANIFEST.json recorded a git commit hash but no dirty
flag, so a manifest could point at a commit that does not contain the code
that actually produced the run's numbers, with no way to detect that from
the manifest alone."""

from __future__ import annotations

import subprocess

from fraudprobe import manifest as manifest_module
from fraudprobe.manifest import write_manifest


def _minimal_result():
    return {"version": "0.1.0", "mode": "single", "seed": 1, "data_source": "synthetic:100rows"}


class _FakeCompleted:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def test_manifest_reports_clean_tree(tmp_path, monkeypatch):
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "status"]:
            return _FakeCompleted("")  # no output => clean
        if cmd[:2] == ["git", "rev-parse"]:
            return _FakeCompleted("deadbeef\n")
        return real_run(cmd, **kwargs)  # e.g. `uname -p` from platform.processor()

    monkeypatch.setattr(subprocess, "run", fake_run)
    manifest = write_manifest(_minimal_result(), tmp_path, 1.0)
    assert manifest["git_dirty"] is False
    assert manifest["git_diff_stat"] is None


def test_manifest_reports_dirty_tree_with_diff_stat(tmp_path, monkeypatch):
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "status"]:
            return _FakeCompleted(" M fraudprobe/pipeline.py\n")
        if cmd[:2] == ["git", "rev-parse"]:
            return _FakeCompleted("deadbeef\n")
        if cmd[:2] == ["git", "diff"]:
            return _FakeCompleted(" fraudprobe/pipeline.py | 4 ++--\n 1 file changed\n")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    manifest = write_manifest(_minimal_result(), tmp_path, 1.0)
    assert manifest["git_dirty"] is True
    assert "pipeline.py" in manifest["git_diff_stat"]


def test_manifest_git_dirty_none_when_git_unavailable(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    manifest = write_manifest(_minimal_result(), tmp_path, 1.0)
    assert manifest["git_dirty"] is None
    assert manifest["git_diff_stat"] is None


def test_manifest_against_real_repo_matches_actual_git_status(tmp_path):
    # Sanity check against the real repo state (not mocked) — git_dirty should
    # agree with `git status --porcelain` run directly.
    manifest = write_manifest(_minimal_result(), tmp_path, 1.0)
    real_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True,
        cwd=manifest_module.__file__.rsplit("/", 1)[0],
    ).stdout.strip())
    assert manifest["git_dirty"] == real_dirty
