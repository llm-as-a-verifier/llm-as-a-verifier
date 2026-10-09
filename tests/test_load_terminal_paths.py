"""Regression test exercises ``load_terminal`` itself on Windows paths."""

import json
import ntpath
import os
from types import SimpleNamespace
from unittest.mock import patch

from llm_verifier import loaders


def test_load_terminal_preserves_windows_task_names(tmp_path):
    trajectory = tmp_path / "trial_trajectory.json"
    trajectory.write_text(json.dumps({
        "trial_name": "trial",
        "reward": 1,
        "trajectory": {"steps": []},
    }))
    task_dir = "C:\\work\\agent\\chess-best-move\\"

    def glob(pattern):
        if pattern.endswith("*/"):
            return [task_dir]
        return [str(trajectory)]

    fake_path = SimpleNamespace(
        isabs=os.path.isabs,
        join=os.path.join,
        basename=ntpath.basename,
        normpath=ntpath.normpath,
    )
    with patch.object(loaders.globmod, "glob", side_effect=glob), \
            patch.object(loaders.os, "path", fake_path):
        tasks, runs = loaders.load_terminal({"agent_dir": "ignored"}, str(tmp_path))

    assert runs == 1
    assert list(tasks) == ["chess-best-move"]
    assert tasks["chess-best-move"][0]["trial_name"] == "trial"
