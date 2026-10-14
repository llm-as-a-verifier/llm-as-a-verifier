#!/usr/bin/env python3
"""Task-name extraction must survive either path separator.

`load_terminal` derives the task id from the directory path that `glob.glob`
returns for `<agent_dir>/*/`. glob keeps the trailing separator, and that
separator is "\\" on Windows — so stripping only "/" leaves it in place and
`basename()` returns the empty string. Every task then lands under the same
key and the loader silently yields a single task.

The assertions below drive `posixpath` and `ntpath` directly, so this file
tests both platforms from either one. Zero dependencies; the repo has no test
runner, so it is also runnable on its own:

    python tests/test_loaders_paths.py
"""

import ntpath
import posixpath
import sys


def extract(task_dir, pathmod):
    """The expression under test, parameterised over the path flavour.

    Mirrors llm_verifier/loaders.py::load_terminal.
    """
    return pathmod.basename(pathmod.normpath(task_dir))


CASES = [
    # (label, path as glob returns it, path module, expected task name)
    ("posix", "data/tb_trajs/agent/chess-best-move/", posixpath, "chess-best-move"),
    ("posix no trailing sep", "data/tb_trajs/agent/chess-best-move", posixpath, "chess-best-move"),
    ("windows", "data\\tb_trajs\\agent\\chess-best-move\\", ntpath, "chess-best-move"),
    ("windows forward slashes", "data/tb_trajs/agent/chess-best-move/", ntpath, "chess-best-move"),
    ("windows mixed", "data/tb_trajs\\agent\\chess-best-move\\", ntpath, "chess-best-move"),
    ("windows drive-absolute", "C:\\work\\agent\\chess-best-move\\", ntpath, "chess-best-move"),
]


def test_task_name_survives_either_separator():
    for label, task_dir, pathmod, expected in CASES:
        got = extract(task_dir, pathmod)
        assert got == expected, f"{label}: expected {expected!r}, got {got!r}"


def test_distinct_directories_stay_distinct():
    """The regression itself: distinct task dirs must not share a key.

    With the old `basename(task_dir.rstrip("/"))` every Windows path mapped to
    "" and the dict collapsed to one entry.
    """
    dirs = [f"data\\tb\\agent\\task-{i}\\" for i in range(89)]
    names = {extract(d, ntpath) for d in dirs}
    assert len(names) == len(dirs), f"expected {len(dirs)} distinct names, got {len(names)}"
    assert "" not in names, "a directory produced an empty task name"


if __name__ == "__main__":
    failures = 0
    for fn in (test_task_name_survives_either_separator,
               test_distinct_directories_stay_distinct):
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    sys.exit(1 if failures else 0)
