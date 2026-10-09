"""Regression tests for resolving a bare criteria name from an installed
package, where neither the repository root nor the working directory has a
``criteria/`` folder."""

import os
import subprocess
import sys
import zipfile

import pytest

from llm_verifier import prompts

CRITERIA_FILE = """\
# Bundled

## Ground Truth Note

The note.

## Criteria

### Only Criterion

The instruction.
"""

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _installed_layout(tmp_path):
    """A site-packages-shaped tree: the package directory carries its own
    ``criteria/``, and nothing sits beside the package."""
    pkg_dir = tmp_path / "site-packages" / "llm_verifier"
    (pkg_dir / "criteria").mkdir(parents=True)
    (pkg_dir / "criteria" / "bundled.md").write_text(CRITERIA_FILE)
    return pkg_dir


def test_bare_name_resolves_from_the_installed_package(tmp_path, monkeypatch):
    pkg_dir = _installed_layout(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.setattr(prompts, "__file__", str(pkg_dir / "prompts.py"))
    monkeypatch.chdir(elsewhere)

    note, criteria = prompts.load_prompts("bundled")

    assert note == "The note."
    assert [c["id"] for c in criteria] == ["only_criterion"]
    assert criteria[0]["description"] == "The instruction."


def test_missing_name_still_reports_every_searched_directory(tmp_path,
                                                             monkeypatch):
    pkg_dir = _installed_layout(tmp_path)
    monkeypatch.setattr(prompts, "__file__", str(pkg_dir / "prompts.py"))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        prompts.load_prompts("absent")

    message = str(excinfo.value)
    assert str(pkg_dir / "criteria" / "absent.md") in message
    assert os.path.join(str(tmp_path), "criteria", "absent.md") in message


@pytest.mark.skipif(
    subprocess.call([sys.executable, "-c", "import build"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL) != 0,
    reason="requires the `build` package",
)
def test_wheel_ships_every_criteria_file(tmp_path):
    """The lookup above only helps if the files are actually in the wheel."""
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path),
         REPO_ROOT],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    shipped = {os.path.basename(name)
               for name in zipfile.ZipFile(wheels[0]).namelist()
               if name.startswith("llm_verifier/criteria/")}

    source = os.path.join(REPO_ROOT, "criteria")
    expected = {name for name in os.listdir(source) if name.endswith(".md")}
    assert expected, "no criteria files in the repository to ship"
    assert shipped == expected
