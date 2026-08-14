"""
Load verifier criteria + ground-truth note from a criteria file.

Criteria files live in criteria/<benchmark>.md. Expected layout:

    # <title>

    ## Ground Truth Note

    <one paragraph the verifier always sees>

    ## Criteria

    ### <Criterion Name>

    <criterion description>

Each `### Criterion Name` heading is the display name; its id (the
score-cache key) is slugged from the name, or pinned explicitly with
`### Criterion Name {#id}`.
"""

import os
import re

# Optional trailing `{#id}` anchor on a criterion heading (Markdown-style).
_CRIT_ID = re.compile(r"^(.*?)\s*\{#([A-Za-z0-9_-]+)\}\s*$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

_FORMAT_HINT = """\
Expected criteria-file layout (see criteria/TEMPLATE.md):

    # <title>

    ## Ground Truth Note        <- optional

    <one paragraph the verifier always sees>

    ## Criteria

    ### <Criterion Name>        <- id is slugged from the name;
                                   pin one with `### Name {#id}`
    <instruction for this criterion>
"""


def _criteria_dirs():
    """Directories searched for a bare benchmark name, in order: the
    ``criteria/`` folder at the repository root (next to the ``llm_verifier``
    package), then ``criteria/`` under the current working directory."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(os.path.dirname(pkg_dir), "criteria"),
        os.path.join(os.getcwd(), "criteria"),
    ]


def _read_criteria(path):
    """Resolve a criteria argument — a filesystem path or a bare benchmark
    name mapping to ``criteria/<name>.md`` — to its file contents."""
    if os.path.isfile(path):
        with open(path) as f:
            return f.read()
    name = os.path.basename(path)
    if not name.endswith(".md"):
        name += ".md"
    searched = []
    for d in _criteria_dirs():
        candidate = os.path.join(d, name)
        searched.append(candidate)
        if os.path.isfile(candidate):
            with open(candidate) as f:
                return f.read()
    raise FileNotFoundError(
        f"criteria {path!r} not found: not a file on disk, and no prompt "
        f"named {name!r} in " + " or ".join(searched)
    )


def load_prompts(path):
    """Return (ground_truth_note, criteria) where criteria is a list of
    {"id", "name", "description"} dicts in file order. HTML comments are
    stripped, so criteria files can carry author notes the verifier never
    sees. Raises ``ValueError`` if no criteria can be parsed."""
    text = _HTML_COMMENT.sub("", _read_criteria(path))
    lines = text.splitlines()

    ground_truth_note = ""
    criteria = []
    seen = set()        # ids already used in this file

    section = None      # "ground_truth" | "criteria" | None
    cur = None          # current criterion dict
    buf = []            # accumulated body lines

    def flush():
        nonlocal ground_truth_note, cur, buf
        text = "\n".join(buf).strip()
        if section == "ground_truth" and not ground_truth_note:
            ground_truth_note = text
        elif cur is not None:
            cur["description"] = text
            criteria.append(cur)
            cur = None
        buf = []

    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            flush()
            heading = line[3:].strip().lower()
            if "ground truth" in heading:
                section = "ground_truth"
            elif "criteri" in heading:
                section = "criteria"
            else:
                section = None
        elif line.startswith("### ") and section == "criteria":
            flush()
            heading = line[4:].strip()
            m = _CRIT_ID.match(heading)
            if m:                       # `### Name {#explicit_id}`
                name, cid = m.group(1).strip(), m.group(2).strip()
            else:                       # `### Name` -> id slugged from the name
                name, cid = heading, _slug(heading)
            cur = {"id": _dedup_id(cid, seen), "name": name}
        elif line.startswith("# "):
            continue
        else:
            buf.append(line)
    flush()

    if not criteria:
        raise ValueError(
            f"no criteria found in {path!r} — check the `## Criteria` section "
            f"and its `### Criterion Name` headings.\n\n{_FORMAT_HINT}")
    empty = [c["id"] for c in criteria if not c.get("description")]
    if empty:
        raise ValueError(
            f"criteria in {path!r} have empty instructions: {empty}. "
            f"Each `### Criterion Name` heading needs a body the verifier can "
            f"score with.")
    return ground_truth_note, criteria


def select_criteria(criteria, ids):
    """Subset + order `criteria` by the given list of ids. If `ids` is falsy,
    return all criteria in file order."""
    if not ids:
        return list(criteria)
    by_id = {c["id"]: c for c in criteria}
    missing = [cid for cid in ids if cid not in by_id]
    if missing:
        raise KeyError(f"criteria not found in prompt file: {missing}")
    return [by_id[cid] for cid in ids]


def _slug(text):
    """Derive a criterion id from free text: lowercase, alnum + underscores,
    truncated to 40 chars."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:40].rstrip("_") or "criterion"


def _dedup_id(cid, seen):
    """Make `cid` unique against `seen` by appending _2, _3, ..."""
    out, n = cid, 1
    while out in seen:
        n += 1
        out = f"{cid}_{n}"
    seen.add(out)
    return out


def normalize_criteria(criteria):
    """Normalize a flexible `criteria` argument into the canonical list of
    ``{"id", "name", "description"}`` dicts.

    Accepts:
      - a dict mapping name -> description
        (``{"Root cause": "Did the agent fix the real cause?"}``)
      - a list of strings, each used as both name and description
      - a list of dicts with at least ``description``; missing ``id`` /
        ``name`` are derived (id is slugged from the name)

    Bundled benchmark names and criteria-file paths are strings and are
    handled by :func:`load_prompts`, not here.
    """
    if isinstance(criteria, dict):
        criteria = [{"name": name, "description": desc}
                    for name, desc in criteria.items()]

    out, seen = [], set()
    for i, raw in enumerate(criteria):
        if isinstance(raw, str):
            cid, name, desc = "", raw, raw
        elif isinstance(raw, dict):
            cid = str(raw.get("id") or "")
            name = str(raw.get("name") or "")
            desc = str(raw.get("description") or "")
        else:
            raise TypeError(
                f"criteria[{i}] must be a str or dict, got {type(raw).__name__}")
        if not desc:
            raise ValueError(f"criteria[{i}] is missing a 'description'")
        name = name or cid or _slug(desc)
        cid = _dedup_id(cid or _slug(name), seen)
        out.append({"id": cid, "name": name, "description": desc})
    if not out:
        raise ValueError("criteria is empty")
    return out


def _main(argv):
    """Preview a criteria file exactly as the verifier will see it:

        python -m llm_verifier my_criteria.md
    """
    if len(argv) != 1:
        print("usage: python -m llm_verifier <criteria.md | benchmark name>")
        return 2
    note, criteria = load_prompts(argv[0])
    print(f"ground-truth note: {note or '(none)'}\n")
    print(f"{len(criteria)} criteria:")
    for c in criteria:
        print(f"\n### {c['name']}  (id: {c['id']})\n{c['description']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
