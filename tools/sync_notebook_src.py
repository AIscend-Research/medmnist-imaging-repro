#!/usr/bin/env python3
"""Keep the notebook's embedded ``src/`` package in sync with ``src/`` on disk.

``notebooks/replication/medmnist_replication.ipynb`` carries the whole ``src/``
package base64-encoded in one cell so the notebook runs on Kaggle with no clone.
That copy is a duplicate, and duplicates drift. This tool is the only sanctioned
way to update it.

    python tools/sync_notebook_src.py --check    # CI/pre-commit: fail on drift
    python tools/sync_notebook_src.py            # rewrite the cell from src/

Dependency-free (stdlib only) so it runs anywhere, including without torch.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO, "src")
NOTEBOOK = os.path.join(REPO, "notebooks", "replication", "medmnist_replication.ipynb")

# The cell is identified by this marker, not by index, so inserting cells above
# it does not silently break the sync.
MARKER = "SRC_B64"
DICT_RE = re.compile(r"SRC_B64 = \{.*?\n\}", re.S)


def src_files():
    """Every ``src/*.py``, as ``{basename: text}``."""
    return {
        name: open(os.path.join(SRC_DIR, name), encoding="utf-8").read()
        for name in sorted(os.listdir(SRC_DIR))
        if name.endswith(".py")
    }


def render_dict(files):
    lines = ["SRC_B64 = {"]
    for name, text in files.items():
        blob = base64.b64encode(text.encode("utf-8")).decode("ascii")
        lines.append(f"    {name!r}: {blob!r},")
    lines.append("}")
    return "\n".join(lines)


def load_notebook():
    with open(NOTEBOOK, encoding="utf-8") as f:
        return json.load(f)


def find_cell(nb):
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and MARKER in "".join(cell["source"]):
            return i
    raise SystemExit(f"no cell containing {MARKER!r} found in {NOTEBOOK}")


def embedded_files(nb):
    """Decode the notebook's copy back into ``{basename: text}``."""
    source = "".join(nb["cells"][find_cell(nb)]["source"])
    match = DICT_RE.search(source)
    if match is None:
        raise SystemExit("could not locate the SRC_B64 dict literal in the cell")
    namespace: dict = {}
    exec(match.group(0), namespace)  # noqa: S102 - our own generated literal
    return {k: base64.b64decode(v).decode("utf-8") for k, v in namespace[MARKER].items()}


def diff(disk, embedded):
    """Return (missing, extra, changed) basenames."""
    missing = sorted(set(disk) - set(embedded))   # on disk, not in notebook
    extra = sorted(set(embedded) - set(disk))     # in notebook, deleted from disk
    changed = sorted(k for k in set(disk) & set(embedded) if disk[k] != embedded[k])
    return missing, extra, changed


def check():
    disk = src_files()
    embedded = embedded_files(load_notebook())
    missing, extra, changed = diff(disk, embedded)
    for name in missing:
        print(f"  MISSING from notebook : {name}")
    for name in extra:
        print(f"  STALE in notebook     : {name} (no longer in src/)")
    for name in changed:
        print(f"  DRIFTED               : {name} "
              f"(disk {len(disk[name])}b vs notebook {len(embedded[name])}b)")
    if missing or extra or changed:
        print(f"\n{len(missing) + len(extra) + len(changed)} file(s) out of sync. "
              f"Run: python tools/sync_notebook_src.py")
        return 1
    print(f"in sync: {len(disk)} files, {sum(len(v) for v in disk.values()):,} bytes")
    return 0


def write():
    disk = src_files()
    nb = load_notebook()
    idx = find_cell(nb)
    source = "".join(nb["cells"][idx]["source"])
    if not DICT_RE.search(source):
        raise SystemExit("could not locate the SRC_B64 dict literal in the cell")

    # Replace only the dict literal; the loader code below it is preserved.
    updated = DICT_RE.sub(lambda _: render_dict(disk), source, count=1)
    nb["cells"][idx]["source"] = updated.splitlines(keepends=True)
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["execution_count"] = None

    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"embedded {len(disk)} files into {os.path.relpath(NOTEBOOK, REPO)}")
    return check()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify only; exit 1 on drift (does not modify the notebook)")
    args = ap.parse_args(argv)
    return check() if args.check else write()


if __name__ == "__main__":
    sys.exit(main())
