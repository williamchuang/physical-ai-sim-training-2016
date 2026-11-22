#!/usr/bin/env python3
"""Rewrites the repository tree in README.md from the files git tracks.

The tree sits between the markers <!-- tree:start --> and <!-- tree:end -->.
Each folder is annotated with the tagline of its own README.md: the text after
" — " in that README's first heading.

    python3 tools/update_readme_tree.py          # rewrite README.md
    python3 tools/update_readme_tree.py --check  # exit 1 if README.md is out of date

The pre-commit hook (tools/install-git-hooks.sh) runs it before every commit,
and the GitHub workflow runs --check on every push.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START, END = "<!-- tree:start -->", "<!-- tree:end -->"
NAME = "physical-ai-sim-training-2016"


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if (ROOT / f).exists()]


# Folders that hold no README of their own (a README under .github/ would replace
# the repository's main README on GitHub).
FALLBACK = {".github": "GitHub settings", ".github/workflows": "Continuous checks: README tree, core build and tests"}


def tagline(folder: Path) -> str:
    rel = folder.relative_to(ROOT).as_posix()
    if rel in FALLBACK:
        return FALLBACK[rel]
    readme = folder / "README.md"
    if not readme.exists():
        return ""
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            parts = line[2:].split(" — ", 1)
            return parts[1].strip() if len(parts) == 2 else ""
    return ""


def build_tree(files: list[str]) -> str:
    tree: dict = {}
    for f in files:
        node = tree
        for part in f.split("/"):
            node = node.setdefault(part, {})

    rows: list[tuple[str, str]] = [(f"{NAME}/", "")]

    def walk(node: dict, prefix: str, path: Path) -> None:
        # folders first, then files, each alphabetically (case-insensitive)
        items = sorted(node.items(), key=lambda kv: (not kv[1], kv[0].lower()))
        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            branch = "└── " if last else "├── "
            if child:
                rows.append((prefix + branch + name + "/", tagline(path / name)))
                walk(child, prefix + ("    " if last else "│   "), path / name)
            else:
                rows.append((prefix + branch + name, ""))

    walk(tree, "", ROOT)
    width = max(len(r[0]) for r in rows if r[1]) + 2
    lines = [(left.ljust(width) + note).rstrip() for left, note in rows]
    return "```text\n" + "\n".join(lines) + "\n```"


def render(text: str, tree: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        sys.exit(f"README.md has no {START} ... {END} block")
    return pattern.sub(lambda _: f"{START}\n{tree}\n{END}", text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report whether README.md is current")
    args = ap.parse_args()
    old = README.read_text(encoding="utf-8")
    new = render(old, build_tree(tracked_files()))
    if args.check:
        if new != old:
            print("README.md: the repository tree is out of date; run python3 tools/update_readme_tree.py")
            sys.exit(1)
        print("README.md: the repository tree is current")
        return
    if new != old:
        README.write_text(new, encoding="utf-8")
        print("README.md: repository tree updated")


if __name__ == "__main__":
    main()
