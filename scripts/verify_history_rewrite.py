"""Check that a history rewrite removed only the intended paths.

Compares an untouched mirror of the repository (the backup) with the
rewritten one, branch by branch and tag by tag:

1. Tip trees: every file in the old tip that is missing from the new tip
   must match a removal pattern, and every other file must have the same
   content (blob id) in both.
2. Code history: for each branch, the number of commits that touch a code
   file (.cu, .cuh, .h, .hpp, .cpp, .py, .ipynb) must be unchanged, and so
   must the sequence of code-file blob ids along the first-parent chain.

Usage:
    python scripts/verify_history_rewrite.py OLD_MIRROR NEW_MIRROR \
        --glob '*.avi' --glob '*.h5' \
        --prefix CuSuperHelium/CuSuperHelium/temp/ --prefix CuSuperHelium/Python/figures/

Exits with status 1 if any check fails.
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys

CODE_EXTENSIONS = (".cu", ".cuh", ".h", ".hpp", ".cpp", ".py", ".ipynb")


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def refs(repo: str) -> dict[str, str]:
    out = git(repo, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads", "refs/tags")
    return dict(line.split(" ", 1) for line in out.splitlines() if line)


def tree(repo: str, rev: str) -> dict[str, str]:
    """path -> blob id for every file in the tree of rev."""
    out = git(repo, "ls-tree", "-r", "-z", "--full-tree", rev)
    files = {}
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        files[path] = meta.split()[2]
    return files


def is_removed(path: str, globs: list[str], prefixes: list[str]) -> bool:
    name = path.rsplit("/", 1)[-1]
    return any(path.startswith(p) for p in prefixes) or any(
        fnmatch.fnmatch(name, g) or fnmatch.fnmatch(path, g) for g in globs)


def code_history(repo: str, ref: str) -> tuple[int, list[str]]:
    """Commits touching code files, and the code-file blob ids they introduce (oldest first)."""
    out = git(repo, "log", "--reverse", "--format=%x00%H", "--raw", "--no-abbrev", "--no-renames", ref,
              "--", *[f"*{e}" for e in CODE_EXTENSIONS])
    commits = 0
    blobs = []
    for chunk in out.split("\0"):
        lines = [l for l in chunk.splitlines()[1:] if l.startswith(":")]
        if not lines:
            continue
        commits += 1
        for l in lines:
            meta, path = l.split("\t", 1)
            blobs.append(f"{path}:{meta.split()[3]}")
    return commits, blobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--glob", action="append", default=[], help="removed file-name glob, e.g. '*.h5'")
    ap.add_argument("--prefix", action="append", default=[], help="removed directory prefix, with trailing /")
    args = ap.parse_args()

    old_refs, new_refs = refs(args.old), refs(args.new)
    failures = []
    if set(old_refs) != set(new_refs):
        failures.append(f"ref sets differ: only old {sorted(set(old_refs) - set(new_refs))}, "
                        f"only new {sorted(set(new_refs) - set(old_refs))}")

    for ref in sorted(set(old_refs) & set(new_refs)):
        old_tree, new_tree = tree(args.old, ref), tree(args.new, ref)
        missing = [p for p in old_tree if p not in new_tree]
        bad_missing = [p for p in missing if not is_removed(p, args.glob, args.prefix)]
        added = [p for p in new_tree if p not in old_tree]
        changed = [p for p in new_tree if p in old_tree and new_tree[p] != old_tree[p]]
        kept_removed = [p for p in new_tree if is_removed(p, args.glob, args.prefix)]

        old_code, old_blobs = code_history(args.old, ref)
        new_code, new_blobs = code_history(args.new, ref)

        problems = []
        if bad_missing:
            problems.append(f"lost files {bad_missing[:5]}")
        if added or changed:
            problems.append(f"added {added[:5]} changed {changed[:5]}")
        if kept_removed:
            problems.append(f"removal pattern still present {kept_removed[:5]}")
        if old_code != new_code:
            problems.append(f"code-touching commits {old_code} -> {new_code}")
        if old_blobs != new_blobs:
            problems.append("code-file blob sequence differs")

        status = "FAIL" if problems else "ok"
        print(f"{status:4} {ref}: {len(new_tree)} files, {len(missing)} removed, "
              f"{new_code} code commits" + (f" -- {'; '.join(problems)}" if problems else ""))
        if problems:
            failures.append(ref)

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print(f"\nAll {len(old_refs)} refs passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
