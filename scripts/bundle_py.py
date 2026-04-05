"""Collect all project .py files except main.py into a manifest and optional concatenated bundle."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

SKIP_DIR_NAMES = frozenset({
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "tests",
    "scripts",
})
SKIP_MAIN_NAMES = frozenset({"main.py"})
# Omitted from the bundle (e.g. large or redundant when inlining other modules).
SKIP_BUNDLE_NAMES = frozenset({"utils.py"})


def _local_module_stems(root: Path) -> frozenset[str]:
    """Stems of project ``*.py`` files (for stripping cross-imports in ``write_bundle``)."""
    stems: set[str] = set()
    root = root.resolve()
    for p in root.rglob("*.py"):
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        if p.name in SKIP_MAIN_NAMES:
            continue
        stems.add(p.stem)
    return frozenset(stems)


def _string_names_from_ast_listlike(value: ast.AST) -> list[str]:
    if not isinstance(value, (ast.List, ast.Tuple)):
        return []
    out: list[str] = []
    for elt in value.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            out.append(elt.value)
    return out


def remove_module_level_all_and_collect_names(source: str) -> tuple[str, list[str]]:
    """
    Remove a top-level ``__all__ = [...]`` assignment; return new source and the names.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name) or t.id != "__all__":
            continue
        end_ln = node.end_lineno
        if end_ln is None:
            return source, []
        start = node.lineno - 1
        names = _string_names_from_ast_listlike(node.value)
        new_src = "".join(lines[:start] + lines[end_ln:])
        return new_src, names
    return source, []


def merge_all_public_names(in_order: list[list[str]]) -> list[str]:
    """Concatenate name lists in order, dropping duplicates (first wins)."""
    seen: set[str] = set()
    out: list[str] = []
    for group in in_order:
        for n in group:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def format_merged_all_footer(names: list[str]) -> str:
    if not names:
        return ""
    parts = [f"    {n!r},\n" for n in names]
    return "__all__ = [\n" + "".join(parts) + "]\n"


def strip_local_project_imports(source: str, local_stems: frozenset[str]) -> str:
    """Drop ``import`` / ``from`` lines that reference other project modules (flat bundle)."""
    new_lines: list[str] = []
    for line in source.splitlines(keepends=True):
        s = line.strip()
        if s.startswith("from "):
            m = re.match(r"^from\s+\.*(\w+)\s+import\s+", s)
            if m and m.group(1) in local_stems:
                continue
        if s.startswith("import "):
            m = re.match(r"^import\s+([\w\s,]+?)(?:\s+#.*)?$", s)
            if m:
                parts = [x.strip() for x in m.group(1).split(",") if x.strip()]
                bases = [p.split()[0] for p in parts]
                if bases and all(b in local_stems for b in bases):
                    continue
        new_lines.append(line)
    return "".join(new_lines)


def iter_py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    root = root.resolve()
    for p in root.rglob("*.py"):
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        if p.name in SKIP_MAIN_NAMES:
            continue
        if p.name in SKIP_BUNDLE_NAMES:
            continue
        out.append(p)
    return sorted(out, key=lambda x: str(x).lower())


def write_manifest(paths: list[Path], root: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(p.relative_to(root)).replace("\\", "/") for p in paths]
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_bundle(paths: list[Path], root: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    local_stems = _local_module_stems(root)
    chunks: list[str] = []
    all_groups: list[list[str]] = []
    for p in paths:
        rel = p.relative_to(root).as_posix()
        chunks.append(f"# === {rel} ===\n")
        raw = p.read_text(encoding="utf-8")
        body, names = remove_module_level_all_and_collect_names(raw)
        all_groups.append(names)
        body = strip_local_project_imports(body, local_stems)
        chunks.append(body)
        if not chunks[-1].endswith("\n"):
            chunks.append("\n")
        chunks.append("\n")
    merged = merge_all_public_names(all_groups)
    footer = format_merged_all_footer(merged)
    if footer:
        chunks.append("# === __all__ (merged from bundled modules) ===\n")
        chunks.append(footer)
    dest.write_text("".join(chunks), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="List / bundle .py sources (excludes main.py, utils.py, tests/, scripts/)."
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (default: repo root).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output",
        help="Output directory (default: output/).",
    )
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="Print paths to stdout only; do not write files.",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    paths = iter_py_files(root)

    if args.list_only:
        for p in paths:
            print(p.relative_to(root).as_posix())
        print(
            f"# {len(paths)} file(s) (excluded: main.py, utils.py, tests/, scripts/)",
            file=sys.stderr,
        )
        return

    out_dir = args.out_dir.resolve()
    manifest = out_dir / "py_bundle_manifest.txt"
    bundle = out_dir / "py_bundle.txt"
    write_manifest(paths, root, manifest)
    write_bundle(paths, root, bundle)
    print(f"Wrote {manifest.relative_to(root)} ({len(paths)} paths)")
    print(f"Wrote {bundle.relative_to(root)}")


if __name__ == "__main__":
    main()
