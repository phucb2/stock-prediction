"""Remove generated artifacts: output/, Submit.ipynb, __pycache__, *.pyc."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_NAMES = frozenset({".git", ".venv", "venv"})


def _under_skip(path: Path) -> bool:
    return any(part in SKIP_NAMES for part in path.parts)


def main() -> None:
    removed: list[str] = []

    out = ROOT / "output"
    if out.is_dir():
        shutil.rmtree(out)
        removed.append(str(out.relative_to(ROOT)))

    submit = ROOT / "Submit.ipynb"
    if submit.is_file():
        submit.unlink()
        removed.append(submit.name)

    for d in sorted(ROOT.rglob("__pycache__"), key=lambda p: len(p.parts), reverse=True):
        if _under_skip(d):
            continue
        shutil.rmtree(d, ignore_errors=True)
        removed.append(str(d.relative_to(ROOT)))

    for p in ROOT.rglob("*.pyc"):
        if _under_skip(p):
            continue
        p.unlink(missing_ok=True)
        removed.append(str(p.relative_to(ROOT)))

    if removed:
        print("Removed:")
        for r in removed:
            print(f"  {r}")
    else:
        print("Nothing to clean.")


if __name__ == "__main__":
    main()
