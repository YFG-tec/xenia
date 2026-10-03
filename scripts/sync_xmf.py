"""Sync the built kernel + Python sources into the live `import xmf` tree.

`xmf._native` loads `xmf_core.dll` / `libxmf_core.so` from the directory
that contains `_native.py`. On this machine that is usually the installed
copy under site-packages, not `build/` and not the git worktree. After a
C (or Python package) edit, pytest is lying until this script has run.

Usage (repo root, same interpreter that will run pytest):

    python scripts/sync_xmf.py           # cmake --build, then copy
    python scripts/sync_xmf.py --check   # print paths only
    python scripts/sync_xmf.py --no-build
    python scripts/sync_xmf.py --pip     # fallback: pip install .

Do not invent a third copy command. This file is the standard step.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PKG = ROOT / "src" / "xmf"
# Filter by platform: both trees share the build/ directory, so the
# artifact scan must only match the current OS's kernel library (a stale
# Windows DLL must never win the mtime race on WSL, or vice versa).
if sys.platform == "win32":
    LIB_NAMES = ("xmf_core.dll",)
else:
    LIB_NAMES = ("libxmf_core.so", "xmf_core.so")


def _mtime(path: Path) -> str:
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def _installed_pkg() -> Path:
    try:
        import xmf  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "import xmf failed. Install once with `pip install .` "
            "before using this script."
        ) from exc
    dest = Path(xmf.__file__).resolve().parent
    if dest.name != "xmf":
        raise SystemExit(f"unexpected import location: {dest}")
    return dest


def _newest_artifact() -> Path:
    found: list[Path] = []
    build = ROOT / "build"
    if build.is_dir():
        for name in LIB_NAMES:
            found.extend(build.rglob(name))
    if not found:
        raise SystemExit(
            "no kernel artifact under build/. Run `pip install .` once "
            "to create the CMake tree, or use --pip."
        )
    return max(found, key=lambda p: p.stat().st_mtime)


def _cmake_build(artifact: Path) -> None:
    build_dir = artifact.parent
    print(f"BUILD  {build_dir}")
    subprocess.run(
        ["cmake", "--build", str(build_dir)],
        check=True,
    )


def _copy_tree(artifact: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    lib_dest = dest / artifact.name
    shutil.copy2(artifact, lib_dest)
    print(f"LIB    {artifact} -> {lib_dest}")
    print(f"       mtime {_mtime(lib_dest)}")
    if dest.resolve() == SRC_PKG.resolve():
        print("PY     skipped (import xmf is the worktree)")
        return
    copied = 0
    for src in SRC_PKG.glob("*.py"):
        shutil.copy2(src, dest / src.name)
        copied += 1
    print(f"PY     {copied} files -> {dest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="print paths only")
    ap.add_argument("--no-build", action="store_true", help="copy without cmake")
    ap.add_argument(
        "--pip",
        action="store_true",
        help="full reinstall instead of copy (slow, correct)",
    )
    args = ap.parse_args()

    dest = _installed_pkg()
    print(f"DEST   {dest}")
    print(f"SRC_PY {SRC_PKG}")

    if args.pip:
        if args.check:
            print("MODE   pip install . (check: not run)")
            return 0
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "."],
            cwd=ROOT,
            check=True,
        )
        dest = _installed_pkg()
        print(f"DEST   {dest} (after pip)")
        print("SYNCED")
        return 0

    artifact = _newest_artifact()
    print(f"ART    {artifact}  mtime {_mtime(artifact)}")
    if args.check:
        live = dest / artifact.name
        print(f"LIVE   {live}  exists={live.is_file()}", end="")
        if live.is_file():
            print(f"  mtime {_mtime(live)}")
        else:
            print()
        return 0

    if not args.no_build:
        _cmake_build(artifact)
        artifact = _newest_artifact()

    _copy_tree(artifact, dest)
    print("SYNCED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
