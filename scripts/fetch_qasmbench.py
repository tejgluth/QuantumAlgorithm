"""Fetch QASMBench from GitHub into artifacts/benchmarks/qasmbench_src."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="https://github.com/pnnl/QASMBench",
        help="Repository URL to clone.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("artifacts/benchmarks/qasmbench_src"),
        help="Destination directory for the clone.",
    )
    return parser.parse_args(argv)


def _network_available(repo: str) -> bool:
    try:
        subprocess.run(
            ["git", "ls-remote", repo],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dest = args.dest.expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[fetch_qasmbench] Destination already exists, leaving as-is: {dest}")
        return 0

    if not _network_available(args.repo):
        print(
            "[fetch_qasmbench] Network unavailable or git not configured; "
            "skipping clone. Provide QASMBENCH_ROOT manually."
        )
        return 0

    print(f"[fetch_qasmbench] Cloning {args.repo} into {dest} ...")
    result = subprocess.run(["git", "clone", "--depth", "1", args.repo, str(dest)], check=False)
    if result.returncode != 0:
        print(
            f"[fetch_qasmbench] Clone failed with code {result.returncode}; "
            "proceeding without local copy."
        )
        return 0

    print("[fetch_qasmbench] Clone complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
