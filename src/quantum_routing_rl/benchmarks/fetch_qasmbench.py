"""Best-effort QASMBench fetcher with git or zip fallback."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen


def _has_qasm(dest: Path) -> bool:
    return any(dest.rglob("*.qasm"))


def _clone(repo: str, dest: Path) -> bool:
    git = shutil.which("git")
    if not git:
        return False
    try:
        result = subprocess.run(
            [git, "clone", "--depth", "1", repo, str(dest)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return False
    return result.returncode == 0


def _download_zip(repo: str, dest: Path) -> bool:
    # GitHub style: https://github.com/org/repo -> .../archive/refs/heads/main.zip
    zip_urls = [
        f"{repo}/archive/refs/heads/main.zip",
        f"{repo}/archive/refs/heads/master.zip",
    ]
    for url in zip_urls:
        try:
            with urlopen(url) as resp:
                data = resp.read()
        except Exception:
            continue
        with tempfile.TemporaryDirectory() as td:
            tmp_zip = Path(td) / "qasmbench.zip"
            tmp_zip.write_bytes(data)
            with zipfile.ZipFile(tmp_zip) as zf:
                members = zf.namelist()
                prefix = members[0].split("/")[0]
                zf.extractall(td)
                extracted = Path(td) / prefix
                dest.mkdir(parents=True, exist_ok=True)
                for item in extracted.iterdir():
                    target = dest / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)
            return True
    return False


def ensure_qasmbench(dest: Path, repo: str = "https://github.com/pnnl/QASMBench") -> Path | None:
    """Fetch QASMBench into ``dest`` if missing. Returns path when qasm present."""

    dest = dest.expanduser()
    if dest.exists() and _has_qasm(dest):
        print(f"[fetch_qasmbench] found existing qasm under {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.parent / f"{dest.name}_tmp"
    if tmp_dest.exists():
        shutil.rmtree(tmp_dest, ignore_errors=True)

    print(f"[fetch_qasmbench] attempting git clone to {tmp_dest}")
    if _clone(repo, tmp_dest) and _has_qasm(tmp_dest):
        dest.mkdir(parents=True, exist_ok=True)
        for item in tmp_dest.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        shutil.rmtree(tmp_dest, ignore_errors=True)
        print(f"[fetch_qasmbench] clone successful -> {dest}")
        return dest

    shutil.rmtree(tmp_dest, ignore_errors=True)

    print("[fetch_qasmbench] falling back to zip download")
    if _download_zip(repo, tmp_dest) and _has_qasm(tmp_dest):
        dest.mkdir(parents=True, exist_ok=True)
        for item in tmp_dest.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        shutil.rmtree(tmp_dest, ignore_errors=True)
        print(f"[fetch_qasmbench] downloaded and extracted into {dest}")
        return dest

    print("[fetch_qasmbench] failed to fetch QASMBench; please download manually.")
    return None
