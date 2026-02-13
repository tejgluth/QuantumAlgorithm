"""Cross-platform project bootstrap: create venv and install dependencies."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_VENV_DIR = ".venv"
VENV_ENV_VAR = "QRRL_VENV_DIR"


class BootstrapError(Exception):
    """Custom error for bootstrap failures."""


def _log(msg: str) -> None:
    print(f"[bootstrap] {msg}")


def _run(
    cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    _log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check, env=env)


def _venv_python(venv_dir: Path) -> Path:
    candidates = [
        venv_dir / "bin" / "python3",
        venv_dir / "bin" / "python",
        venv_dir / "Scripts" / "python.exe",
        venv_dir / "Scripts" / "python",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise BootstrapError(f"Unable to find python inside venv at {venv_dir}")


def _ensure_venv(venv_dir: Path) -> Path:
    if not venv_dir.exists():
        _log(f"creating venv at {venv_dir}")
        _run([sys.executable, "-m", "venv", str(venv_dir)])
    else:
        _log(f"using existing venv at {venv_dir}")
    return _venv_python(venv_dir)


def _pip_install(py: Path, packages: list[str]) -> None:
    if not packages:
        return
    _run([str(py), "-m", "pip", "install", "-U", *packages])


def _torch_cuda_available(py: Path) -> bool:
    try:
        out = subprocess.check_output(
            [str(py), "-c", "import torch; print(torch.cuda.is_available())"]
        )
    except subprocess.CalledProcessError:
        return False
    return out.strip().decode().lower() == "true"


def _try_install_torch_cuda(py: Path) -> None:
    """Best-effort CUDA torch install for Linux."""
    if platform.system() != "Linux":
        _log("--cuda requested but non-Linux OS detected; skipping CUDA install")
        return
    if shutil.which("nvidia-smi") is None:
        _log("nvidia-smi not found; skipping CUDA torch install")
        return

    _log("attempting torch install (default index)")
    try:
        _pip_install(py, ["torch"])
        if _torch_cuda_available(py):
            _log("CUDA is available after default torch install")
            return
    except subprocess.CalledProcessError:
        _log("default torch install failed; continuing")

    for cu_tag in ("cu121", "cu118"):
        try:
            _log(f"attempting torch {cu_tag} wheels")
            _run(
                [
                    str(py),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "torch",
                    "--index-url",
                    f"https://download.pytorch.org/whl/{cu_tag}",
                ]
            )
            if _torch_cuda_available(py):
                _log(f"CUDA available after installing torch from {cu_tag} index")
                return
        except subprocess.CalledProcessError:
            _log(f"torch install from {cu_tag} index failed; trying next option")

    _log("CUDA-capable torch not detected; continuing with CPU torch")


def _try_install_aer_gpu(py: Path) -> None:
    try:
        _log("attempting qiskit-aer-gpu install")
        _pip_install(py, ["qiskit-aer-gpu"])
    except subprocess.CalledProcessError:
        _log("qiskit-aer-gpu install failed; continuing without GPU Aer")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev", action="store_true", help="Install dev tools (ruff, pytest, pytest-cov)"
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Best-effort CUDA torch install (Linux + NVIDIA only)"
    )
    parser.add_argument(
        "--aer-gpu", action="store_true", help="Attempt qiskit-aer GPU wheels if CUDA detected"
    )
    parser.add_argument(
        "--venv",
        dest="venv_dir",
        type=Path,
        default=None,
        help="Override venv directory (defaults to QRRL_VENV_DIR or .venv)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    venv_dir = args.venv_dir or Path(os.environ.get(VENV_ENV_VAR, DEFAULT_VENV_DIR))
    py = _ensure_venv(venv_dir)

    _pip_install(py, ["pip", "setuptools", "wheel"])

    base_packages = [
        "qiskit",
        "qiskit-aer",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "networkx",
    ]
    _pip_install(py, base_packages)

    if args.dev:
        dev_packages = ["ruff", "pytest", "pytest-cov"]
        _pip_install(py, dev_packages)

    # Install project editable
    _run([str(py), "-m", "pip", "install", "-e", "."])

    if args.cuda:
        _try_install_torch_cuda(py)

    if args.aer_gpu and _torch_cuda_available(py):
        _try_install_aer_gpu(py)
    elif args.aer_gpu:
        _log("--aer-gpu requested but CUDA torch not available; skipping")

    torch_cuda = _torch_cuda_available(py)
    _log(f"torch.cuda.is_available() -> {torch_cuda}")

    _log(f"venv python: {py}")
    _log("Next steps:")
    _log(f"  {py} scripts/run.py test")
    _log(f"  {py} scripts/run.py gauntlet-small")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
