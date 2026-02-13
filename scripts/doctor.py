"""Environment inspector to help debug cross-OS issues."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from importlib.util import find_spec


def _try_import(name: str):
    try:
        module = __import__(name)
    except Exception as exc:  # pragma: no cover - diagnostic only
        return None, exc
    return module, None


def _check_torch():
    torch, err = _try_import("torch")
    if err:
        return "missing", str(err)
    cuda_available = getattr(torch, "cuda", None) and torch.cuda.is_available()
    return "ok", f"version={torch.__version__}, cuda={cuda_available}"


def _check_qiskit():
    qiskit, err = _try_import("qiskit")
    if err:
        return "missing", str(err)
    aer, aer_err = _try_import("qiskit_aer")
    detail = f"qiskit={qiskit.__version__}"
    if aer and not aer_err:
        detail += f", qiskit-aer={aer.__version__}"
    else:
        detail += ", qiskit-aer missing"
    return "ok", detail


def _check_ruff():
    if find_spec("ruff"):
        completed = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True
        )
        if completed.returncode == 0:
            return "ok", completed.stdout.strip()
        return "error", completed.stderr.strip()
    return "missing", "ruff not importable"


def main() -> int:
    print("[doctor] platform:", platform.platform())
    print("[doctor] python:", sys.executable)
    print("[doctor] version:", sys.version.split()[0])
    print("[doctor] venv:", sys.prefix)
    print("[doctor] nvidia-smi:", shutil.which("nvidia-smi") or "not found")

    status, detail = _check_qiskit()
    print(f"[doctor] qiskit: {status} ({detail})")

    status, detail = _check_torch()
    print(f"[doctor] torch: {status} ({detail})")

    status, detail = _check_ruff()
    print(f"[doctor] ruff: {status} ({detail})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
