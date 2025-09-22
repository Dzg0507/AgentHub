import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any


def run_command(args: list[str], cwd: Path, timeout_sec: int = 600) -> Dict[str, Any]:
    """Run a command and capture stdout/stderr reliably."""
    try:
        proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            out, err = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            err = (err or "") + f"\nTimeout after {timeout_sec}s"
        return {"returncode": proc.returncode, "stdout": out or "", "stderr": err or ""}
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": str(e)}


def ensure_python_venv(project_dir: Path) -> Dict[str, Any]:
    """Create a per-project Python venv if missing and install requirements.txt if present.
    Returns info dict: {created: bool, python: str, pip_stdout: str, pip_stderr: str}
    """
    info: Dict[str, Any] = {"created": False, "python": sys.executable}
    venv_dir = project_dir / ".venv"
    try:
        if not venv_dir.exists():
            import venv as _venv
            builder = _venv.EnvBuilder(with_pip=True, clear=False)
            builder.create(str(venv_dir))
            info["created"] = True
        # paths
        if os.name == "nt":
            py_exec = str((venv_dir / "Scripts" / "python.exe").resolve())
            pip_exec = str((venv_dir / "Scripts" / "pip.exe").resolve())
        else:
            py_exec = str((venv_dir / "bin" / "python").resolve())
            pip_exec = str((venv_dir / "bin" / "pip").resolve())
        info["python"] = py_exec
        # install reqs
        req = project_dir / "requirements.txt"
        if req.exists():
            proc = subprocess.Popen(
                [pip_exec, "install", "--disable-pip-version-check", "--no-input", "-r", str(req)],
                cwd=str(project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                out, err = proc.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                info["pip_stderr"] = (info.get("pip_stderr", "") + "\nTimeout after 300s").strip()
            info["pip_stdout"] = out or ""
            info["pip_stderr"] = (info.get("pip_stderr", "") + "\n" + (err or "")).strip()
    except Exception as e:
        info["error"] = str(e)
    return info


