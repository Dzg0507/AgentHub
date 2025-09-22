from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import sys
import re


def run_python_snippet_in_dir(
    snippet: str, cwd: Path, timeout_sec: int = 20
) -> Dict[str, Any]:
    cwd.mkdir(parents=True, exist_ok=True)
    temp_file = cwd / "_snippet_exec.py"
    temp_file.write_text(snippet, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, temp_file.name],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return {
            "returncode": -1,
            "stdout": stdout,
            "stderr": f"Timeout after {timeout_sec}s\n" + (stderr or ""),
        }
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except Exception:
            pass
    return {"returncode": proc.returncode, "stdout": stdout, "stderr": stderr}


_BANNED_IMPORT_TOKENS = [
    "pygame",
    "flask",
    "fastapi",
    "requests",
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "tensorflow",
    "sqlalchemy",
    "psycopg2",
    "matplotlib",
    "seaborn",
    "opencv",
    "cv2",
    "PIL",
    "pillow",
    "django",
    "boto3",
    "sklearn",
    "airflow",
    "pydantic",
]


def extract_python_code_block(text: str) -> Optional[str]:
    """Extract Python code block from text, handling various formats"""
    if not text:
        return None

    # 1) Standard triple backtick blocks
    pattern = r"```python\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2) Triple backtick without language specifier
    pattern = r"```\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Check if it looks like Python code
        if any(
            keyword in content
            for keyword in ["import ", "def ", "class ", "print(", "if __name__"]
        ):
            return content

    # 3) Started fence without terminator: take everything after the marker
    match = re.search(r"```python\s*\n([\s\S]*)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # 4) Heuristic detection: collect lines that look like Python
    code_like = re.compile(
        r"^\s*(from\s+\w+\s+import|import\s+\w+|def\s+\w+\(|class\s+\w+|"
        r"if\s+__name__\s*==\s*['\"]__main__['\"]|with\s+open\(|try:|except|"
        r"for\s+\w+\s+in\s+|while\s+|os\.|json\.|pathlib\.|open\(|print\(|@\w+)"
    )
    lines = text.splitlines()
    indices = [i for i, ln in enumerate(lines) if code_like.search(ln)]
    if len(indices) >= 3:
        start = max(0, indices[0] - 0)
        # extend to the end, but drop trailing prose if present: stop when
        # we hit two consecutive blank lines after last code-like line
        end = len(lines)
        last_code_idx = indices[-1]
        # trim trailing prose
        for i in range(last_code_idx + 1, len(lines) - 1):
            if lines[i].strip() == "" and lines[i + 1].strip() == "":
                end = i
                break
        snippet = "\n".join(lines[start:end]).strip()
        return snippet if snippet else None

    return None


def snippet_is_stdlib_only(snippet: str) -> bool:
    """Very lightweight check: block obvious third‑party imports and interactive bits."""
    lowered = snippet.lower()
    if (
        "pip install" in lowered
        or "dockerfile" in lowered
        or "requirements.txt" in lowered
    ):
        return False
    if re.search(r"\binput\s*\(", lowered):
        return False
    import_lines = re.findall(
        r"^\s*(?:from|import)\s+([\w\.]+)", snippet, flags=re.MULTILINE
    )
    for mod in import_lines:
        base = mod.split(".")[0].lower()
        if base in (t.lower() for t in _BANNED_IMPORT_TOKENS):
            return False
    return True
