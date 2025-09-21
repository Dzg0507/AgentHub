# tools.py
import json
from pathlib import Path
   

def read_file(file_path: str) -> str:
    """
    Simple wrapper that reads a text file and returns its content.
    Raises IOError if the file cannot be opened.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
def append_to_file(file_path: str, content: str) -> str:
    """
    Append *content* to *file_path*.
    Returns a short success message; raises IOError on failure.
    """
    # Ensure the parent directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    # Guard against empty content – the agents always send a non‑empty string,
    # but we keep this check for safety.
    if not content or not str(content).strip():
        raise ValueError("Attempted to write empty content to file.")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content)          # No extra newline – the generated code already ends with one
    return f"✅ Wrote {len(content)} characters to {file_path}"
