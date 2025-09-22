from pathlib import Path


def read_local_text(rel_path: str) -> str | None:
    try:
        base = Path(__file__).resolve().parent.parent
        p = (base / rel_path).resolve()
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    return None


def load_prompt(rel_path: str, default_text: str) -> str:
    txt = read_local_text(rel_path)
    return txt if (txt and txt.strip()) else default_text
