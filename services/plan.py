from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import re


class ProjectStatus:
    """Project status constants"""

    PENDING = "pending"
    ENHANCING = "enhancing"
    GENERATING = "generating"
    INSTALLING = "installing"
    EXECUTING = "executing"
    TESTING = "testing"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentProject:
    """Agent project data model"""

    project_id: str
    prompt: str
    enhanced_prompt: str = ""
    language: str = "python"
    guidance: str = "standard"
    status: str = ProjectStatus.PENDING
    files_created: List[str] = None
    execution_log: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.files_created is None:
            self.files_created = []
        if self.execution_log is None:
            self.execution_log = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict:
        """Convert project to dictionary"""
        return {
            "project_id": self.project_id,
            "prompt": self.prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "language": self.language,
            "guidance": self.guidance,
            "status": self.status,
            "files_created": self.files_created,
            "execution_log": self.execution_log,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class FilePlanItem:
    name: str
    content: str
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


def validate_fileplan_and_outputs(
    plan: List[FilePlanItem], sandbox_dir: Path
) -> Dict[str, list]:
    seen = set()
    duplicates = []
    invalid = []
    missing = []
    for item in plan:
        rel = item.name.lstrip("/\\")
        if rel in seen:
            duplicates.append(rel)
        seen.add(rel)
        if rel == "" or rel.startswith(".."):
            invalid.append(rel)
        p = sandbox_dir / rel
        if not p.exists():
            missing.append(rel)
    missing_imports = []
    for py in sandbox_dir.rglob("*.py"):
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in re.findall(
            r"^\s*from\s+([\w_]+)\s+import|^\s*import\s+([\w_]+)",
            txt,
            flags=re.MULTILINE,
        ):
            mod = (m[0] or m[1]).strip()
            if not mod:
                continue
            candidate_py = sandbox_dir / f"{mod}.py"
            candidate_pkg = sandbox_dir / mod / "__init__.py"
            if not candidate_py.exists() and not candidate_pkg.exists():
                missing_imports.append(mod)
    return {
        "duplicates": duplicates,
        "invalid": invalid,
        "missing_files": sorted(set(missing)),
        "missing_imports": sorted(set(missing_imports)),
    }
