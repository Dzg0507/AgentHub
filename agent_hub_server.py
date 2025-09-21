#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Collaborative Agent Hub Server
A centralized hub for multi-agent code creation, enhancement, and deployment
"""
import os
import sys
import json
import asyncio
import threading
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import subprocess
import re
from dotenv import load_dotenv
import logging

# Web server dependencies
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
import uvicorn

# AG2 dependencies (prefer ag2, fallback to legacy autogen)
try:
    import ag2 as autogen
    from ag2 import LLMConfig
except Exception:
    try:
        import autogen  # legacy name
        from autogen import LLMConfig  # type: ignore
    except Exception as e:
        raise ImportError(
            "Missing LLM library. Install one of: ag2[gemini] or autogen"
        ) from e

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---

log_file = Path("agent_hub.log")

# Configure logging with better encoding handling
class SafeStreamHandler(logging.StreamHandler):
    """Safe stream handler that handles encoding issues"""
    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            # Remove emojis and try again
            if hasattr(record, 'msg'):
                record.msg = str(record.msg).encode('ascii', 'ignore').decode('ascii')
            super().emit(record)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        SafeStreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Suppress some of the verbose logging
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Configuration for the language model using AG2's new format
config_list = [
    {
        "model": "gemini-1.5-flash",
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "api_type": "google",
        # For Google models, do not set temperature and top_p together
        "temperature": 0.1,
        "stop": ["```python", "```"],
    }
]
llm_config = LLMConfig(config_list=config_list)

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
WORKSPACE_DIR = Path("agent_workspace")
PROJECTS_DIR = WORKSPACE_DIR / "projects"
DEPLOY_DIR = WORKSPACE_DIR / "deploy"
TEMP_TTL_MINUTES = 60  # temporary project retention

# Create necessary directories
for directory in [WORKSPACE_DIR, PROJECTS_DIR, DEPLOY_DIR]:
    directory.mkdir(exist_ok=True)

def _dir_age_minutes(path: Path) -> float:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return 0
    return max(0.0, (datetime.now().timestamp() - mtime) / 60.0)

async def cleanup_old_projects(ttl_minutes: int = TEMP_TTL_MINUTES) -> None:
    """Delete project directories older than TTL and prune in-memory records."""
    try:
        for sub in PROJECTS_DIR.iterdir():
            if sub.is_dir() and _dir_age_minutes(sub) > ttl_minutes:
                import shutil
                try:
                    shutil.rmtree(sub)
                    # remove project record if exists
                    pid = sub.name
                    if pid in projects:
                        del projects[pid]
                    log.info(f"Cleaned expired project: {pid}")
                except Exception as e:
                    log.warning(f"Failed to clean project {sub}: {e}")
    except Exception as e:
        log.warning(f"Project cleanup scan failed: {e}")

async def periodic_cleanup_task(interval_minutes: int = 10):
    while True:
        await cleanup_old_projects()
        await asyncio.sleep(max(60, interval_minutes * 60))

# --- Data Models ---
class ProjectStatus:
    PENDING = "pending"
    ENHANCING = "enhancing"
    GENERATING = "generating"
    INSTALLING = "installing"
    EXECUTING = "executing"
    TESTING = "testing"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentProject:
    def __init__(self, project_id: str, prompt: str, status: str = ProjectStatus.PENDING, language: str = "python"):
        self.project_id = project_id
        self.original_prompt = prompt
        self.enhanced_prompt = ""
        self.status = status
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.error_message = ""
        self.files_created = []
        self.execution_log = []
        self.deployment_info = {}
        self.language = (language or "python").lower()

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "original_prompt": self.original_prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message,
            "files_created": self.files_created,
            "execution_log": self.execution_log,
            "deployment_info": self.deployment_info,
            "language": self.language,
        }

# Global project storage
projects: Dict[str, AgentProject] = {}

# ---------------- Debug Utilities (Two-Agent Minimal Test) ---------------- #

def extract_python_code_block(text: str) -> Optional[str]:
    """Extract Python code from model output.

    Priority:
    1) Explicit fenced block ```python ... ```
    2) Any fenced block ``` ... ``` or ~~~ ... ~~~
    3) Heuristic: if it "walks and quacks like code", return the most likely
       code region using a regex for typical Python lines.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1) fenced with language
    m = re.search(r"```python\r?\n([\s\S]*?)```", text, re.MULTILINE)
    if m:
        return m.group(1)

    # 2) any fenced block (backticks or tildes)
    m = re.search(r"```\r?\n([\s\S]*?)```", text, re.MULTILINE)
    if m:
        return m.group(1)
    m = re.search(r"~~~\r?\n([\s\S]*?)~~~", text, re.MULTILINE)
    if m:
        return m.group(1)

    # 2b) started fence without terminator: take everything after the marker
    m = re.search(r"```python\r?\n([\s\S]*)$", text, re.MULTILINE)
    if m:
        return m.group(1)

    # 3) Heuristic detection: collect lines that look like Python
    code_like = re.compile(
        r"^\s*(from\s+\w+\s+import|import\s+\w+|def\s+\w+\(|class\s+\w+|"
        r"if\s+__name__\s*==\s*['\"]__main__['\"]|with\s+open\(|try:|except|"
        r"for\s+\w+\s+in\s+|while\s+|os\.|json\.|pathlib\.|open\(|print\(|@\w+)")
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
            if lines[i].strip() == '' and lines[i + 1].strip() == '':
                end = i
                break
        snippet = "\n".join(lines[start:end]).strip()
        return snippet if snippet else None

    return None

def run_python_snippet_in_dir(snippet: str, cwd: Path, timeout_sec: int = 20) -> Dict[str, Any]:
    """Run a python code snippet in a subprocess with given working directory."""
    cwd.mkdir(parents=True, exist_ok=True)
    temp_file = cwd / "_snippet_exec.py"
    temp_file.write_text(snippet, encoding="utf-8")

    # Important: pass only the filename when using cwd to avoid duplicated paths
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
        return {"returncode": -1, "stdout": stdout, "stderr": f"Timeout after {timeout_sec}s\n" + stderr}
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except Exception:
            pass

    return {"returncode": proc.returncode, "stdout": stdout, "stderr": stderr}

# --- Snippet validation and fallback ---
_BANNED_IMPORT_TOKENS = [
    # common third-party libs that break stdlib-only promise
    "pygame", "flask", "fastapi", "requests", "numpy", "pandas", "scipy", "torch",
    "tensorflow", "sqlalchemy", "psycopg2", "matplotlib", "seaborn", "opencv", "cv2",
    "PIL", "pillow", "django", "boto3", "sklearn", "airflow", "pydantic",
]

def snippet_is_stdlib_only(snippet: str) -> bool:
    """Very lightweight check: block obvious third‑party imports and interactive bits.
    Not a full parser—good enough to keep the executor safe and deterministic.
    """
    lowered = snippet.lower()
    # Disallow interactive and network/package guidance
    if "pip install" in lowered or "dockerfile" in lowered or "requirements.txt" in lowered:
        return False
    if re.search(r"\binput\s*\(", lowered):  # avoid interactive
        return False
    # Block common third-party imports
    import_lines = re.findall(r"^\s*(?:from|import)\s+([\w\.]+)", snippet, flags=re.MULTILINE)
    for mod in import_lines:
        base = mod.split(".")[0].lower()
        if base in (t.lower() for t in _BANNED_IMPORT_TOKENS):
            return False
    return True

def minimal_scaffold_snippet(project_dir_str: str) -> str:
    return (
        "import os, json\n"
        f"project_dir = r'{project_dir_str}'\n"
        "os.makedirs(project_dir, exist_ok=True)\n"
        "readme = os.path.join(project_dir, 'README.md')\n"
        "app_py = os.path.join(project_dir, 'app.py')\n"
        "with open(readme, 'w', encoding='utf-8') as f:\n"
        "    f.write('# Project\n\nGenerated by Agent Hub. Stdlib-only scaffold.')\n"
        "with open(app_py, 'w', encoding='utf-8') as f:\n"
        "    f.write('print(\"Hello from Agent Hub!\")')\n"
        "print('FILE_CREATED:' + readme)\n"
        "print('FILE_CREATED:' + app_py)\n"
    )

# --- Enhanced Agent Definitions ---
def create_agents():
    """Create all the specialized agents for the hub"""

    # Prompt Enhancement Agent - Takes user prompts and makes them more comprehensive
    prompt_enhancer = autogen.ConversableAgent(
        name="Prompt_Enhancer",
        system_message="""You are a Prompt Enhancement Specialist. Your role is to take user prompts and enhance them for better code generation.

ENHANCEMENT PROCESS:
1. Analyze the user's request thoroughly
2. Add specific technical requirements and constraints
3. Include best practices and error handling
4. Add testing and validation requirements
5. Specify file structure and organization
6. Include deployment considerations

OUTPUT FORMAT:
- Start with "## ENHANCED PROMPT:" followed by the enhanced prompt
- Make the prompt 3-5x more detailed than the original
- Include specific technologies, frameworks, and requirements
- Add security, performance, and maintainability considerations

Example enhancement:
Original: "Create a web app"
Enhanced: "Create a full-stack web application with the following specifications:
- Frontend: React with TypeScript, responsive design
- Backend: FastAPI with async support
- Database: PostgreSQL with SQLAlchemy ORM
- Authentication: JWT with role-based access control
- Testing: Unit tests with pytest, integration tests
- Deployment: Docker containerization with CI/CD pipeline
- Security: Input validation, SQL injection prevention, XSS protection
- Performance: Caching, database optimization, CDN integration"

Always provide comprehensive, production-ready specifications.""",
        llm_config=llm_config,
    )

    # Code Executor Agent - Executes generated code
    code_executor = autogen.UserProxyAgent(
        name="Code_Executor",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": str(WORKSPACE_DIR),
            "use_docker": False,
        },
        system_message="""You are the Code Executor. You execute Python code that other agents provide to you.

CRITICAL WORKFLOW:
1. When you receive Python code in ```python blocks, EXECUTE IT IMMEDIATELY
2. Report execution results, including any files created or modified
3. Log all execution steps and outputs
4. Handle errors gracefully and provide detailed error information
5. Confirm successful file operations

You have access to create and modify files in the workspace directory.""",
    )

    # Solutions Architect - Generates complete system implementations
    solutions_architect = autogen.ConversableAgent(
        name="Solutions_Architect",
        system_message="""You are a Solutions Architect. Generate complete Python code that creates files. Use only standard library imports. Create simple, working applications.

CRITICAL: Always provide executable Python code in a code block that Code_Executor can run immediately. The code should create actual files on disk.

Example format:
```python
# Create project files
import os
import json

# Create directory
os.makedirs("my_project", exist_ok=True)

# Create main file
with open("my_project/app.py", "w") as f:
    f.write('print("Hello from my app!")')

# Create README
with open("my_project/README.md", "w") as f:
    f.write("# My Project")

print("Project created successfully!")
```

Always generate complete, runnable code.""",
        llm_config=llm_config,
    )

    # Senior Developer - Enhances and refactors existing code
    senior_developer = autogen.ConversableAgent(
        name="Senior_Developer",
        system_message="""You are a Senior Developer focused on code quality, enhancement, and optimization.

YOUR ROLE:
- Enhance existing code with better patterns
- Add error handling, security, and performance improvements
- Refactor for maintainability and scalability
- Generate enhancement code for Code_Executor to execute

FOCUS AREAS:
- Error handling and exception management
- Security vulnerabilities and input validation
- Performance optimization and efficiency
- Code organization and maintainability
- Best practices implementation

Always generate Python code that enhances existing work.""",
        llm_config=llm_config,
    )

    # Quality Assurance - Tests and validates implementations
    quality_assurance = autogen.ConversableAgent(
        name="Quality_Assurance",
        system_message="""You are a Quality Assurance specialist ensuring code excellence and reliability.

YOUR ROLE:
- Generate comprehensive tests for implementations
- Validate functionality and requirements
- Check for edge cases and error conditions
- Generate test code for Code_Executor to run

TESTING APPROACH:
- Unit tests for individual components
- Integration tests for complete workflows
- Error handling validation
- Performance benchmarking
- Security testing

Always generate test code that validates the implementation thoroughly.""",
        llm_config=llm_config,
    )

    # Deployment Specialist - Handles packaging and deployment
    deployment_specialist = autogen.ConversableAgent(
        name="Deployment_Specialist",
        system_message="""You are a Deployment Specialist focused on packaging and deploying applications.

YOUR ROLE:
- Create deployment scripts and configurations
- Generate Docker files and deployment manifests
- Create CI/CD pipeline configurations
- Generate deployment automation code

DEPLOYMENT FOCUS:
- Docker containerization
- Environment configuration
- Startup scripts and service management
- Health checks and monitoring
- Rollback procedures

Always generate deployment-ready code and configurations.""",
        llm_config=llm_config,
    )

    return {
        "prompt_enhancer": prompt_enhancer,
        "code_executor": code_executor,
        "solutions_architect": solutions_architect,
        "senior_developer": senior_developer,
        "quality_assurance": quality_assurance,
        "deployment_specialist": deployment_specialist
    }

# --- FilePlan and Validation Helpers ---
from dataclasses import dataclass

@dataclass
class FilePlanItem:
    path: str
    purpose: str
    content_hint: str | None = None

def _read_local_text(rel_path: str) -> str | None:
    try:
        base = Path(__file__).parent
        p = (base / rel_path).resolve()
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    return None

def _load_prompt(rel_path: str, default_text: str) -> str:
    txt = _read_local_text(rel_path)
    return txt if (txt and txt.strip()) else default_text

FILEPLAN_PROMPT = _load_prompt(
    "prompts/fileplan_prompt.txt",
    (
        "PHASE 1 – FILEPLAN (JSON ONLY): Return a JSON object with key 'files' listing all files to create.\n"
        "Each entry: {path: string (relative to project_dir), purpose: string}. No code blocks.\n"
    ),
)

import json as _json

def parse_fileplan(text: str) -> list[FilePlanItem]:
    try:
        obj = _json.loads(text)
        items = []
        for f in (obj.get("files") or []):
            p = str(f.get("path", "")).strip()
            if not p:
                continue
            items.append(FilePlanItem(path=p, purpose=str(f.get("purpose", "")).strip()))
        return items
    except Exception:
        return []

def validate_fileplan_and_outputs(plan: list[FilePlanItem], sandbox_dir: Path) -> dict:
    # Detect duplicates and invalid paths; check missing files; scan imports
    seen = set()
    duplicates = []
    invalid = []
    missing = []
    for item in plan:
        rel = item.path.lstrip("/\\")
        if rel in seen:
            duplicates.append(rel)
        seen.add(rel)
        if rel == "" or rel.startswith(".."):
            invalid.append(rel)
        p = (sandbox_dir / rel)
        if not p.exists():
            missing.append(rel)
    # import scan
    missing_imports = []
    for py in sandbox_dir.rglob("*.py"):
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in re.findall(r"^\s*from\s+([\w_]+)\s+import|^\s*import\s+([\w_]+)", txt, flags=re.MULTILINE):
            mod = (m[0] or m[1]).strip()
            if not mod:
                continue
            # If it's stdlib likely ignore; if module corresponds to local file, ensure presence
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

# --- Plan Manifest Helpers ---
def _plan_manifest_path(sandbox_dir: Path) -> Path:
    return sandbox_dir / "__plan__.json"

def write_plan_manifest(sandbox_dir: Path, items: list[FilePlanItem], status_by_path: dict[str, str] | None = None) -> None:
    manifest = {
        "files": [
            {
                "path": it.path,
                "purpose": it.purpose,
                "status": (status_by_path or {}).get(it.path, "placeholder"),
            }
            for it in items
        ],
        "updated_at": datetime.now().isoformat(),
    }
    try:
        _plan_manifest_path(sandbox_dir).write_text(_json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        pass

# --- Lightweight Quality Rubric and Checks ---
# This rubric is enforced post-generation: the generator is asked to fix findings via tools.
QUALITY_RUBRIC = _load_prompt(
    "prompts/quality_rubric.txt",
    (
        "CODE QUALITY RUBRIC:\n"
        "- Use only Python standard library\n"
        "- No input() or interactive I/O; prefer functions with parameters\n"
        "- Modules: snake_case.py; Classes: CamelCase; functions/vars: snake_case\n"
        "- Provide module-level docstring and function docstrings (one-liners ok)\n"
        "- Keep lines <= 100 chars where practical\n"
        "- Include README.md with usage instructions\n"
        "- Include at least one simple test (unittest) if any logic exists\n"
    ),
)

def run_quality_checks(sandbox_dir: Path) -> list[str]:
    """Static checks using stdlib only; return list of human-readable issues."""
    issues: list[str] = []
    try:
        import ast
    except Exception:
        return issues

    # 1) Ensure README
    if not (sandbox_dir / "README.md").exists():
        issues.append("Missing README.md with usage instructions")

    # 2) At least one test if code exists
    has_py = any(p.suffix == ".py" for p in sandbox_dir.rglob("*.py"))
    has_test = any("test" in p.name.lower() for p in sandbox_dir.rglob("*.py"))
    if has_py and not has_test:
        issues.append("No tests found; add a basic unittest file")

    # 3) Parse each .py file and check simple conventions
    for py in sandbox_dir.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            issues.append(f"Cannot read {py.name}")
            continue

        # discourage overly long lines
        for i, line in enumerate(text.splitlines(), 1):
            if len(line) > 120:
                issues.append(f"{py.name}:{i} line >120 chars; wrap for readability")
                break

        # AST checks
        try:
            tree = ast.parse(text or "", filename=str(py))
        except Exception as e:
            issues.append(f"{py.name} AST parse error: {e}")
            continue

        # module docstring
        if ast.get_docstring(tree) is None:
            issues.append(f"{py.name} missing module docstring")

        # function/class docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    issues.append(f"{py.name}:{node.lineno} function '{node.name}' missing docstring")
            if isinstance(node, ast.ImportFrom):
                # prevent relative imports
                if node.level and node.level > 0:
                    issues.append(f"{py.name}:{node.lineno} relative import not allowed")

        # discourage input() usage
        if "input(" in text:
            issues.append(f"{py.name} uses input(); avoid interactive I/O")

    return sorted(set(issues))

# --- Core Pipeline Functions ---
async def enhance_prompt(agent: AgentProject, agents: dict) -> bool:
    """Enhance the user prompt with comprehensive requirements"""
    try:
        agent.status = ProjectStatus.ENHANCING
        agent.updated_at = datetime.now()

        log.info(f"Enhancing prompt for project {agent.project_id}")

        # Get enhanced prompt from the prompt enhancer agent
        response = agents["prompt_enhancer"].initiate_chat(
            agents["solutions_architect"],
            message=f"Enhance this user request for code generation: {agent.original_prompt}",
            max_turns=3,
        )

        # Extract enhanced prompt from response
        last_message = response.chat_history[-1]["content"]
        if "## ENHANCED PROMPT:" in last_message:
            agent.enhanced_prompt = last_message.split("## ENHANCED PROMPT:")[1].strip()
        else:
            agent.enhanced_prompt = last_message

        agent.execution_log.append(f"Prompt enhanced: {agent.enhanced_prompt}")
        return True

    except Exception as e:
        # Fallback: continue with the original prompt so downstream steps (including scaffold) can proceed
        agent.enhanced_prompt = agent.original_prompt
        agent.execution_log.append(f"Enhancement error '{e}'. Using original prompt.")
        log.warning(f"Enhancement failed for {agent.project_id}, using original prompt: {e}")
        return True

async def generate_code(agent: AgentProject, agents: dict) -> bool:
    """Generate complete code implementation into a per-project sandbox.
    This uses the same robust approach as /debug/two-agent to guarantee files.
    """
    try:
        agent.status = ProjectStatus.GENERATING
        agent.updated_at = datetime.now()

        log.info(f"Generating code for project {agent.project_id}")

        # Create a sandbox specific to this project under PROJECTS_DIR
        sandbox_dir = PROJECTS_DIR / agent.project_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Seed deterministic instruction/context files to reduce prompt size and guide agents
        try:
            instructions_dir = sandbox_dir / "instructions"
            instructions_dir.mkdir(parents=True, exist_ok=True)
            # Enhanced prompt or original
            context_text = (agent.enhanced_prompt or agent.original_prompt or "").strip()
            (instructions_dir / "CONTEXT.md").write_text(context_text, encoding="utf-8")
            # Constraints / Rubric
            (instructions_dir / "CONSTRAINTS.md").write_text(QUALITY_RUBRIC, encoding="utf-8")
            # Planning guide
            planning_guide = (
                "Use the FilePlan first. Read instructions/CONTEXT.md for requirements and "
                "instructions/CONSTRAINTS.md for quality rules. Then create files using tools only.\n"
                "Favor small cohesive modules over monoliths. Keep lines <= 100 chars."
            )
            (instructions_dir / "PLANNING_GUIDE.md").write_text(planning_guide, encoding="utf-8")
        except Exception:
            pass

        # Prefer TOOL-CALLING generation: expose file ops into the sandbox
        generator = autogen.ConversableAgent(
            name="Project_Generator",
            system_message=(
                "You are a code generation agent. Create project files by CALLING TOOLS only.\n"
                "Available tools: makedirs(path), write_file(path, content, mode), list_tree(path='.'), zip_folder(path='.'), read_text(path, max_bytes=100000)\n"
                "HARD RULES:\n"
                "- Output root (project_dir): " + str(sandbox_dir.as_posix()) + "\n"
                "- Use only stdlib content. No external packages or interactive input.\n"
                "- NEVER emit code blocks or pseudo tool scripts (no ```python, no tool_code).\n"
                "BEFORE EVERY MESSAGE DO THIS CHECKLIST:\n"
                "1) Think and list the complete file plan (all modules referenced, tests, README).\n"
                "2) For each file in the plan, call makedirs/write_file to create it under project_dir.\n"
                "3) If your previous reply referenced a module that doesn't exist yet, create it now via write_file.\n"
                "4) Read context from instructions/CONTEXT.md and constraints from instructions/CONSTRAINTS.md via read_text().\n"
                "5) When done, call list_tree(project_dir). Optionally call zip_folder(project_dir).\n"
                "6) If anything is missing, repeat steps until the tree is complete."
            ),
            llm_config=llm_config,
        )

        # Tool implementations bound to this sandbox
        def _abs(rel: str) -> Path:
            rel = rel.strip().lstrip("/\\")
            p = (sandbox_dir / rel).resolve()
            if not str(p).startswith(str(sandbox_dir.resolve())):
                raise ValueError("Path escapes sandbox")
            return p

        def tool_makedirs(path: str) -> dict:
            p = _abs(path)
            p.mkdir(parents=True, exist_ok=True)
            return {"ok": True, "path": str(p), "created": True}

        def tool_write_file(path: str, content: str, mode: str = "w") -> dict:
            p = _abs(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            # idempotent: if exists with same content, skip write
            prior = None
            if p.exists() and "b" not in mode:
                try:
                    prior = p.read_text(encoding="utf-8")
                except Exception:
                    prior = None
            if prior is not None and prior == content:
                return {"ok": True, "path": str(p), "bytes": len(content.encode("utf-8")), "skipped": True}
            # chunk write for large files
            if "b" in mode:
                data = content.encode("utf-8", errors="ignore")
                with open(p, mode) as f:
                    f.write(data)
            else:
                CHUNK = 200_000
                with open(p, mode, encoding="utf-8") as f:
                    for i in range(0, len(content), CHUNK):
                        f.write(content[i:i+CHUNK])
            return {"ok": True, "path": str(p), "bytes": len(content.encode("utf-8")), "skipped": False}

        def tool_list_tree(path: str = ".") -> dict:
            root = _abs(path)
            files = []
            for q in root.rglob("*"):
                if q.is_file():
                    files.append(str(q.relative_to(sandbox_dir)))
            return {"root": str(root), "files": files, "count": len(files)}

        def tool_zip_folder(path: str = ".", dest_zip: str | None = None) -> dict:
            root = _abs(path)
            import zipfile
            zip_path = _abs(dest_zip) if dest_zip else (sandbox_dir / "project.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for q in root.rglob("*"):
                    if q.is_file():
                        z.write(q, q.relative_to(root))
            return {"zip_path": str(zip_path)}

        def tool_read_text(path: str, max_bytes: int = 100000) -> dict:
            """Read up to max_bytes of a text file inside the sandbox (UTF-8)."""
            p = _abs(path)
            if not p.exists() or not p.is_file():
                return {"ok": False, "error": "not_found"}
            try:
                data = p.read_bytes()[: max(1, min(max_bytes, 5_000_000))]
                try:
                    txt = data.decode("utf-8", errors="ignore")
                except Exception:
                    txt = ""
                return {"ok": True, "path": str(p), "text": txt, "bytes": len(data)}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Register tools
        for func in [tool_makedirs, tool_write_file, tool_list_tree, tool_zip_folder, tool_read_text]:
            try:
                generator.register_function(func)
            except Exception:
                pass

        tool_user = autogen.UserProxyAgent(name="Tool_User", human_input_mode="NEVER")
        # Plan-then-act loop
        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                # Phase 1: request FilePlan JSON
                plan_context = agent.enhanced_prompt or agent.original_prompt
                plan_reply = generator.generate_reply(
                    messages=[{
                        "role": "user",
                        "content": (
                            FILEPLAN_PROMPT
                            + "\nCONTEXT (use this to decide files):\n"
                            + str(plan_context or "")
                            + "\nReturn JSON only."
                        ),
                    }]
                )
                plan_text = plan_reply.get("content") if isinstance(plan_reply, dict) else str(plan_reply)
                plan_items = parse_fileplan(plan_text or "")
                if not plan_items:
                    agent.execution_log.append("FilePlan parse failed; retrying")
                else:
                    # Phase 2: create files via tools
                    # 2a: Materialize placeholders deterministically and write a plan manifest
                    for item in plan_items:
                        rel = item.path
                        tool_makedirs(str(Path(rel).parent))
                        p = _abs(rel)
                        if not p.exists():
                            tool_write_file(rel, f"# {item.purpose}\n")
                    write_plan_manifest(sandbox_dir, plan_items, {it.path: "placeholder" for it in plan_items})

                    # 2b: Populate real content for each planned file using tools
                    plan_context = agent.enhanced_prompt or agent.original_prompt
                    for it in plan_items:
                        rel = it.path
                        p = _abs(rel)
                        existing = ""
                        try:
                            existing = p.read_text(encoding="utf-8") if p.exists() else ""
                        except Exception:
                            existing = ""
                        is_placeholder = (not existing.strip()) or existing.lstrip().startswith("# ") or len(existing) < 30
                        if is_placeholder:
                            try:
                                agent.execution_log.append(f"FILL: requesting content for {rel}")
                                fill_msg = (
                                    "Write the FULL content for the file '" + rel + "' using TOOLS ONLY.\n"
                                    "Call write_file('" + rel + "', <complete_content>) exactly once with the final content.\n"
                                    "Rules: standard library only; include a module docstring; functions with docstrings; clear structure.\n"
                                    "No code fences. No extraneous text.\n"
                                    "CONTEXT:\n" + str(plan_context or "") + "\n"
                                    "PURPOSE:\n" + str(it.purpose or "")
                                )
                                generator.initiate_chat(tool_user, message=fill_msg, max_turns=8)
                            except Exception as e:
                                agent.execution_log.append(f"FILL error for {rel}: {e}")

                    # 2c: Validate and iterate fixes
                    for item in plan_items:
                        # create parent directories and write minimal placeholders first
                        rel = item.path
                        tool_makedirs(str(Path(rel).parent))
                        # create placeholder if file missing
                        p = _abs(rel)
                        if not p.exists():
                            tool_write_file(rel, f"# {item.purpose}\n")
                    # Validate current outputs
                    report = validate_fileplan_and_outputs(plan_items, sandbox_dir)
                    # If missing imports or files, ask the agent to create them now via tools
                    loops = 0
                    while (report["missing_files"] or report["missing_imports"]) and loops < 12:
                        missing_msg = {
                            "role": "user",
                            "content": (
                                "Create the following missing items using tools only, guided by the project context below.\n" +
                                ("Missing files:\n- " + "\n- ".join(report["missing_files"]) + "\n" if report["missing_files"] else "") +
                                ("Missing imports (create corresponding modules or packages):\n- " + "\n- ".join(report["missing_imports"]) if report["missing_imports"] else "") +
                                "\nCONTEXT:\n" + str(plan_context or "")
                            ),
                        }
                        generator.initiate_chat(tool_user, message=missing_msg["content"], max_turns=15)
                        report = validate_fileplan_and_outputs(plan_items, sandbox_dir)
                        loops += 1
                    # Quality gate: run static checks and request fixes via tools
                    quality_issues = run_quality_checks(sandbox_dir)
                    qloops = 0
                    while quality_issues and qloops < 6:
                        fix_msg = (
                            "Apply the following QUALITY FIXES using tools only (no code blocks).\n" +
                            QUALITY_RUBRIC + "\nIssues:\n- " + "\n- ".join(quality_issues) +
                            "\nCONTEXT:\n" + str(plan_context or "")
                        )
                        generator.initiate_chat(tool_user, message=fix_msg, max_turns=10)
                        quality_issues = run_quality_checks(sandbox_dir)
                        qloops += 1
                    # 2d: mark plan entries that now exist as 'filled'
                    status_by = {}
                    for it in plan_items:
                        try:
                            fp = _abs(it.path)
                            ok = fp.exists() and (len(fp.read_text(encoding="utf-8", errors="ignore").strip()) > 30)
                        except Exception:
                            ok = (_abs(it.path)).exists()
                        if ok:
                            status_by[it.path] = "filled"
                    write_plan_manifest(sandbox_dir, plan_items, status_by)
                    # Completeness gate
                    tree = tool_list_tree(".")
                    if len(tree["files"]) >= len(plan_items) and not report["missing_files"] and not report["missing_imports"]:
                        break
                    else:
                        agent.execution_log.append(f"Gate not met (files:{len(tree['files'])}/{len(plan_items)} missing_files:{len(report['missing_files'])} missing_imports:{len(report['missing_imports'])})")
            except Exception as e:
                agent.execution_log.append(f"Plan loop error: {e}")
            # retry will continue

        # Final collection
        created_files = []
        if sandbox_dir.exists():
            for p in sandbox_dir.rglob("*"):
                if p.is_file():
                    created_files.append(str(p.relative_to(WORKSPACE_DIR)))
        agent.files_created = created_files

        # If tools did not produce files, fallback to code-block pathway
        if len(created_files) == 0:
            reply = generator.generate_reply(
                messages=[{"role": "user", "content": f"If you cannot call tools, emit a single python code block that writes files into project_dir."}]
            )
            text = reply.get("content") if isinstance(reply, dict) else str(reply)
            code_block = extract_python_code_block(text or "")
            if not code_block:
                agent.execution_log.append("No code block produced by generator")
                return False

            preamble = (
                f"import os\n"
                f"project_dir = r'{sandbox_dir.as_posix()}'\n"
                f"os.makedirs(project_dir, exist_ok=True)\n"
            )
            snippet = preamble + "\n" + code_block

            # Validate and optionally replace with scaffold
            if not snippet_is_stdlib_only(snippet):
                agent.execution_log.append("Non-stdlib or interactive code detected; using minimal scaffold.")
                snippet = minimal_scaffold_snippet(sandbox_dir.as_posix())

            result = run_python_snippet_in_dir(snippet, cwd=sandbox_dir)

            # Record outputs and collect files
            agent.execution_log.extend([
                f"GENERATOR_STDOUT: {result.get('stdout','').strip()}",
                f"GENERATOR_STDERR: {result.get('stderr','').strip()}",
            ])

            created_files = []
            if sandbox_dir.exists():
                for p in sandbox_dir.rglob("*"):
                    if p.is_file():
                        created_files.append(str(p.relative_to(WORKSPACE_DIR)))
            agent.files_created = created_files
            ok = result.get("returncode", 1) == 0 and len(created_files) > 0
            if not ok:
                # final fallback: always ensure two files
                agent.execution_log.append("Execution failed; writing guaranteed scaffold.")
                scaffold = minimal_scaffold_snippet(sandbox_dir.as_posix())
                result2 = run_python_snippet_in_dir(scaffold, cwd=sandbox_dir)
                created_files = []
                for p in sandbox_dir.rglob("*"):
                    if p.is_file():
                        created_files.append(str(p.relative_to(WORKSPACE_DIR)))
                agent.files_created = created_files
                return result2.get("returncode", 1) == 0 and len(created_files) > 0

        # Optional compile check for .py files
        try:
            import py_compile as _pyc
            compile_errors = []
            for py in sandbox_dir.rglob("*.py"):
                try:
                    _pyc.compile(str(py), doraise=True)
                except Exception as ce:
                    compile_errors.append(f"{py.name}: {ce}")
            if compile_errors:
                agent.execution_log.append("py_compile errors:\n" + "\n".join(compile_errors))
        except Exception:
            pass
        
        return len(agent.files_created) > 0

    except Exception as e:
        agent.status = ProjectStatus.FAILED
        agent.error_message = f"Code generation failed: {str(e)}"
        log.error(f"Code generation failed: {e}")
        return False

async def execute_code(agent: AgentProject, agents: dict) -> bool:
    """Execute the generated code"""
    try:
        agent.status = ProjectStatus.INSTALLING
        agent.updated_at = datetime.now()

        log.info(f"Preparing environment for project {agent.project_id}")

        project_dir = PROJECTS_DIR / agent.project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        # If a requirements.txt exists, create and use a per-project venv and install deps
        req = project_dir / "requirements.txt"
        venv_dir = project_dir / ".venv"
        py_exec = sys.executable

        if req.exists():
            try:
                # Create venv
                import venv as _venv
                builder = _venv.EnvBuilder(with_pip=True, clear=False)
                builder.create(str(venv_dir))
                agent.execution_log.append("VENV: created per-project virtual environment")

                # Determine python path inside venv
                if os.name == "nt":
                    py_exec = str((venv_dir / "Scripts" / "python.exe").resolve())
                    pip_exec = str((venv_dir / "Scripts" / "pip.exe").resolve())
                else:
                    py_exec = str((venv_dir / "bin" / "python").resolve())
                    pip_exec = str((venv_dir / "bin" / "pip").resolve())

                # Install with timeout and a basic allowlist gate (optional: expand rules)
                agent.execution_log.append("VENV: installing requirements.txt")
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
                    agent.execution_log.append("VENV: pip install timed out after 300s")
                agent.execution_log.append("PIP_STDOUT: " + (out or "").strip())
                if err:
                    agent.execution_log.append("PIP_STDERR: " + err.strip())
            except Exception as e:
                agent.execution_log.append(f"VENV error: {e}")

        # After install, mark executing and summarize files
        agent.status = ProjectStatus.EXECUTING
        agent.updated_at = datetime.now()
        log.info(f"Executing code for project {agent.project_id}")

        if project_dir.exists():
            files = list(project_dir.rglob("*"))
            agent.files_created = [str(f.relative_to(WORKSPACE_DIR)) for f in files if f.is_file()]
            agent.execution_log.append(f"Environment ready. Files present: {len(agent.files_created)}")
        else:
            agent.execution_log.append("Environment ready, but no files found.")

        return True

    except Exception as e:
        agent.status = ProjectStatus.FAILED
        agent.error_message = f"Code execution failed: {str(e)}"
        log.error(f"Code execution failed: {e}")
        return False

async def run_tests(agent: AgentProject, agents: dict) -> bool:
    """Run tests on the implementation"""
    try:
        agent.status = ProjectStatus.TESTING
        agent.updated_at = datetime.now()

        log.info(f"Running tests for project {agent.project_id}")

        # Generate and run tests
        response = agents["quality_assurance"].initiate_chat(
            agents["code_executor"],
            message=f"Generate and run comprehensive tests for the implementation based on: {agent.enhanced_prompt}. Provide test code in a code block for Code_Executor to execute.",
            max_turns=3,
        )

        agent.execution_log.append(f"Testing completed: {len(response.chat_history)} turns")
        return True

    except Exception as e:
        agent.status = ProjectStatus.FAILED
        agent.error_message = f"Testing failed: {str(e)}"
        log.error(f"Testing failed: {e}")
        return False

async def deploy_code(agent: AgentProject, agents: dict) -> bool:
    """Package and deploy the implementation"""
    try:
        agent.status = ProjectStatus.DEPLOYING
        agent.updated_at = datetime.now()

        log.info(f"Deploying code for project {agent.project_id}")

        # Generate deployment configuration
        response = agents["deployment_specialist"].initiate_chat(
            agents["code_executor"],
            message=f"Create deployment configuration and scripts for: {agent.enhanced_prompt}. Provide deployment code in a code block for Code_Executor to execute.",
            max_turns=3,
        )

        # Get list of created files
        project_dir = PROJECTS_DIR / agent.project_id
        if project_dir.exists():
            files = list(project_dir.rglob("*"))
            agent.files_created = [str(f.relative_to(WORKSPACE_DIR)) for f in files if f.is_file()]

        agent.execution_log.append(f"Deployment completed. Files created: {len(agent.files_created)}")
        agent.deployment_info = {
            "deployed_at": datetime.now().isoformat(),
            "file_count": len(agent.files_created),
            "project_path": str(project_dir.relative_to(WORKSPACE_DIR))
        }

        return True

    except Exception as e:
        agent.status = ProjectStatus.FAILED
        agent.error_message = f"Deployment failed: {str(e)}"
        log.error(f"Deployment failed: {e}")
        return False

async def process_project_full_pipeline(project_id: str):
    """Process a project through the complete pipeline"""
    if project_id not in projects:
        log.error(f"Project {project_id} not found")
        return

    agent = projects[project_id]
    agents = create_agents()

    log.info(f"Starting full pipeline for project {project_id}")

    # Execute pipeline steps
    pipeline_steps = [
        ("Prompt Enhancement", enhance_prompt),
        ("Code Generation", generate_code),
        ("Code Execution", execute_code),
        ("Testing", run_tests),
        ("Deployment", deploy_code)
    ]

    for step_name, step_func in pipeline_steps:
        log.info(f"Executing {step_name} for project {project_id}")
        success = await step_func(agent, agents)

        if not success:
            log.error(f"Pipeline failed at {step_name} for project {project_id}")
            # Error recovery: if code generation failed, force a minimal scaffold and continue
            if step_name == "Code Generation":
                try:
                    _ok = _force_minimal_scaffold(agent)
                    if _ok:
                        agent.execution_log.append("Recovery: Minimal scaffold created after generation failure. Continuing pipeline.")
                        continue
                except Exception as rec_e:
                    agent.execution_log.append(f"Recovery attempt failed: {rec_e}")
            break

    if agent.status != ProjectStatus.FAILED:
        agent.status = ProjectStatus.COMPLETED
        agent.updated_at = datetime.now()
        log.info(f"Pipeline completed successfully for project {project_id}")

# --- FastAPI Web Server ---
app = FastAPI(
    title="Advanced Agent Hub",
    description="A centralized hub for multi-agent code creation, enhancement, and deployment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Advanced Agent Hub Server",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs",
        "projects_endpoint": "/projects"
    }

def _force_minimal_scaffold(agent: AgentProject) -> bool:
    """Write a minimal scaffold into the project sandbox and update files_created."""
    project_dir = PROJECTS_DIR / agent.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    snippet = minimal_scaffold_snippet(project_dir.as_posix())
    result = run_python_snippet_in_dir(snippet, cwd=project_dir)
    # Record snippet outputs for debugging
    try:
        agent.execution_log.append("SCAFFOLD_STDOUT: " + (result.get("stdout", "").strip()))
        if result.get("stderr"):
            agent.execution_log.append("SCAFFOLD_STDERR: " + result.get("stderr", "").strip())
    except Exception:
        pass

    def _collect() -> list[str]:
        files: list[str] = []
        if project_dir.exists():
            for q in project_dir.rglob("*"):
                if q.is_file():
                    files.append(str(q.relative_to(WORKSPACE_DIR)))
        return files

    created_files = _collect()

    # Fallback: if snippet failed or produced no files, write minimal files directly
    if result.get("returncode", 1) != 0 or len(created_files) == 0:
        try:
            readme = project_dir / "README.md"
            app_py = project_dir / "app.py"
            project_dir.mkdir(parents=True, exist_ok=True)
            readme.write_text("# Project\n\nGenerated by Agent Hub. Stdlib-only scaffold.", encoding="utf-8")
            app_py.write_text('print("Hello from Agent Hub!")', encoding="utf-8")
        except Exception as e:
            agent.execution_log.append(f"Direct scaffold write error: {e}")
        created_files = _collect()

    agent.files_created = created_files
    if len(created_files) > 0:
        agent.error_message = agent.error_message or "Recovered with minimal scaffold."
        return True
    return False

@app.post("/projects")
async def create_project(request: dict):
    """Create a new project with the given prompt"""
    prompt = request.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Create project ID and temp project directory (temp-like behavior)
    project_id = str(uuid.uuid4())
    (PROJECTS_DIR / project_id).mkdir(parents=True, exist_ok=True)

    # Create project
    project = AgentProject(project_id, prompt)
    projects[project_id] = project

    log.info(f"Created new project {project_id} with prompt: {prompt[:100]}...")

    # Start background processing
    asyncio.create_task(process_project_full_pipeline(project_id))
    # trigger async cleanup in background (fire-and-forget)
    asyncio.create_task(cleanup_old_projects())

    return {
        "project_id": project_id,
        "status": "processing",
        "message": "Project created and processing started"
    }

@app.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Get project status and information"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    return project.to_dict()

@app.get("/projects")
async def list_projects():
    """List all projects with their status"""
    return {
        "projects": [project.to_dict() for project in projects.values()],
        "total_count": len(projects)
    }

@app.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    # Clean up project files
    project_dir = PROJECTS_DIR / project_id
    if project_dir.exists():
        import shutil
        shutil.rmtree(project_dir)

    del projects[project_id]
    log.info(f"Deleted project {project_id}")

    return {"message": "Project deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_projects": len(projects),
        "workspace": str(WORKSPACE_DIR),
        "projects_dir": str(PROJECTS_DIR),
        "deploy_dir": str(DEPLOY_DIR)
    }

@app.get("/projects/{project_id}/download")
async def download_project(project_id: str):
    """Download a project as a ZIP file (robust on Windows)."""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    project_dir = PROJECTS_DIR / project_id

    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project files not found")

    # Check if project has any files
    files = list(project_dir.rglob("*"))
    if not files:
        raise HTTPException(status_code=404, detail="No files found in project")

    # Create ZIP file on disk for stable download
    import zipfile
    zip_path = project_dir / f"{project_id}.zip"
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                if file_path.is_file():
                    arcname = file_path.relative_to(project_dir)
                    zipf.write(file_path, arcname)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Zipping failed: {e}")

    # Create a meaningful filename
    project_name = project.enhanced_prompt.lower().replace(" ", "_")[:50] if project.enhanced_prompt else project_id
    filename = f"{project_name}_project.zip"

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=filename,
    )

@app.get("/projects/{project_id}/files")
async def list_project_files(project_id: str):
    """List all files in a project"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = PROJECTS_DIR / project_id

    if not project_dir.exists():
        return {"files": [], "total_size": 0}

    files = []
    total_size = 0

    for file_path in project_dir.rglob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            files.append({
                "name": file_path.name,
                "path": str(file_path.relative_to(WORKSPACE_DIR)),
                "size": size,
                "modified": file_path.stat().st_mtime
            })
            total_size += size

    return {
        "files": sorted(files, key=lambda x: x["name"]),
        "total_files": len(files),
        "total_size": total_size
    }

@app.get("/projects/{project_id}/logs")
async def get_project_logs(project_id: str):
    """Return execution logs for a project"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"execution_log": projects[project_id].execution_log}

@app.post("/projects/{project_id}/retry")
async def retry_project(project_id: str):
    """Retry the pipeline from Code Generation onward. If last status was GENERATING failure,
    re-run generation; otherwise continue from the next failed stage. Returns new status."""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    agent = projects[project_id]
    # simple rule: if failed, try generation again; else just re-run full pipeline
    try:
        agent.execution_log.append("User-triggered retry.")
        agent.status = ProjectStatus.PENDING
        agent.updated_at = datetime.now()
        asyncio.create_task(process_project_full_pipeline(project_id))
        return {"message": "Retry started", "status": agent.status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retry failed: {e}")

@app.post("/projects/{project_id}/force-scaffold")
async def force_scaffold(project_id: str):
    """Force-create a minimal scaffold in the project directory for download and proof of save."""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    agent = projects[project_id]
    try:
        ok = _force_minimal_scaffold(agent)
        if not ok:
            raise RuntimeError("Scaffold creation failed")
        agent.status = ProjectStatus.COMPLETED
        agent.updated_at = datetime.now()
        return {"ok": True, "files_created": len(agent.files_created)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaffold failed: {e}")

@app.post("/debug/two-agent")
async def debug_two_agent(payload: Dict[str, Any]):
    """Run a minimal two-agent pipeline to test file creation end-to-end.

    Request JSON: { "prompt": "<what to build>" }
    Returns the sandbox path, created files, and stdout/stderr.
    """
    prompt = (payload or {}).get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")

    sandbox_id = uuid.uuid4().hex[:8]
    sandbox_dir = WORKSPACE_DIR / f"debug_sandbox_{sandbox_id}"

    generator = autogen.ConversableAgent(
        name="Generator",
        system_message=(
            "Generate ONE python code block that creates files using ONLY stdlib.\n"
            "Hard requirements:\n"
            "- Use the provided variable project_dir (already defined) as the root output directory.\n"
            "- No input(), network calls, or external packages.\n"
            "- Print 'FILE_CREATED:' + filepath for every file you write.\n"
            "- Code must be complete and executable."
        ),
        llm_config=llm_config,
    )

    reply = generator.generate_reply(
        messages=[{"role": "user", "content": f"Create a tiny project for: {prompt}. Use project_dir variable."}]
    )
    text = reply.get("content") if isinstance(reply, dict) else str(reply)
    code_block = extract_python_code_block(text or "")
    if not code_block:
        return {"ok": False, "error": "No code block produced", "raw": text}

    preamble = (
        f"import os\n"
        f"project_dir = r'{sandbox_dir.as_posix()}'\n"
        f"os.makedirs(project_dir, exist_ok=True)\n"
    )
    # Execute within the sandbox; run_python_snippet_in_dir uses filename with cwd
    snippet = preamble + "\n" + code_block
    result = run_python_snippet_in_dir(snippet, cwd=sandbox_dir)

    created_files: List[str] = []
    if sandbox_dir.exists():
        for p in sandbox_dir.rglob("*"):
            if p.is_file():
                created_files.append(str(p.relative_to(WORKSPACE_DIR)))

    return {
        "ok": result.get("returncode") == 0 and len(created_files) > 0,
        "returncode": result.get("returncode"),
        "stdout": result.get("stdout"),
        "stderr": result.get("stderr"),
        "sandbox": str(sandbox_dir.relative_to(WORKSPACE_DIR)),
        "files": created_files,
        "raw": text,
    }

# HTML Interface
@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    """Serve a simple web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Agent Hub</title>
        <link rel="icon" href="data:;base64,iVBORw0KGgo=" />
        <style>
            :root {
                --bg: #0e1116;
                --text: #e6edf3;
                --panel: #161b22;
                --border: #30363d;
                --primary: #238636;
                --primary-hover: #2ea043;
                --accent: #58a6ff;
                --muted: #8b949e;
                --pending: #fff3cd;
                --processing: #d1ecf1;
                --completed: #d4edda;
                --failed: #f8d7da;
            }
            .light {
                --bg: #ffffff;
                --text: #0b0d0f;
                --panel: #f6f8fa;
                --border: #d0d7de;
                --primary: #0969da;
                --primary-hover: #0a66c2;
                --accent: #0969da;
                --muted: #4b5563;
            }
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: var(--bg); color: var(--text); }
            .container { display: flex; gap: 20px; }
            .panel { flex: 1; border: 1px solid var(--border); background: var(--panel); padding: 20px; border-radius: 8px; }
            textarea { width: 100%; height: 150px; margin: 10px 0; background: #0b1220; color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 8px; }
            .light textarea { background: #fff; }
            button { padding: 10px 16px; background: var(--primary); color: white; border: none; border-radius: 6px; cursor: pointer; }
            button:hover { background: var(--primary-hover); }
            button[disabled] { opacity: 0.6; cursor: not-allowed; }
            .toolbar { display:flex; justify-content: space-between; align-items:center; margin-bottom: 12px; }
            .switch { display:flex; align-items:center; gap:8px; color: var(--muted); }
            .status { margin: 10px 0; padding: 10px; border-radius: 6px; border: 1px solid var(--border); background: rgba(88,166,255,0.05); }
            .status.pending { background: var(--pending); color: #856404; }
            .status.processing { background: var(--processing); color: #0c5460; }
            .status.completed { background: var(--completed); color: #155724; }
            .status.failed { background: var(--failed); color: #721c24; }
            pre { background: #0b1220; padding: 10px; border-radius: 6px; overflow-x: auto; color: #e6edf3; border: 1px solid var(--border); }
            .light pre { background: #f6f8fa; color: #24292f; }
            .progress { height: 8px; background: #2d333b; border-radius: 9999px; overflow: hidden; border: 1px solid var(--border); }
            .bar { height: 100%; background: var(--accent); width: 0%; transition: width 0.4s ease; }
            .badge { display:inline-block; background: var(--accent); color:#0b0d0f; padding:2px 8px; border-radius:9999px; font-size:12px; margin-left:6px; }
            .actions { display:flex; gap:8px; margin-top:8px; }
            .spinner { width:16px; height:16px; border:2px solid var(--border); border-top-color: var(--accent); border-radius:50%; animation: spin 0.8s linear infinite; display:inline-block; vertical-align: middle; }
            @keyframes spin { to { transform: rotate(360deg); } }
            .pill { display:inline-flex; align-items:center; gap:6px; padding:4px 8px; background: rgba(88,166,255,0.12); border:1px solid var(--border); border-radius:9999px; font-size:12px; color: var(--text); }
            .muted { color: var(--muted); font-size:12px; }
        </style>
    </head>
    <body>
        <div class="toolbar">
            <div>
                <h1 style="margin:0">Advanced Agent Hub</h1>
                <div style="color:var(--muted)">A centralized hub for multi-agent code creation, enhancement, and deployment</div>
            </div>
            <div class="switch">
                <label><input type="checkbox" id="themeToggle"> Light theme</label>
                <label><input type="checkbox" id="autoRefreshToggle" checked> Auto refresh</label>
                <span id="loading" style="display:none; color: var(--accent)">Loading...</span>
            </div>
        </div>
        <script>
            // Minimal fallbacks to ensure buttons work even if later script fails
            (function(){
                function postJSON(url, payload, cb){
                    try{
                        var xhr = new XMLHttpRequest();
                        xhr.open('POST', url, true);
                        xhr.setRequestHeader('Content-Type','application/json');
                        xhr.onreadystatechange = function(){
                            if(xhr.readyState === 4){
                                var data = null, err = null;
                                try { data = JSON.parse(xhr.responseText || '{}'); }
                                catch(e){ err = e; }
                                cb(err, data, xhr.status);
                            }
                        };
                        xhr.send(JSON.stringify(payload || {}));
                    }catch(e){ cb(e); }
                }
                window.runDebugTest = window.runDebugTest || function(){
                    var promptEl = document.getElementById('debugPrompt');
                    var prompt = promptEl ? promptEl.value : 'Create two tiny files: README.md and app.py that prints Hello';
                    var resEl = document.getElementById('debugResult');
                    if (resEl) resEl.innerHTML = 'Running...';
                    postJSON('/debug/two-agent', { prompt: prompt }, function(err, data){
                        if (!resEl) return;
                        if (err) { resEl.textContent = 'Error: ' + (err.message || err); return; }
                        try {
                            var filesList = '';
                            (data.files || []).forEach(function(f){ filesList += '<li>' + f + '</li>'; });
                            if (!filesList) filesList = '<li>(no files)</li>';
                            var cls = data.ok ? 'completed' : 'failed';
                            resEl.innerHTML = ''
                              + '<div class="status ' + cls + '">'
                              + '<div><strong>Ok:</strong> ' + data.ok + '</div>'
                              + '<div><strong>Sandbox:</strong> ' + (data.sandbox || '') + '</div>'
                              + '<div><strong>Return code:</strong> ' + (data.returncode || '') + '</div>'
                              + '<div><strong>Files:</strong><ul>' + filesList + '</ul></div>'
                              + '<div><strong>Stdout:</strong><pre>' + String(data.stdout||'').replace(/</g,'&lt;') + '</pre></div>'
                              + '<div><strong>Stderr:</strong><pre>' + String(data.stderr||'').replace(/</g,'&lt;') + '</pre></div>'
                              + '</div>';
                        } catch (e) {
                            resEl.textContent = JSON.stringify(data || {}, null, 2);
                        }
                    });
                };
                window.createProject = window.createProject || function(){
                    var el = document.getElementById('promptInput');
                    var p = el ? el.value : '';
                    if (!p || !p.trim()) { alert('Please enter a project description'); return; }
                    var btn = document.getElementById('createBtn');
                    if (btn) { btn.disabled = true; btn.textContent = 'Creating...'; }
                    postJSON('/projects', { prompt: p }, function(err, data, status){
                        if (btn) { btn.disabled = false; btn.textContent = 'Create Project'; }
                        if (err || (status && status >= 400)) { alert('Error creating project'); return; }
                        alert('Project created: ' + (data && data.project_id ? data.project_id : 'unknown'));
                        try { if (el) el.value = ''; } catch(_){}
                        try { if (typeof loadProjects === 'function') loadProjects(); } catch(_){}
                    });
                };
            })();
        </script>

        <div class="container">
            <div class="panel">
                <h2>Create New Project</h2>
                <textarea id="promptInput" placeholder="Enter your project description here..."></textarea>
                <button id="createBtn" type="button" onclick="createProject()">Create Project</button>
            </div>

            <div class="panel">
                <h2>Project Status</h2>
                <div id="projectsList"></div>
            </div>
        </div>

        <div class="panel">
                <h2>Active Projects</h2>
            <div id="activeProjects"></div>
        </div>

        <div class="panel">
            <h2>Debug: Two-Agent File Creation Test</h2>
            <textarea id="debugPrompt" placeholder="e.g., Create two tiny files: README.md and app.py that prints Hello">Create two tiny files: README.md and app.py that prints Hello</textarea>
            <button id="debugBtn" type="button" onclick="runDebugTest()">Run Debug Test</button>
            <div id="debugResult" style="margin-top:12px;"></div>
        </div>

        <script>
            async function createProject() {
                const prompt = document.getElementById('promptInput').value;
                if (!prompt.trim()) {
                    alert('Please enter a project description');
                    return;
                }

                const btn = document.getElementById('createBtn');
                if (btn && btn.disabled) return; // prevent double clicks
                const originalText = btn ? btn.textContent : '';
                if (btn) { btn.disabled = true; btn.textContent = 'Creating...'; }

                try {
                    const response = await fetch('/projects', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: prompt })
                    });

                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(`HTTP ${response.status}: ${errorText}`);
                    }

                    const result = await response.json();
                    alert('Project created: ' + result.project_id);
                    document.getElementById('promptInput').value = '';
                    loadProjects();
                } catch (error) {
                    alert('Error creating project: ' + error.message);
                    console.error('Project creation error:', error);
                } finally {
                    if (btn) { btn.disabled = false; btn.textContent = originalText; }
                }
            }

            let lastCounts = new Map();

            async function loadProjects() {
                const response = await fetch('/projects');
                const data = await response.json();

                const projectsList = document.getElementById('projectsList');
                const activeProjects = document.getElementById('activeProjects');

                projectsList.innerHTML = '';
                activeProjects.innerHTML = '';

                function statusToPercent(s) {
                    const order = ['pending','enhancing','generating','executing','testing','deploying','completed'];
                    const idx = order.indexOf((s||'').toLowerCase());
                    return Math.max(0, Math.min(100, Math.round((idx/(order.length-1))*100)));
                }

                data.projects.forEach(project => {
                    const statusClass = 'status ' + project.status;
                    const createdAt = new Date(project.created_at).toLocaleString();
                    const pct = statusToPercent(project.status);

                    const savingPill = project.status !== 'completed' && project.status !== 'failed' ? '<span class="pill"><span class="spinner"></span> Working…</span>' : (project.status === 'failed' ? '<span class="pill">Failed</span>' : '<span class="pill">Ready</span>');
                    var block = ''
                        + '<div class="status ' + statusClass + '">'
                        +   '<div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">'
                        +     '<div>'
                        +       '<strong>Project: ' + project.project_id.substring(0,8) + '...</strong>'
                        +       '<div class="muted">Created: ' + createdAt + '</div>'
                        +     '</div>'
                        +     savingPill
                        +   '</div>'
                        +   '<div class="progress" title="' + pct + '%" style="margin-top:8px;"><div class="bar" style="width:' + pct + '%"></div></div>'
                        +   '<div class="actions">'
                        +     '<button onclick="loadProjectDetails(\'' + project.project_id + '\')">Details</button>'
                        +     '<button onclick="viewProjectLogs(\'' + project.project_id + '\')">Logs</button>'
                        +     '<button style="background:#d1242f" onclick="deleteProject(\'' + project.project_id + '\')">Delete</button>'
                        +     '<span id="fileCount-' + project.project_id + '" class="badge">Files: 0</span>'
                        +     (project.status === 'failed' ? '<button style="background:#d97706" onclick="retryProject(\'' + project.project_id + '\')">Retry</button>' : '')
                        +     (project.status === 'failed' ? '<button style="background:#6b7280" onclick="forceScaffold(\'' + project.project_id + '\')">Force Scaffold</button>' : '')
                        +   '</div>'
                        + '</div>';
                    projectsList.innerHTML += block;

                    // Fetch live file counts for each project (polling)
                    fetch(`/projects/${project.project_id}/files`).then(r => r.json()).then(info => {
                        const el = document.getElementById(`fileCount-${project.project_id}`);
                        if (el) {
                            const prev = lastCounts.has(project.project_id) ? lastCounts.get(project.project_id) : 0;
                            el.textContent = `Files: ${info.total_files}`;
                            if (info.total_files > prev) {
                                lastCounts.set(project.project_id, info.total_files);
                                el.style.background = '#34d399';
                                el.style.color = '#0b0d0f';
                                setTimeout(()=>{ el.style.background = 'var(--accent)'; el.style.color='#0b0d0f'; }, 900);
                            } else {
                                lastCounts.set(project.project_id, info.total_files);
                            }
                        }
                    }).catch(()=>{});

                    if (project.status === 'completed') {
                        var cblock = ''
                            + '<div>'
                            + '<strong>[COMPLETED] ' + project.project_id.substring(0,8) + '...</strong>'
                            + '<span id="activeFileCount-' + project.project_id + '" class="badge">Files: ...</span><br>'
                            + '<button onclick="downloadProject(\'' + project.project_id + '\')">Download ZIP</button>'
                            + '<button onclick="viewFiles(\'' + project.project_id + '\')">View Files</button>'
                            + '</div>';
                        activeProjects.innerHTML += cblock;

                        // Update completed file counts from API
                        fetch(`/projects/${project.project_id}/files`).then(r => r.json()).then(info => {
                            const el2 = document.getElementById(`activeFileCount-${project.project_id}`);
                            if (el2) el2.textContent = `Files: ${info.total_files}`;
                        }).catch(()=>{});
                    }
                });
            }

            async function loadProjectDetails(projectId) {
                const response = await fetch(`/projects/${projectId}`);
                const project = await response.json();
                alert('Project Details:\n' +
                      '- Status: ' + project.status + '\n' +
                      '- Files Created: ' + project.files_created.length + '\n' +
                      '- Enhanced Prompt: ' + (project.enhanced_prompt || '').substring(0,200) + '...');
            }

            async function downloadProject(projectId) {
                const response = await fetch(`/projects/${projectId}`);
                const project = await response.json();

                // Create download link for project files
                const link = document.createElement('a');
                link.href = `/projects/${projectId}/download`;
                link.download = `project_${projectId}.zip`;
                link.click();
            }

            async function viewFiles(projectId) {
                try {
                    const response = await fetch(`/projects/${projectId}/files`);
                    const data = await response.json();

                    if (data.files.length === 0) {
                        alert(`No files found in project ${projectId}`);
                        return;
                    }

                    let fileList = 'Files in project ' + projectId.substring(0, 8) + '...:\n\n';
                    fileList += 'Total: ' + data.total_files + ' files (' + data.total_size + ' bytes)\n\n';

                    data.files.forEach(file => {
                        const sizeKB = (file.size / 1024).toFixed(1);
                        const date = new Date(file.modified * 1000).toLocaleString();
                        fileList += file.name + ' (' + sizeKB + ' KB) - ' + date + '\n';
                    });

                    alert(fileList);
                } catch (error) {
                    alert('Error loading files: ' + error.message);
                }
            }

            async function viewProjectLogs(projectId) {
                try {
                    const response = await fetch(`/projects/${projectId}/logs`);
                    const data = await response.json();
                    const logs = (data.execution_log || []).join('\n');
                    alert(`Logs for ${projectId.substring(0,8)}...\n\n${logs || '(no logs recorded yet)'}`);
                } catch (e) {
                    alert('Error loading logs');
                }
            }

            async function retryProject(projectId) {
                try {
                    document.getElementById('loading').style.display = 'inline';
                    const res = await fetch(`/projects/${projectId}/retry`, { method: 'POST' });
                    if (!res.ok) throw new Error('HTTP ' + res.status);
                    alert('Retry triggered.');
                    loadProjects();
                } catch (e) {
                    alert('Retry failed: ' + e.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }

            async function forceScaffold(projectId) {
                if (!confirm('Force-create a minimal scaffold for this project?')) return;
                try {
                    document.getElementById('loading').style.display = 'inline';
                    const res = await fetch(`/projects/${projectId}/force-scaffold`, { method: 'POST' });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data && data.detail ? data.detail : ('HTTP ' + res.status));
                    alert('Scaffold created. Files: ' + (data.files_created || 0));
                    loadProjects();
                } catch (e) {
                    alert('Force scaffold failed: ' + e.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }

            async function deleteProject(projectId) {
                if (!confirm('Delete this project and its files?')) return;
                try {
                    const res = await fetch(`/projects/${projectId}`, { method: 'DELETE' });
                    if (!res.ok) throw new Error('HTTP ' + res.status);
                    loadProjects();
                } catch (e) {
                    alert('Delete failed: ' + e.message);
                }
            }

            async function runDebugTest() {
                const prompt = document.getElementById('debugPrompt').value || 'Create two tiny files: README.md and app.py that prints Hello';
                const resEl = document.getElementById('debugResult');
                resEl.innerHTML = 'Running...';
                const btn = document.getElementById('debugBtn');
                if (btn && btn.disabled) return; // prevent double clicks
                const originalText = btn ? btn.textContent : '';
                if (btn) { btn.disabled = true; btn.textContent = 'Running...'; }
                try {
                    document.getElementById('loading').style.display = 'inline';
                    const response = await fetch('/debug/two-agent', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt })
                    });
                    const data = await response.json();
                    let filesList = '';
                    (data.files || []).forEach(function(f){ filesList += '<li>' + f + '</li>'; });
                    if (!filesList) filesList = '<li>(no files)</li>';
                    var cls = data.ok ? 'completed' : 'failed';
                    resEl.innerHTML = ''
                        + '<div class="status ' + cls + '">'
                        + '<div><strong>Ok:</strong> ' + data.ok + '</div>'
                        + '<div><strong>Sandbox:</strong> ' + (data.sandbox || '') + '</div>'
                        + '<div><strong>Return code:</strong> ' + (data.returncode || '') + '</div>'
                        + '<div><strong>Files:</strong><ul>' + filesList + '</ul></div>'
                        + '<div><strong>Stdout:</strong><pre>' + String(data.stdout||'').replace(/</g,'&lt;') + '</pre></div>'
                        + '<div><strong>Stderr:</strong><pre>' + String(data.stderr||'').replace(/</g,'&lt;') + '</pre></div>'
                        + '</div>';
                    // refresh projects list so counts update
                    loadProjects();
                } catch (err) {
                    resEl.innerHTML = 'Error: ' + err.message;
                } finally {
                    if (btn) { btn.disabled = false; btn.textContent = originalText; }
                    document.getElementById('loading').style.display = 'none';
                }
            }

            // Load projects on page load
            function setThemeFromStorage(){
                const saved = localStorage.getItem('theme') || 'dark';
                if (saved === 'light') document.body.classList.add('light'); else document.body.classList.remove('light');
                document.getElementById('themeToggle').checked = (saved === 'light');
            }
            setThemeFromStorage();
            document.getElementById('themeToggle').addEventListener('change', (e)=>{
                if (e.target.checked) { document.body.classList.add('light'); localStorage.setItem('theme','light'); }
                else { document.body.classList.remove('light'); localStorage.setItem('theme','dark'); }
            });

            // Ensure buttons are bound even if inline onclick is ignored
            const dbgBtnEl = document.getElementById('debugBtn');
            if (dbgBtnEl) dbgBtnEl.addEventListener('click', runDebugTest);
            const crtBtnEl = document.getElementById('createBtn');
            if (crtBtnEl) crtBtnEl.addEventListener('click', createProject);

            let refreshTimer = null;
            function startAutoRefresh(){
                if (refreshTimer) clearInterval(refreshTimer);
                refreshTimer = setInterval(()=>{ if (document.getElementById('autoRefreshToggle').checked) { document.getElementById('loading').style.display='inline'; loadProjects().finally(()=>document.getElementById('loading').style.display='none'); } }, 2000);
            }

            loadProjects();
            startAutoRefresh();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- Main Execution ---
def main():
    """Main entry point"""
    log.info("=== Advanced Agent Hub Server ===")
    log.info(f"Workspace directory: {WORKSPACE_DIR}")
    log.info(f"Server starting on: http://{SERVER_HOST}:{SERVER_PORT}")
    log.info(f"Web UI available at: http://{SERVER_HOST}:{SERVER_PORT}/ui")
    log.info("Starting server...")

    # Start the server
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)

if __name__ == "__main__":
    main()
