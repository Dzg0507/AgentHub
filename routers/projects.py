#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projects router for Agent Control Hub
Handles all project-related endpoints
"""
import os
import json
import asyncio
import uuid
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

# Add project root to Python path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project dependencies
from services.plan import AgentProject, ProjectStatus
from utils.env import ensure_python_venv, run_command
from utils.files import run_python_snippet_in_dir, extract_python_code_block
from services.pipeline import process_project_full_pipeline, cleanup_old_projects, _force_minimal_scaffold

# Configuration
PROJECTS_DIR = Path("agent_workspace/projects")
WORKSPACE_DIR = Path("agent_workspace")

# Global projects store
projects: Dict[str, AgentProject] = {}

# Create router
projects_router = APIRouter()

@projects_router.post("/projects")
async def create_project(request: dict):
    """Create a new project with the given prompt"""
    prompt = request.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Create project ID and temp project directory (temp-like behavior)
    project_id = str(uuid.uuid4())
    (PROJECTS_DIR / project_id).mkdir(parents=True, exist_ok=True)

    # Optional language hint (default python)
    language = str(request.get("language", "python")).lower().strip() or "python"
    guidance = str(request.get("guidance", "standard")).lower().strip() or "standard"

    # Create project
    project = AgentProject(project_id, prompt, language=language, guidance=guidance)
    projects[project_id] = project

    # Start background processing
    asyncio.create_task(process_project_full_pipeline(project_id))
    # trigger async cleanup in background (fire-and-forget)
    asyncio.create_task(cleanup_old_projects())

    return {
        "project_id": project_id,
        "status": "processing",
        "message": "Project created and processing started"
    }

@projects_router.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Get project status and information"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    return project.to_dict()

@projects_router.get("/projects")
async def list_projects():
    """List all projects with their status"""
    return {
        "projects": [project.to_dict() for project in projects.values()],
        "total_count": len(projects)
    }

@projects_router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    # Clean up project files
    project_dir = PROJECTS_DIR / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir)

    del projects[project_id]

    return {"message": "Project deleted successfully"}

@projects_router.get("/projects/{project_id}/download")
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

@projects_router.get("/projects/{project_id}/files")
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

@projects_router.get("/projects/{project_id}/logs")
async def get_project_logs(project_id: str):
    """Return execution logs for a project"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = projects[project_id]
    project_dir = PROJECTS_DIR / project_id
    
    # Try to read from execution.log file first
    log_file = project_dir / "execution.log"
    file_logs = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                file_logs = [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            pass
    
    # Combine file logs with in-memory logs
    all_logs = file_logs + project.execution_log
    
    return {"execution_log": all_logs}

@projects_router.post("/projects/{project_id}/retry")
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

@projects_router.post("/projects/{project_id}/force-scaffold")
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

@projects_router.post("/projects/{project_id}/venv")
async def create_project_venv(project_id: str):
    """Create per-project venv and install requirements if present."""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    info = ensure_python_venv(project_dir)
    return {"ok": True, **info}

@projects_router.post("/projects/{project_id}/setup")
async def setup_project(project_id: str):
    """Run stack-specific setup commands (install deps/build)."""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    agent = projects[project_id]
    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    lang = getattr(agent, "language", "python").lower()
    logs: list[dict] = []
    def log_step(name: str, result: dict):
        logs.append({"step": name, "returncode": result.get("returncode"), "stdout": result.get("stdout"), "stderr": result.get("stderr")})
    if lang == "python":
        info = ensure_python_venv(project_dir)
        logs.append({"step": "venv", **info})
    elif lang in ("node", "react-ts", "threejs"):
        # prefer npm if available
        cmd = ["npm", "install"]
        result = run_command(cmd, cwd=project_dir, timeout_sec=900)
        log_step("npm install", result)
    elif lang == "go":
        result = run_command(["go", "mod", "tidy"], cwd=project_dir, timeout_sec=300)
        log_step("go mod tidy", result)
    elif lang == "rust":
        result = run_command(["cargo", "build"], cwd=project_dir, timeout_sec=900)
        log_step("cargo build", result)
    elif lang == "java":
        result = run_command(["gradle", "build"], cwd=project_dir, timeout_sec=900)
        log_step("gradle build", result)
    else:
        logs.append({"step": "noop", "returncode": 0, "stdout": "No setup for language", "stderr": ""})
    return {"ok": True, "logs": logs}
