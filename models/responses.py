#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pydantic response models for Agent Control Hub API
"""
from typing import List, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel

class ProjectStatusResponse(BaseModel):
    """Response model for project status"""
    project_id: str
    status: str
    prompt: str
    enhanced_prompt: Optional[str] = None
    language: str = "python"
    guidance: str = "standard"
    files_created: List[str] = []
    execution_log: List[str] = []
    created_at: datetime
    updated_at: datetime

class ProjectListResponse(BaseModel):
    """Response model for project list"""
    projects: List[ProjectStatusResponse]
    total_count: int

class ProjectCreateResponse(BaseModel):
    """Response model for project creation"""
    project_id: str
    status: str
    message: str

class ProjectDeleteResponse(BaseModel):
    """Response model for project deletion"""
    message: str

class ProjectRetryResponse(BaseModel):
    """Response model for project retry"""
    message: str
    status: str

class ProjectScaffoldResponse(BaseModel):
    """Response model for force scaffold"""
    ok: bool
    files_created: int

class ProjectVenvResponse(BaseModel):
    """Response model for venv creation"""
    ok: bool
    venv_path: Optional[str] = None
    python_path: Optional[str] = None
    message: Optional[str] = None

class ProjectSetupResponse(BaseModel):
    """Response model for project setup"""
    ok: bool
    logs: List[Dict[str, Any]]

class ProjectFilesResponse(BaseModel):
    """Response model for project files"""
    files: List[Dict[str, Any]]
    total_files: int
    total_size: int

class ProjectLogsResponse(BaseModel):
    """Response model for project logs"""
    execution_log: List[str]

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    active_projects: int
    workspace: str
    projects_dir: str
    deploy_dir: str

class RootResponse(BaseModel):
    """Response model for root endpoint"""
    message: str
    version: str
    status: str
    docs_url: str
    projects_endpoint: str

class ErrorResponse(BaseModel):
    """Response model for errors"""
    detail: str
