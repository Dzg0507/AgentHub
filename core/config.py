#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core configuration for Agent Control Hub
Centralizes constants, paths, and environment settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

# Workspace configuration
WORKSPACE_DIR = Path("agent_workspace")
PROJECTS_DIR = WORKSPACE_DIR / "projects"
DEPLOY_DIR = WORKSPACE_DIR / "deploy"
TEMP_TTL_MINUTES = 60  # temporary project retention

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging configuration
LOG_FILE = Path("agent_hub.log")
LOG_LEVEL = "INFO"

# LLM Configuration
LLM_CONFIG = [
    {
        "model": "gemini-1.5-flash",
        "api_key": GOOGLE_API_KEY,
        "api_type": "google",
        "temperature": 0.1,
        "stop": ["```python", "```"],
    }
]

# Project settings
DEFAULT_LANGUAGE = "python"
DEFAULT_GUIDANCE = "standard"
SUPPORTED_LANGUAGES = [
    "python",
    "node",
    "javascript",
    "react-ts",
    "react",
    "threejs",
    "go",
    "rust",
    "java",
]
GUIDANCE_LEVELS = ["minimal", "standard", "explicit"]

# File patterns
BANNED_IMPORT_TOKENS = [
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

# Pipeline settings
PIPELINE_STEPS = [
    ("Prompt Enhancement", "enhance_prompt"),
    ("Code Generation", "generate_code"),
    ("Code Execution", "execute_code"),
    ("Testing", "run_tests"),
    ("Deployment", "deploy_code"),
]

# Timeout settings
COMMAND_TIMEOUT = 900  # 15 minutes
EXECUTION_TIMEOUT = 30  # 30 seconds
TEST_TIMEOUT = 60  # 1 minute
