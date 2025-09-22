#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to test server startup
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")

try:
    print("1. Testing core.config import...")
    from core.config import SERVER_HOST, SERVER_PORT

    print("✅ core.config imported successfully")
except Exception as e:
    print(f"❌ core.config import failed: {e}")
    sys.exit(1)

try:
    print("2. Testing services.plan import...")
    from services.plan import AgentProject, ProjectStatus

    print("✅ services.plan imported successfully")
except Exception as e:
    print(f"❌ services.plan import failed: {e}")
    sys.exit(1)

try:
    print("3. Testing utils.files import...")
    from utils.files import run_python_snippet_in_dir, extract_python_code_block

    print("✅ utils.files imported successfully")
except Exception as e:
    print(f"❌ utils.files import failed: {e}")
    sys.exit(1)

try:
    print("4. Testing agents.factory import...")
    from agents.factory import create_agents

    print("✅ agents.factory imported successfully")
except Exception as e:
    print(f"❌ agents.factory import failed: {e}")
    sys.exit(1)

try:
    print("5. Testing services.pipeline import...")
    from services.pipeline import (
        process_project_full_pipeline,
        cleanup_old_projects,
        _force_minimal_scaffold,
    )

    print("✅ services.pipeline imported successfully")
except Exception as e:
    print(f"❌ services.pipeline import failed: {e}")
    sys.exit(1)

try:
    print("6. Testing routers.projects import...")
    from routers.projects import projects_router, projects

    print("✅ routers.projects imported successfully")
except Exception as e:
    print(f"❌ routers.projects import failed: {e}")
    sys.exit(1)

try:
    print("7. Testing server.app import...")
    from server.app import app

    print("✅ server.app imported successfully")
except Exception as e:
    print(f"❌ server.app import failed: {e}")
    sys.exit(1)

print("🎉 All imports successful! Server should be able to start.")
