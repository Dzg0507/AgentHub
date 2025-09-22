#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main FastAPI application for Agent Control Hub
"""
import asyncio
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import routers
from routers.projects import projects_router, projects

# Import services
from services.pipeline import periodic_cleanup_task
from services import pipeline

# Import configuration
from core.config import SERVER_HOST, SERVER_PORT, WORKSPACE_DIR, PROJECTS_DIR, DEPLOY_DIR

# Connect the projects store between modules
pipeline.projects = projects

# Create necessary directories
for directory in [WORKSPACE_DIR, PROJECTS_DIR, DEPLOY_DIR]:
    directory.mkdir(exist_ok=True)

# Start background cleanup task using lifespan
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    asyncio.create_task(periodic_cleanup_task())
    yield
    # Shutdown (if needed)
    pass

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Advanced Agent Hub",
        description="A centralized hub for multi-agent code creation, enhancement, and deployment",
        version="1.0.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(projects_router)

    # Root endpoint
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

    # Health check endpoint
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

    # HTML Interface
    @app.get("/ui", response_class=HTMLResponse)
    async def get_ui():
        """Serve the HTML interface"""
        from utils.prompts import load_template
        return load_template("ui.html")

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Agent Hub Server on {SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
