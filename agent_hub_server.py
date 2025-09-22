#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Collaborative Agent Hub Server
A centralized hub for multi-agent code creation, enhancement, and deployment

This is the main entry point that imports from the new modular structure.
For the new modular version, use: python server/app.py
"""
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the new modular app
from server.app import app

# Export the app for backward compatibility
__all__ = ["app"]

# Legacy configuration (for backward compatibility)
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

if __name__ == "__main__":
    import uvicorn

    print(f"Starting Agent Hub Server on {SERVER_HOST}:{SERVER_PORT}")
    print("Using new modular structure")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
