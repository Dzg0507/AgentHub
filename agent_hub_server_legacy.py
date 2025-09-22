#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy Agent Hub Server - Import from new modular structure
This file maintains backward compatibility while the refactor is in progress
"""
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the new modular app
from server.app import app

# Export the app for backward compatibility
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    from core.config import SERVER_HOST, SERVER_PORT
    
    print(f"Starting Agent Hub Server on {SERVER_HOST}:{SERVER_PORT}")
    print("Using new modular structure")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
