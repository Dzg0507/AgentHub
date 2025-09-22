#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Control Hub - Quick Launcher
Simple launcher script for easy access
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Quick launcher for Agent Control Hub"""
    print("🤖 Agent Control Hub - Quick Launcher")
    print("=" * 40)
    
    if len(sys.argv) < 2:
        print("Usage: python run.py <command>")
        print()
        print("Available commands:")
        print("  ui        - Start Streamlit UI")
        print("  server    - Start FastAPI server")
        print("  both      - Start both UI and server")
        print("  test      - Run tests")
        print("  help      - Show this help")
        return
    
    command = sys.argv[1].lower()
    
    if command == "ui":
        print("🎨 Starting Streamlit UI...")
        os.system("python scripts/start_hub.py --streamlit-only")
    elif command == "server":
        print("🚀 Starting FastAPI server...")
        os.system("python scripts/start_hub.py --server-only")
    elif command == "both":
        print("🚀 Starting both UI and server...")
        os.system("python scripts/start_hub.py")
    elif command == "test":
        print("🧪 Running tests...")
        os.system("python -m pytest tests/ -v")
    elif command == "help":
        print("Available commands:")
        print("  ui        - Start Streamlit UI only")
        print("  server    - Start FastAPI server only")
        print("  both      - Start both UI and server")
        print("  test      - Run the test suite")
        print("  help      - Show this help message")
    else:
        print(f"Unknown command: {command}")
        print("Run 'python run.py help' for available commands")

if __name__ == "__main__":
    main()
