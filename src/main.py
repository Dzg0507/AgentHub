#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Control Hub - Main Entry Point
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for Agent Control Hub"""
    print("🤖 Agent Control Hub")
    print("=" * 50)
    print("Available commands:")
    print("  streamlit  - Start Streamlit UI")
    print("  server     - Start FastAPI server")
    print("  test       - Run tests")
    print("  help       - Show this help")
    print()

    if len(sys.argv) < 2:
        print("Usage: python -m src.main <command>")
        print("Run 'python -m src.main help' for more information")
        return

    command = sys.argv[1].lower()

    if command == "streamlit":
        from src.ui.streamlit.streamlit_app import main as streamlit_main

        streamlit_main()
    elif command == "server":
        from server.app import main as server_main

        server_main()
    elif command == "test":
        import subprocess

        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    elif command == "help":
        print("Available commands:")
        print("  streamlit  - Start the Streamlit web interface")
        print("  server     - Start the FastAPI backend server")
        print("  test       - Run the test suite")
        print("  help       - Show this help message")
    else:
        print(f"Unknown command: {command}")
        print("Run 'python -m src.main help' for available commands")


if __name__ == "__main__":
    main()
