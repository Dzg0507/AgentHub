#!/usr/bin/env python3
"""
Setup script for Agent Hub Server
"""
import os
from pathlib import Path

def main():
    print("Agent Hub Server Setup")
    print("=" * 30)

    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("\n1. Setting up environment file...")
        api_key = input("Enter your Google API Key (or press Enter to skip): ").strip()

        if api_key:
            with open(env_file, 'w') as f:
                f.write(f"GOOGLE_API_KEY={api_key}\n")
            print("✓ .env file created with your API key")
        else:
            print("⚠ No API key provided. You'll need to set GOOGLE_API_KEY in .env file")
            with open(env_file, 'w') as f:
                f.write("# Add your Google API key here\n# GOOGLE_API_KEY=your_key_here\n")
    else:
        print("✓ .env file already exists")

    # Check dependencies
    print("\n2. Checking dependencies...")
    try:
        import fastapi
        import uvicorn
        import autogen
        print("✓ All required packages are installed")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Please install required packages:")
        print("pip install fastapi uvicorn")
        return

    # Check API key
    print("\n3. Checking API key...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("✓ Google API key is configured")
    else:
        print("⚠ Google API key not found in environment")
        print("  Please set GOOGLE_API_KEY in your .env file")

    print("\n4. Setup complete!")
    print("\nTo start the server:")
    print("  python agent_hub_server.py")
    print("\nOr use the simple startup script:")
    print("  python start_server.py")
    print("\nServer will be available at: http://127.0.0.1:8000")
    print("Web UI at: http://127.0.0.1:8000/ui")

if __name__ == "__main__":
    main()
