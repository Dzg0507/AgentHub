#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to check if the server starts correctly
"""
import requests
import time
import subprocess
import sys
import signal
import os

def test_server_startup():
    """Test if the server can start successfully"""
    print("🧪 Testing server startup...")
    
    # Try to start the server
    try:
        # Try the new modular version first
        if os.path.exists("server/app.py"):
            cmd = [sys.executable, "server/app.py"]
        else:
            cmd = [sys.executable, "agent_hub_server.py"]
        
        print(f"Starting server with: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("❌ Server failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
        
        print("✅ Server process is running")
        
        # Try to connect to the server
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is responding to health checks")
                data = response.json()
                print(f"Server status: {data.get('status', 'unknown')}")
            else:
                print(f"⚠️ Server responded with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Could not connect to server: {e}")
        
        # Clean up
        process.terminate()
        process.wait()
        print("✅ Server stopped cleanly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

if __name__ == "__main__":
    test_server_startup()
