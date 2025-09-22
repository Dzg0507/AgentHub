#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Control Hub - Unified Startup Script
Starts both FastAPI server and Streamlit UI with one command
"""
import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path
import argparse
import json
from typing import Optional, List

class HubManager:
    """Manages the Agent Control Hub processes"""
    
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.streamlit_process: Optional[subprocess.Popen] = None
        self.running = False
        
    def start_server(self, host: str = "127.0.0.1", port: int = 8000) -> bool:
        """Start the FastAPI server"""
        try:
            print(f"🚀 Starting FastAPI server on {host}:{port}...")
            
            # Try the new modular version first
            if Path("server/app.py").exists():
                cmd = [sys.executable, "server/app.py"]
            else:
                cmd = [sys.executable, "agent_hub_server.py"]
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait a moment to check if server started successfully
            time.sleep(2)
            if self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                print(f"❌ Server failed to start:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
            
            print(f"✅ FastAPI server started successfully on {host}:{port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def start_streamlit(self, port: int = 8501) -> bool:
        """Start the Streamlit UI"""
        try:
            print(f"🎨 Starting Streamlit UI on port {port}...")
            
            # Check if enhanced UI exists
            if Path("src/ui/streamlit/streamlit_app.py").exists():
                cmd = [sys.executable, "-m", "streamlit", "run", "src/ui/streamlit/streamlit_app.py", "--server.port", str(port)]
            else:
                print("❌ streamlit_app.py not found")
                return False
            
            self.streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait a moment to check if streamlit started successfully
            time.sleep(3)
            if self.streamlit_process.poll() is not None:
                stdout, stderr = self.streamlit_process.communicate()
                print(f"❌ Streamlit failed to start:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
            
            print(f"✅ Streamlit UI started successfully on http://localhost:{port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Streamlit: {e}")
            return False
    
    def stop_server(self):
        """Stop the FastAPI server"""
        if self.server_process and self.server_process.poll() is None:
            print("🛑 Stopping FastAPI server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            print("✅ FastAPI server stopped")
    
    def stop_streamlit(self):
        """Stop the Streamlit UI"""
        if self.streamlit_process and self.streamlit_process.poll() is None:
            print("🛑 Stopping Streamlit UI...")
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
            print("✅ Streamlit UI stopped")
    
    def stop_all(self):
        """Stop all processes"""
        self.running = False
        self.stop_streamlit()
        self.stop_server()
    
    def monitor_processes(self):
        """Monitor processes and restart if needed"""
        while self.running:
            try:
                # Check server
                if self.server_process and self.server_process.poll() is not None:
                    print("⚠️ Server process died, restarting...")
                    self.start_server()
                
                # Check streamlit
                if self.streamlit_process and self.streamlit_process.poll() is not None:
                    print("⚠️ Streamlit process died, restarting...")
                    self.start_streamlit()
                
                time.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
                time.sleep(5)
    
    def start_hub(self, server_host: str = "127.0.0.1", server_port: int = 8000, streamlit_port: int = 8501, monitor: bool = True):
        """Start the complete Agent Control Hub"""
        print("🤖 Starting Agent Control Hub...")
        print("=" * 50)
        
        # Start server
        if not self.start_server(server_host, server_port):
            return False
        
        # Start streamlit
        if not self.start_streamlit(streamlit_port):
            self.stop_server()
            return False
        
        self.running = True
        
        print("=" * 50)
        print("🎉 Agent Control Hub is running!")
        print(f"📊 FastAPI Server: http://{server_host}:{server_port}")
        print(f"🎨 Streamlit UI: http://localhost:{streamlit_port}")
        print(f"📚 API Docs: http://{server_host}:{server_port}/docs")
        print("=" * 50)
        print("Press Ctrl+C to stop all services")
        
        try:
            if monitor:
                self.monitor_processes()
            else:
                # Just wait for keyboard interrupt
                while self.running:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Agent Control Hub...")
            self.stop_all()
            print("✅ Shutdown complete")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Received shutdown signal...")
    sys.exit(0)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Agent Control Hub - Unified Startup")
    parser.add_argument("--server-host", default="127.0.0.1", help="FastAPI server host")
    parser.add_argument("--server-port", type=int, default=8000, help="FastAPI server port")
    parser.add_argument("--streamlit-port", type=int, default=8501, help="Streamlit UI port")
    parser.add_argument("--no-monitor", action="store_true", help="Disable process monitoring")
    parser.add_argument("--server-only", action="store_true", help="Start only the FastAPI server")
    parser.add_argument("--streamlit-only", action="store_true", help="Start only the Streamlit UI")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create hub manager
    hub = HubManager()
    
    try:
        if args.server_only:
            hub.start_server(args.server_host, args.server_port)
            print("Press Ctrl+C to stop the server")
            while True:
                time.sleep(1)
        elif args.streamlit_only:
            hub.start_streamlit(args.streamlit_port)
            print("Press Ctrl+C to stop Streamlit")
            while True:
                time.sleep(1)
        else:
            hub.start_hub(
                server_host=args.server_host,
                server_port=args.server_port,
                streamlit_port=args.streamlit_port,
                monitor=not args.no_monitor
            )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        hub.stop_all()
    except Exception as e:
        print(f"❌ Error: {e}")
        hub.stop_all()
        sys.exit(1)

if __name__ == "__main__":
    main()
