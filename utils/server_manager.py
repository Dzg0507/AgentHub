#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server management utilities for Agent Control Hub
Provides functions to start, stop, and restart the FastAPI server
"""
import os
import sys
import time
import subprocess
import signal
import psutil
from pathlib import Path
from typing import Optional, Dict, Any
import json


class ServerManager:
    """Manages the FastAPI server process"""

    def __init__(self, server_host: str = "127.0.0.1", server_port: int = 8000):
        self.server_host = server_host
        self.server_port = server_port
        self.server_process: Optional[subprocess.Popen] = None
        self.pid_file = Path("server.pid")

    def is_server_running(self) -> bool:
        """Check if the server is running"""
        try:
            # Check if PID file exists and process is running
            if self.pid_file.exists():
                with open(self.pid_file, "r") as f:
                    pid = int(f.read().strip())
                return psutil.pid_exists(pid) and psutil.Process(pid).is_running()

            # Fallback: check if port is in use
            for conn in psutil.net_connections():
                if conn.laddr.port == self.server_port and conn.status == "LISTEN":
                    return True

            return False
        except Exception:
            return False

    def get_server_status(self) -> Dict[str, Any]:
        """Get detailed server status"""
        status = {
            "running": False,
            "pid": None,
            "port": self.server_port,
            "host": self.server_host,
            "url": f"http://{self.server_host}:{self.server_port}",
            "uptime": None,
            "memory_usage": None,
            "cpu_percent": None,
        }

        try:
            if self.pid_file.exists():
                with open(self.pid_file, "r") as f:
                    pid = int(f.read().strip())

                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    status.update(
                        {
                            "running": True,
                            "pid": pid,
                            "uptime": time.time() - process.create_time(),
                            "memory_usage": process.memory_info().rss
                            / 1024
                            / 1024,  # MB
                            "cpu_percent": process.cpu_percent(),
                        }
                    )
        except Exception as e:
            status["error"] = str(e)

        return status

    def start_server(self) -> Dict[str, Any]:
        """Start the FastAPI server"""
        try:
            if self.is_server_running():
                return {"success": False, "message": "Server is already running"}

            print(
                f"🚀 Starting FastAPI server on {self.server_host}:{self.server_port}..."
            )

            # Try the new modular version first
            if Path("server/app.py").exists():
                cmd = [sys.executable, "server/app.py"]
            else:
                cmd = [sys.executable, "agent_hub_server.py"]

            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # Start in new session to detach from parent
            )

            # Save PID
            with open(self.pid_file, "w") as f:
                f.write(str(self.server_process.pid))

            # Wait a moment to check if server started successfully
            time.sleep(3)

            if self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                return {
                    "success": False,
                    "message": f"Server failed to start: {stderr}",
                    "stdout": stdout,
                    "stderr": stderr,
                }

            return {
                "success": True,
                "message": f"Server started successfully on {self.server_host}:{self.server_port}",
                "pid": self.server_process.pid,
                "url": f"http://{self.server_host}:{self.server_port}",
            }

        except Exception as e:
            return {"success": False, "message": f"Failed to start server: {e}"}

    def stop_server(self) -> Dict[str, Any]:
        """Stop the FastAPI server"""
        try:
            if not self.is_server_running():
                return {"success": False, "message": "Server is not running"}

            # Get PID from file
            pid = None
            if self.pid_file.exists():
                with open(self.pid_file, "r") as f:
                    pid = int(f.read().strip())

            if pid and psutil.pid_exists(pid):
                process = psutil.Process(pid)
                process.terminate()

                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    process.kill()

                # Clean up PID file
                if self.pid_file.exists():
                    self.pid_file.unlink()

                return {"success": True, "message": "Server stopped successfully"}
            else:
                return {"success": False, "message": "Server process not found"}

        except Exception as e:
            return {"success": False, "message": f"Failed to stop server: {e}"}

    def restart_server(self) -> Dict[str, Any]:
        """Restart the FastAPI server"""
        try:
            # Stop server
            stop_result = self.stop_server()
            if not stop_result["success"]:
                return stop_result

            # Wait a moment
            time.sleep(2)

            # Start server
            start_result = self.start_server()
            if start_result["success"]:
                return {"success": True, "message": "Server restarted successfully"}
            else:
                return start_result

        except Exception as e:
            return {"success": False, "message": f"Failed to restart server: {e}"}

    def get_server_logs(self, lines: int = 50) -> Dict[str, Any]:
        """Get recent server logs"""
        try:
            log_file = Path("agent_hub.log")
            if not log_file.exists():
                return {"success": False, "message": "Log file not found"}

            # Read last N lines
            with open(log_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                recent_lines = (
                    all_lines[-lines:] if len(all_lines) > lines else all_lines
                )

            return {
                "success": True,
                "logs": [line.strip() for line in recent_lines],
                "total_lines": len(all_lines),
            }

        except Exception as e:
            return {"success": False, "message": f"Failed to read logs: {e}"}


# Global server manager instance
server_manager = ServerManager()


def get_server_status() -> Dict[str, Any]:
    """Get server status (convenience function)"""
    return server_manager.get_server_status()


def start_server() -> Dict[str, Any]:
    """Start server (convenience function)"""
    return server_manager.start_server()


def stop_server() -> Dict[str, Any]:
    """Stop server (convenience function)"""
    return server_manager.stop_server()


def restart_server() -> Dict[str, Any]:
    """Restart server (convenience function)"""
    return server_manager.restart_server()


def is_server_running() -> bool:
    """Check if server is running (convenience function)"""
    return server_manager.is_server_running()
