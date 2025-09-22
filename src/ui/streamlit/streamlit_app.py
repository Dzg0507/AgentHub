#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Streamlit UI for Agent Control Hub
Provides comprehensive project management with step-by-step process visualization
"""
import time
import io
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

import requests
import streamlit as st

# Import server management utilities
try:
    from utils.server_manager import (
        get_server_status,
        start_server,
        stop_server,
        restart_server,
        is_server_running,
    )

    SERVER_MANAGEMENT_AVAILABLE = True
except ImportError:
    SERVER_MANAGEMENT_AVAILABLE = False

# Configure API base using env var fallback, persisted in session_state
DEFAULT_API_BASE = os.environ.get("AGENT_HUB_API_BASE", "http://127.0.0.1:8000")
if "api_base" not in st.session_state:
    st.session_state["api_base"] = DEFAULT_API_BASE

# Pipeline step definitions for visualization
PIPELINE_STEPS = [
    {
        "name": "Prompt Enhancement",
        "status": "pending",
        "icon": "🎯",
        "description": "Enhance user prompt with technical requirements",
    },
    {
        "name": "File Planning",
        "status": "pending",
        "icon": "📋",
        "description": "Create structured file plan for the project",
    },
    {
        "name": "Code Generation",
        "status": "pending",
        "icon": "⚡",
        "description": "Generate complete, runnable code implementation",
    },
    {
        "name": "Environment Setup",
        "status": "pending",
        "icon": "🔧",
        "description": "Configure virtual environment and dependencies",
    },
    {
        "name": "Code Execution",
        "status": "pending",
        "icon": "▶️",
        "description": "Execute and test the generated code",
    },
    {
        "name": "Testing",
        "status": "pending",
        "icon": "🧪",
        "description": "Run comprehensive tests on the implementation",
    },
    {
        "name": "Deployment",
        "status": "pending",
        "icon": "🚀",
        "description": "Package and prepare for deployment",
    },
]


def normalize_api_base(base: Optional[str]) -> str:
    b = (base or DEFAULT_API_BASE).strip()
    if not b:
        b = DEFAULT_API_BASE
    if "://" not in b:
        b = f"http://{b}"
    return b.rstrip("/")


def api_url(path: str) -> str:
    base = normalize_api_base(st.session_state.get("api_base"))
    return f"{base}{path}"


def post_json(path: str, payload: Optional[Dict[str, Any]] = None) -> requests.Response:
    url = api_url(path)
    return requests.post(url, json=payload or {}, timeout=(5, 300))


def get_json(path: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    url = api_url(path)
    return requests.get(url, params=params or {}, timeout=60)


def delete(path: str) -> requests.Response:
    url = api_url(path)
    return requests.delete(url, timeout=60)


def download_zip(project_id: str) -> Optional[bytes]:
    url = api_url(f"/projects/{project_id}/download")
    try:
        r = requests.get(url, timeout=300)
        if r.status_code == 200:
            return r.content
    except requests.exceptions.RequestException as e:
        st.warning(f"Download failed: {e}")
    return None


def get_status_icon(status: str) -> str:
    """Get appropriate icon for project status"""
    status_icons = {
        "pending": "⏳",
        "processing": "⚙️",
        "enhancing": "🎯",
        "generating": "⚡",
        "installing": "🔧",
        "executing": "▶️",
        "testing": "🧪",
        "deploying": "🚀",
        "completed": "✅",
        "failed": "❌",
    }
    return status_icons.get(status.lower(), "❓")


def get_status_color(status: str) -> str:
    """Get appropriate color for project status"""
    status_colors = {
        "pending": "#FFA500",
        "processing": "#1E90FF",
        "enhancing": "#9370DB",
        "generating": "#32CD32",
        "installing": "#FF8C00",
        "executing": "#00CED1",
        "testing": "#FF69B4",
        "deploying": "#DC143C",
        "completed": "#228B22",
        "failed": "#DC143C",
    }
    return status_colors.get(status.lower(), "#808080")


def render_pipeline_progress(project: Dict[str, Any]):
    """Render pipeline progress visualization"""
    status = project.get("status", "pending").lower()
    files_created = len(project.get("files_created", []))

    # Determine current step based on status
    current_step_index = 0
    if status == "enhancing":
        current_step_index = 0
    elif status == "generating":
        current_step_index = 2
    elif status == "installing":
        current_step_index = 3
    elif status == "executing":
        current_step_index = 4
    elif status == "testing":
        current_step_index = 5
    elif status == "deploying":
        current_step_index = 6
    elif status == "completed":
        current_step_index = len(PIPELINE_STEPS)
    elif status == "failed":
        # Find the step that failed
        execution_log = project.get("execution_log", [])
        for i, step in enumerate(PIPELINE_STEPS):
            if any(step["name"].lower() in log.lower() for log in execution_log):
                current_step_index = i
                break

    # Create progress visualization
    cols = st.columns(len(PIPELINE_STEPS))
    for i, (col, step) in enumerate(zip(cols, PIPELINE_STEPS)):
        with col:
            if i < current_step_index:
                # Completed step
                st.markdown(f"✅ **{step['name']}**")
                st.caption(step["description"])
            elif i == current_step_index and status not in ["completed", "failed"]:
                # Current step
                st.markdown(f"🔄 **{step['name']}**")
                st.caption(step["description"])
                # Add a spinner for current step
                with st.spinner("In progress..."):
                    pass  # Removed blocking sleep
            else:
                # Pending step
                st.markdown(f"⏳ **{step['name']}**")
                st.caption(step["description"])

    # Overall progress bar
    progress = current_step_index / len(PIPELINE_STEPS)
    st.progress(
        progress,
        text=f"Progress: {int(progress * 100)}% ({current_step_index}/{len(PIPELINE_STEPS)} steps)",
    )


def render_project_card(project: Dict[str, Any]):
    """Render an enhanced project card with comprehensive information"""
    project_id = project.get("project_id")
    status = project.get("status", "pending")
    created_at = project.get("created_at")
    updated_at = project.get("updated_at")
    files_created = project.get("files_created", [])
    execution_log = project.get("execution_log", [])
    language = project.get("language", "python")
    guidance = project.get("guidance", "standard")

    # Create expandable card
    with st.expander(
        f"{get_status_icon(status)} Project {project_id[:8]}... - {status.upper()}",
        expanded=False,
    ):
        # Header with key information
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(
                f"**Language:** {language.upper()} | **Guidance:** {guidance.upper()}"
            )
            st.markdown(f"**Created:** {created_at} | **Updated:** {updated_at}")
            st.markdown(f"**Files Created:** {len(files_created)}")

        with col2:
            # Status badge
            status_color = get_status_color(status)
            st.markdown(
                f"<div style='background-color: {status_color}; color: white; padding: 5px; border-radius: 5px; text-align: center;'>{status.upper()}</div>",
                unsafe_allow_html=True,
            )

        with col3:
            # Quick actions
            if st.button("🔄 Refresh", key=f"refresh-{project_id}"):
                st.rerun()

        # Pipeline progress
        st.markdown("### 🚀 Pipeline Progress")
        render_pipeline_progress(project)

        # Files created
        if files_created:
            st.markdown("### 📁 Generated Files")
            for file_path in files_created:
                st.code(file_path)

        # Execution logs
        if execution_log:
            st.markdown("### 📋 Execution Log")
            log_container = st.container()
            with log_container:
                for i, log_entry in enumerate(
                    execution_log[-10:]
                ):  # Show last 10 entries
                    st.text(f"[{i+1}] {log_entry}")

        # Action buttons
        st.markdown("### ⚡ Actions")
        action_cols = st.columns([1, 1, 1, 1, 1])

        with action_cols[0]:
            if st.button("📋 Logs", key=f"logs-{project_id}"):
                r = get_json(f"/projects/{project_id}/logs")
                if r.ok:
                    raw_logs = r.json().get("execution_log", [])
                    norm_logs = (
                        [str(x) for x in raw_logs]
                        if isinstance(raw_logs, list)
                        else [str(raw_logs)]
                    )
                    st.session_state[f"logs-{project_id}"] = norm_logs

        with action_cols[1]:
            if st.button("🔄 Retry", key=f"retry-{project_id}"):
                with st.spinner("Retrying project..."):
                    r = post_json(f"/projects/{project_id}/retry")
                    if r.ok:
                        st.success("Retry initiated")
                        st.rerun()
                    else:
                        st.error(f"Retry failed: {r.status_code}")

        with action_cols[2]:
            if st.button("⚡ Force Scaffold", key=f"scaf-{project_id}"):
                with st.spinner("Creating scaffold..."):
                    r = post_json(f"/projects/{project_id}/force-scaffold")
                    if r.ok:
                        st.success("Scaffold created")
                        st.rerun()
                    else:
                        st.error(f"Scaffold failed: {r.status_code}")

        with action_cols[3]:
            if st.button("🔧 Setup", key=f"setup-{project_id}"):
                with st.spinner("Setting up project..."):
                    r = post_json(f"/projects/{project_id}/setup")
                    if r.ok:
                        setup_data = r.json()
                        st.session_state[f"setup_logs-{project_id}"] = setup_data.get(
                            "logs", []
                        )
                        st.success("Setup completed")
                    else:
                        st.error(f"Setup failed: {r.status_code}")

        with action_cols[4]:
            if st.button("🗑️ Delete", key=f"delete-{project_id}", type="secondary"):
                with st.spinner("Deleting project..."):
                    r = delete(f"/projects/{project_id}")
                    if r.ok:
                        st.success("Project deleted")
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {r.status_code}")

        # Download section
        st.markdown("### 💾 Download")
        download_cols = st.columns([1, 1])

        with download_cols[0]:
            if st.button("📦 Download ZIP", key=f"download-{project_id}"):
                with st.spinner("Preparing download..."):
                    content = download_zip(project_id)
                    if content:
                        st.download_button(
                            label="💾 Save Project",
                            data=content,
                            file_name=f"{project_id}_{language}_project.zip",
                            mime="application/zip",
                            key=f"dlbtn-{project_id}",
                        )
                    else:
                        st.warning("No files available for download yet")

        with download_cols[1]:
            if st.button("🐍 Create Venv", key=f"venv-{project_id}"):
                with st.spinner("Creating virtual environment..."):
                    r = post_json(f"/projects/{project_id}/venv")
                    if r.ok:
                        venv_data = r.json()
                        st.success(f"Venv created: {venv_data.get('venv_path', 'N/A')}")
                    else:
                        st.error(f"Venv creation failed: {r.status_code}")

        # Setup logs
        setup_logs = st.session_state.get(f"setup_logs-{project_id}")
        if setup_logs:
            st.markdown("### 🔧 Setup Output")
            for entry in setup_logs:
                step_name = entry.get("step", "Unknown")
                returncode = entry.get("returncode", "N/A")
                st.markdown(f"**{step_name}** (exit code: {returncode})")

                stdout = entry.get("stdout", "")
                stderr = entry.get("stderr", "")

                if stdout.strip():
                    st.code(stdout, language="bash")
                if stderr.strip():
                    st.error(stderr)


def render_project_creation_form():
    """Render the project creation form with enhanced UI"""
    st.markdown("### 🚀 Create New Project")

    # Check if there's a generated idea
    default_prompt = ""
    if "generated_idea" in st.session_state:
        default_prompt = st.session_state["generated_idea"]
        del st.session_state["generated_idea"]

    # Create columns for text area and button side by side
    col1, col2 = st.columns([4, 1])

    with col1:
        prompt = st.text_area(
            "Project Description",
            value=default_prompt,
            height=150,
            placeholder="Describe what you want to build...\n\nExample: Create a web scraper that extracts product information from e-commerce sites and stores it in a CSV file.",
            help="Be as specific as possible about your requirements, technologies, and desired features.",
        )

    with col2:
        st.markdown("**Try this:**")
        if st.button("🎲 Try this", key="try_this_button", use_container_width=True):
            try:
                from utils.simple_idea_generator import simple_idea_generator

                idea = simple_idea_generator.get_quick_idea()
                st.session_state["generated_idea"] = idea
                st.rerun()
            except ImportError:
                st.error("Idea generator not available")

    # Form for the rest of the inputs
    with st.form("create_project_form"):
        # Language and guidance selection
        lang_col1, lang_col2 = st.columns(2)

        with lang_col1:
            language = st.selectbox(
                "Programming Language",
                ["python", "node", "react-ts", "threejs", "go", "rust", "java"],
                index=0,
                help="Select the primary programming language for your project",
            )

        with lang_col2:
            guidance = st.selectbox(
                "Guidance Level",
                ["minimal", "standard", "explicit"],
                index=1,
                help="Minimal: Basic implementation\nStandard: Balanced approach\nExplicit: Detailed, comprehensive implementation",
            )

        # Advanced options
        with st.expander("Advanced Options"):
            st.markdown("**Pipeline Configuration**")
            auto_setup = st.checkbox(
                "Auto-setup environment",
                value=True,
                help="Automatically create virtual environment and install dependencies",
            )
            auto_test = st.checkbox(
                "Run tests automatically",
                value=True,
                help="Automatically run tests after code generation",
            )

        # Submit button
        submitted = st.form_submit_button(
            "🚀 Create Project", use_container_width=True, type="primary"
        )

    if submitted:
        if not prompt.strip():
            st.error("❌ Please provide a project description")
        else:
            # Show creation process
            with st.spinner("Creating your project..."):
                try:
                    payload = {
                        "prompt": prompt.strip(),
                        "language": language,
                        "guidance": guidance,
                    }

                    r = post_json("/projects", payload)

                    if r.ok:
                        data = r.json()
                        project_id = data.get("project_id")

                        st.success(f"✅ Project created successfully!")
                        st.info(f"Project ID: `{project_id}`")
                        st.markdown(
                            "Your project is now being processed by our AI agents. You can monitor its progress in the Projects tab."
                        )

                        # Auto-refresh to show the new project
                        st.rerun()
                    else:
                        st.error(
                            f"❌ Failed to create project: {r.status_code} {r.text}"
                        )
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Request failed: {e}")


def render_server_status():
    """Render server status and health information"""
    st.markdown("### 🏥 Server Status")

    # Server management section
    if SERVER_MANAGEMENT_AVAILABLE:
        st.markdown("#### 🔧 Server Management")

        # Get server status
        server_status = get_server_status()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if server_status["running"]:
                st.metric("Server Status", "Running", "🟢")
            else:
                st.metric("Server Status", "Stopped", "🔴")

        with col2:
            if server_status["running"]:
                st.metric("PID", server_status.get("pid", "N/A"))
            else:
                st.metric("PID", "N/A")

        with col3:
            if server_status["running"] and server_status.get("uptime"):
                uptime_hours = server_status["uptime"] / 3600
                st.metric("Uptime", f"{uptime_hours:.1f}h")
            else:
                st.metric("Uptime", "N/A")

        with col4:
            if server_status["running"] and server_status.get("memory_usage"):
                st.metric("Memory", f"{server_status['memory_usage']:.1f} MB")
            else:
                st.metric("Memory", "N/A")

        # Server control buttons
        st.markdown("#### ⚡ Server Controls")
        control_cols = st.columns(4)

        with control_cols[0]:
            if st.button("🚀 Start Server", key="start-server"):
                with st.spinner("Starting server..."):
                    result = start_server()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])

        with control_cols[1]:
            if st.button("🛑 Stop Server", key="stop-server"):
                with st.spinner("Stopping server..."):
                    result = stop_server()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])

        with control_cols[2]:
            if st.button("🔄 Restart Server", key="restart-server"):
                with st.spinner("Restarting server..."):
                    result = restart_server()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])

        with control_cols[3]:
            if st.button("📊 Refresh Status", key="refresh-status"):
                st.rerun()

        st.divider()

    # API health check
    st.markdown("#### 🌐 API Health Check")

    try:
        r = get_json("/health")
        if r.ok:
            health = r.json()

            # API status
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "API Status",
                    health.get("status", "Unknown"),
                    "🟢" if health.get("status") == "healthy" else "🔴",
                )

            with col2:
                st.metric("Active Projects", health.get("active_projects", 0))

            with col3:
                st.metric("API Response", "OK", "🟢")

            # Detailed information
            with st.expander("Detailed Server Information"):
                st.json(health)

        else:
            st.error(f"❌ Health check failed: {r.status_code}")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Cannot connect to server: {e}")
        st.info("Make sure the FastAPI server is running on the configured address.")

        if SERVER_MANAGEMENT_AVAILABLE:
            st.info("💡 Use the Server Management controls above to start the server.")


def render_debug_tools():
    """Render debug and testing tools"""
    st.markdown("### 🛠️ Debug Tools")

    # Two-agent debug test
    st.markdown("#### Quick Agent Test")
    st.markdown(
        "Test the agent pipeline with a simple prompt to verify everything is working."
    )

    with st.form("debug_form"):
        debug_prompt = st.text_input(
            "Debug Prompt",
            placeholder="Create a simple calculator app",
            help="Enter a simple prompt to test the agent pipeline",
        )

        if st.form_submit_button("🧪 Run Debug Test"):
            if debug_prompt.strip():
                with st.spinner("Running debug test..."):
                    try:
                        r = post_json("/debug/two-agent", {"prompt": debug_prompt})
                        if r.ok:
                            result = r.json()

                            st.success("✅ Debug test completed!")

                            # Display results
                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric("Success", "✅" if result.get("ok") else "❌")
                                st.metric("Files Created", len(result.get("files", [])))
                                st.metric(
                                    "Return Code", result.get("returncode", "N/A")
                                )

                            with col2:
                                if result.get("files"):
                                    st.markdown("**Created Files:**")
                                    for file in result.get("files", []):
                                        st.code(file)

                            # Raw response
                            with st.expander("Raw Response"):
                                st.json(result)
                        else:
                            st.error(f"❌ Debug test failed: {r.status_code}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Debug request failed: {e}")
            else:
                st.warning("Please enter a debug prompt")


def main():
    """Main application function"""
    st.set_page_config(
        page_title="Agent Control Hub",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state for better memory management
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True
        st.session_state["last_refresh"] = 0

    # Load custom CSS
    try:
        with open("streamlit_styles.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS file not found, continue without custom styling

    # Header
    st.title("🤖 Agent Control Hub")
    st.markdown("**AI-Powered Code Generation & Project Management Platform**")
    st.markdown("Create, manage, and deploy projects using collaborative AI agents")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Quick start section
        if SERVER_MANAGEMENT_AVAILABLE:
            st.markdown("#### 🚀 Quick Start")
            server_running = is_server_running()

            if not server_running:
                if st.button(
                    "🚀 Start Everything", use_container_width=True, type="primary"
                ):
                    with st.spinner("Starting server..."):
                        result = start_server()
                        if result["success"]:
                            st.success("✅ Server started!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Failed: {result['message']}")
            else:
                st.success("✅ Server is running")
                if st.button("🔄 Restart Server", use_container_width=True):
                    with st.spinner("Restarting..."):
                        result = restart_server()
                        if result["success"]:
                            st.success("✅ Server restarted!")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"❌ Failed: {result['message']}")

            st.divider()

        # Server configuration
        st.markdown("#### 🌐 Server Settings")
        api_base_input = st.text_input(
            "API Base URL",
            value=st.session_state.get("api_base", DEFAULT_API_BASE),
            key="api_base_input",
            help="Base URL for the FastAPI backend server",
        )

        if st.button("🔄 Connect", use_container_width=True):
            st.session_state["api_base"] = api_base_input
            st.success("✅ Connected to server")
            st.rerun()

        # LLM Provider configuration
        st.markdown("#### 🤖 LLM Provider Settings")

        # Provider selection
        provider_options = ["gemini", "together", "openrouter", "local"]
        selected_provider = st.selectbox(
            "LLM Provider",
            options=provider_options,
            index=provider_options.index(
                st.session_state.get("llm_provider", "gemini")
            ),
            help="Select the LLM provider to use for code generation",
        )

        # Model selection based on provider
        if selected_provider == "gemini":
            model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
        elif selected_provider == "together":
            model_options = [
                "togethercomputer/CodeLlama-34b-Instruct",
                "togethercomputer/CodeLlama-13b-Instruct",
                "meta-llama/Llama-2-70b-chat-hf",
                "mistralai/Mistral-7B-Instruct-v0.1",
            ]
        elif selected_provider == "openrouter":
            model_options = [
                "openrouter/mistral-7b",
                "openrouter/llama-2-7b",
                "openrouter/codellama-7b",
                "anthropic/claude-3-haiku",
            ]
        else:  # local
            model_options = ["llama2", "codellama", "mistral", "custom"]

        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=0,
            help=f"Select the model to use with {selected_provider}",
        )

        # API Key input
        api_key = st.text_input(
            f"{selected_provider.upper()} API Key",
            type="password",
            help=f"Enter your {selected_provider} API key",
            placeholder="Enter your API key here...",
        )

        if st.button("💾 Save LLM Settings", use_container_width=True):
            st.session_state["llm_provider"] = selected_provider
            st.session_state["llm_model"] = selected_model
            st.session_state["llm_api_key"] = api_key
            st.success(
                f"✅ LLM settings saved: {selected_provider} with {selected_model}"
            )
            st.rerun()

        # Display current settings
        if "llm_provider" in st.session_state:
            st.info(
                f"**Current LLM Settings:** {st.session_state['llm_provider']} - {st.session_state.get('llm_model', 'default')}"
            )

        # Auto-refresh settings
        st.markdown("#### 🔄 Auto-Refresh")
        auto_refresh = st.toggle(
            "Enable Auto-Refresh",
            value=False,
            help="Automatically refresh project status",
        )
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)

        st.divider()

        # Quick stats
        st.markdown("### 📊 Quick Stats")
        try:
            resp = get_json("/projects")
            if resp.ok:
                projects_data = resp.json()
                total_projects = len(projects_data.get("projects", []))
                st.metric("Total Projects", total_projects)
            else:
                st.metric("Total Projects", "Error")
        except:
            st.metric("Total Projects", "Offline")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏠 Dashboard", "📋 Projects", "🏥 Server Status", "🛠️ Debug Tools"]
    )

    with tab1:
        st.markdown("### 🏠 Welcome to Agent Control Hub")
        st.markdown(
            """
        This platform uses advanced AI agents to help you create, develop, and deploy projects automatically. 
        Our collaborative agent system handles everything from prompt enhancement to final deployment.
        """
        )

        # Create new project
        render_project_creation_form()

        # Recent projects preview
        st.markdown("### 📋 Recent Projects")
        try:
            resp = get_json("/projects")
            if resp.ok:
                projects_data = resp.json()
                projects = projects_data.get("projects", [])

                if projects:
                    # Show last 3 projects
                    for project in projects[-3:]:
                        render_project_card(project)
                else:
                    st.info(
                        "No projects yet. Create your first project using the form above!"
                    )
            else:
                st.error("Failed to load projects")
        except:
            st.error("Cannot connect to server")

    with tab2:
        st.markdown("### 📋 All Projects")

        try:
            resp = get_json("/projects")
            if resp.ok:
                projects_data = resp.json()
                projects = projects_data.get("projects", [])

                if projects:
                    st.markdown(f"**Total Projects:** {len(projects)}")

                    # Project filters
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        status_filter = st.selectbox(
                            "Filter by Status",
                            ["All"] + list(set(p.get("status") for p in projects)),
                        )

                    with col2:
                        language_filter = st.selectbox(
                            "Filter by Language",
                            ["All"]
                            + list(set(p.get("language", "python") for p in projects)),
                        )

                    with col3:
                        sort_by = st.selectbox(
                            "Sort by",
                            [
                                "Updated (Newest)",
                                "Updated (Oldest)",
                                "Created (Newest)",
                                "Created (Oldest)",
                            ],
                        )

                    # Filter and sort projects
                    filtered_projects = projects

                    if status_filter != "All":
                        filtered_projects = [
                            p
                            for p in filtered_projects
                            if p.get("status") == status_filter
                        ]

                    if language_filter != "All":
                        filtered_projects = [
                            p
                            for p in filtered_projects
                            if p.get("language", "python") == language_filter
                        ]

                    # Sort projects
                    if sort_by == "Updated (Newest)":
                        filtered_projects.sort(
                            key=lambda x: x.get("updated_at", ""), reverse=True
                        )
                    elif sort_by == "Updated (Oldest)":
                        filtered_projects.sort(key=lambda x: x.get("updated_at", ""))
                    elif sort_by == "Created (Newest)":
                        filtered_projects.sort(
                            key=lambda x: x.get("created_at", ""), reverse=True
                        )
                    elif sort_by == "Created (Oldest)":
                        filtered_projects.sort(key=lambda x: x.get("created_at", ""))

                    st.markdown(f"**Showing {len(filtered_projects)} projects**")

                    # Display projects
                    for project in filtered_projects:
                        render_project_card(project)
                else:
                    st.info("No projects found. Create your first project!")
            else:
                st.error(f"Failed to load projects: {resp.status_code}")
        except Exception as e:
            st.error(f"Error loading projects: {e}")

        # Auto-refresh using Streamlit's built-in mechanism
        if auto_refresh:
            # Show refresh status
            st.info(f"🔄 Auto-refreshing every {refresh_interval} seconds...")

            # Use a more controlled approach with session state
            if "last_refresh" not in st.session_state:
                st.session_state["last_refresh"] = time.time()

            current_time = time.time()
            if current_time - st.session_state["last_refresh"] >= refresh_interval:
                st.session_state["last_refresh"] = current_time
                st.rerun()
        else:
            # Manual refresh button when auto-refresh is disabled
            if st.button("🔄 Refresh Projects", use_container_width=True):
                st.rerun()

    with tab3:
        render_server_status()

    with tab4:
        render_debug_tools()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🤖 Agent Control Hub - Powered by AI Agents | "
        f"Server: {st.session_state.get('api_base', DEFAULT_API_BASE)}"
        "</div>",
        unsafe_allow_html=True,
    )

    # Cleanup old session data to prevent memory leaks
    if (
        "last_refresh" in st.session_state
        and time.time() - st.session_state["last_refresh"] > 300
    ):  # 5 minutes
        # Clean up old session data
        keys_to_remove = [
            key
            for key in st.session_state.keys()
            if key.startswith("logs-") or key.startswith("setup_logs-")
        ]
        for key in keys_to_remove:
            del st.session_state[key]


if __name__ == "__main__":
    main()
