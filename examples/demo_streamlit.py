#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for Agent Control Hub Streamlit UI
Shows how to use the enhanced UI features
"""
import streamlit as st
import time
from datetime import datetime

def demo_pipeline_progress():
    """Demo the pipeline progress visualization"""
    st.markdown("### 🚀 Pipeline Progress Demo")
    
    # Simulate different pipeline states
    pipeline_steps = [
        {"name": "Prompt Enhancement", "icon": "🎯", "status": "completed"},
        {"name": "File Planning", "icon": "📋", "status": "completed"},
        {"name": "Code Generation", "icon": "⚡", "status": "current"},
        {"name": "Environment Setup", "icon": "🔧", "status": "pending"},
        {"name": "Code Execution", "icon": "▶️", "status": "pending"},
        {"name": "Testing", "icon": "🧪", "status": "pending"},
        {"name": "Deployment", "icon": "🚀", "status": "pending"},
    ]
    
    # Create progress visualization
    cols = st.columns(len(pipeline_steps))
    for i, (col, step) in enumerate(zip(cols, pipeline_steps)):
        with col:
            if step["status"] == "completed":
                st.markdown(f"✅ **{step['name']}**")
                st.caption("Completed successfully")
            elif step["status"] == "current":
                st.markdown(f"🔄 **{step['name']}**")
                st.caption("In progress...")
                with st.spinner("Processing..."):
                    time.sleep(0.1)
            else:
                st.markdown(f"⏳ **{step['name']}**")
                st.caption("Waiting...")
    
    # Progress bar
    progress = 2 / len(pipeline_steps)  # 2 completed out of 7
    st.progress(progress, text=f"Progress: {int(progress * 100)}% (2/7 steps)")

def demo_project_card():
    """Demo the enhanced project card"""
    st.markdown("### 📋 Project Card Demo")
    
    # Simulate project data
    project_data = {
        "project_id": "demo-project-123",
        "status": "generating",
        "created_at": "2024-01-15 10:30:00",
        "updated_at": "2024-01-15 10:35:00",
        "files_created": ["app.py", "README.md", "requirements.txt"],
        "execution_log": [
            "Starting prompt enhancement...",
            "Enhanced prompt with technical requirements",
            "Creating file plan...",
            "Generating code implementation..."
        ],
        "language": "python",
        "guidance": "standard"
    }
    
    # Render project card
    with st.expander(f"🤖 Project {project_data['project_id'][:8]}... - {project_data['status'].upper()}", expanded=True):
        # Header
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Language:** {project_data['language'].upper()} | **Guidance:** {project_data['guidance'].upper()}")
            st.markdown(f"**Created:** {project_data['created_at']} | **Updated:** {project_data['updated_at']}")
            st.markdown(f"**Files Created:** {len(project_data['files_created'])}")
        
        with col2:
            status_color = "#32CD32" if project_data['status'] == "completed" else "#1E90FF"
            st.markdown(f"<div style='background-color: {status_color}; color: white; padding: 5px; border-radius: 5px; text-align: center;'>{project_data['status'].upper()}</div>", unsafe_allow_html=True)
        
        with col3:
            if st.button("🔄 Refresh", key="demo-refresh"):
                st.success("Refreshed!")
        
        # Pipeline progress
        st.markdown("### 🚀 Pipeline Progress")
        demo_pipeline_progress()
        
        # Files created
        st.markdown("### 📁 Generated Files")
        for file_path in project_data['files_created']:
            st.code(file_path)
        
        # Execution logs
        st.markdown("### 📋 Execution Log")
        for i, log_entry in enumerate(project_data['execution_log']):
            st.text(f"[{i+1}] {log_entry}")
        
        # Action buttons
        st.markdown("### ⚡ Actions")
        action_cols = st.columns([1, 1, 1, 1, 1])
        
        with action_cols[0]:
            if st.button("📋 Logs", key="demo-logs"):
                st.info("Logs would be displayed here")
        
        with action_cols[1]:
            if st.button("🔄 Retry", key="demo-retry"):
                st.success("Retry initiated")
        
        with action_cols[2]:
            if st.button("⚡ Force Scaffold", key="demo-scaffold"):
                st.success("Scaffold created")
        
        with action_cols[3]:
            if st.button("🔧 Setup", key="demo-setup"):
                st.success("Setup completed")
        
        with action_cols[4]:
            if st.button("🗑️ Delete", key="demo-delete", type="secondary"):
                st.warning("Project would be deleted")

def demo_creation_form():
    """Demo the project creation form"""
    st.markdown("### 🚀 Project Creation Form Demo")
    
    with st.form("demo_create_form"):
        prompt = st.text_area(
            "Project Description",
            height=150,
            value="Create a web scraper that extracts product information from e-commerce sites and stores it in a CSV file.",
            help="Be as specific as possible about your requirements, technologies, and desired features."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "Programming Language",
                ["python", "node", "react-ts", "threejs", "go", "rust", "java"],
                index=0,
                help="Select the primary programming language for your project"
            )
        
        with col2:
            guidance = st.selectbox(
                "Guidance Level",
                ["minimal", "standard", "explicit"],
                index=1,
                help="Minimal: Basic implementation\nStandard: Balanced approach\nExplicit: Detailed, comprehensive implementation"
            )
        
        # Advanced options
        with st.expander("Advanced Options"):
            st.markdown("**Pipeline Configuration**")
            auto_setup = st.checkbox("Auto-setup environment", value=True, help="Automatically create virtual environment and install dependencies")
            auto_test = st.checkbox("Run tests automatically", value=True, help="Automatically run tests after code generation")
        
        submitted = st.form_submit_button("🚀 Create Project", use_container_width=True, type="primary")
        
        if submitted:
            if prompt.strip():
                with st.spinner("Creating your project..."):
                    time.sleep(2)
                    st.success("✅ Project created successfully!")
                    st.info("Project ID: `demo-project-456`")
                    st.markdown("Your project is now being processed by our AI agents. You can monitor its progress in the Projects tab.")
            else:
                st.error("❌ Please provide a project description")

def main():
    """Demo main function"""
    st.set_page_config(
        page_title="Agent Control Hub Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    # Header
    st.title("🤖 Agent Control Hub - UI Demo")
    st.markdown("**Enhanced Streamlit UI Features Demonstration**")
    st.markdown("This demo showcases the improved user interface elements and step-by-step process visualization.")
    
    # Demo sections
    demo_creation_form()
    st.divider()
    demo_pipeline_progress()
    st.divider()
    demo_project_card()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🤖 Agent Control Hub Demo - Enhanced UI Features"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
