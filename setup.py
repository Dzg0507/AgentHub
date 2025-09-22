#!/usr/bin/env python3
"""
Setup script for Agent Control Hub
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
else:
    # Fallback requirements if file doesn't exist
    requirements = [
        "ag2[gemini]>=0.8",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.22",
        "python-dotenv>=1.0",
        "streamlit>=1.37",
        "requests>=2.31",
        "psutil>=5.9",
        "google-generativeai>=0.3.0",
        "pytest>=7.0",
        "pytest-cov>=4.0",
        "flake8>=6.0",
        "black>=23.0",
    ]

setup(
    name="agent-control-hub",
    version="1.0.0",
    author="Agent Control Hub Team",
    author_email="contact@agentcontrolhub.dev",
    maintainer="Dzg0507",
    maintainer_email="contact@agentcontrolhub.dev",
    description="A centralized hub for multi-agent code creation, enhancement, and deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dzg0507/AgentHub",
    project_urls={
        "Homepage": "https://github.com/Dzg0507/AgentHub",
        "Documentation": "https://github.com/Dzg0507/AgentHub#readme",
        "Repository": "https://github.com/Dzg0507/AgentHub.git",
        "Bug Tracker": "https://github.com/Dzg0507/AgentHub/issues",
        "Changelog": "https://github.com/Dzg0507/AgentHub/blob/main/CHANGELOG.md",
        "Contributing": "https://github.com/Dzg0507/AgentHub/blob/main/CONTRIBUTING.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Software Development :: User Interfaces",
        "Framework :: Streamlit",
        "Framework :: FastAPI",
    ],
    keywords=[
        "ai", "artificial-intelligence", "code-generation", "multi-agent", 
        "streamlit", "fastapi", "llm", "machine-learning", "automation",
        "development-tools", "code-assistant", "agent-framework"
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
            "mkdocs-mermaid2-plugin>=1.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
            "pre-commit>=3.0.0",
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
            "mkdocs-mermaid2-plugin>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agent-hub=src.main:main",
            "agent-control-hub=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.in", "*.json"],
    },
    zip_safe=False,
    platforms=["any"],
    license="MIT",
    license_files=["LICENSE"],
)
