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
    author_email="",
    description="A centralized hub for multi-agent code creation, enhancement, and deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dzg0507/AgentHub",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agent-hub=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.in"],
    },
    zip_safe=False,
)
