#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for utility functions
"""
import pytest
import tempfile
from pathlib import Path
from utils.env import ensure_python_venv, run_command
from utils.files import extract_python_code_block, snippet_is_stdlib_only
from services.plan import FilePlanItem, validate_fileplan_and_outputs


def test_extract_python_code_block():
    """Test Python code block extraction"""

    # Test with proper code block
    text_with_code = """
    Here's some code:
    
    ```python
    print("Hello, world!")
    x = 42
    ```
    
    That's the code.
    """
    result = extract_python_code_block(text_with_code)
    assert result is not None
    assert "print(" in result
    assert "x = 42" in result

    # Test with no code block
    text_without_code = "This is just text with no code blocks."
    result = extract_python_code_block(text_without_code)
    assert result is None

    # Test with unclosed code block
    text_unclosed = """
    ```python
    print("Hello")
    # No closing ```
    """
    result = extract_python_code_block(text_unclosed)
    assert result is not None
    assert "print(" in result


def test_snippet_is_stdlib_only():
    """Test stdlib-only validation"""

    # Test stdlib code
    stdlib_code = """
    import os
    import json
    import pathlib
    
    print("Hello, world!")
    """
    assert snippet_is_stdlib_only(stdlib_code) is True

    # Test with third-party imports
    third_party_code = """
    import numpy
    import requests
    
    print("Hello, world!")
    """
    assert snippet_is_stdlib_only(third_party_code) is False

    # Test with interactive code
    interactive_code = """
    import os
    
    user_input = input("Enter something: ")
    print(user_input)
    """
    assert snippet_is_stdlib_only(interactive_code) is False

    # Test with pip install
    pip_code = """
    import os
    
    # pip install numpy
    print("Hello, world!")
    """
    assert snippet_is_stdlib_only(pip_code) is False


def test_run_command():
    """Test command execution"""

    # Test simple command
    result = run_command(["echo", "hello"], cwd=Path.cwd(), timeout_sec=10)
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]

    # Test command that fails
    result = run_command(["python", "-c", "exit(1)"], cwd=Path.cwd(), timeout_sec=10)
    assert result["returncode"] == 1


def test_ensure_python_venv():
    """Test virtual environment creation"""

    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)

        # Create venv
        info = ensure_python_venv(project_dir)

        assert info["created"] is True
        assert "python" in info

        # Verify venv directory exists
        venv_dir = project_dir / ".venv"
        assert venv_dir.exists()


def test_file_plan_item():
    """Test FilePlanItem dataclass"""

    item = FilePlanItem(
        name="test.py", content="Test content", dependencies=["requirements.txt"]
    )

    assert item.name == "test.py"
    assert item.content == "Test content"
    assert item.dependencies == ["requirements.txt"]


def test_validate_fileplan_and_outputs():
    """Test file plan validation"""

    with tempfile.TemporaryDirectory() as temp_dir:
        sandbox_dir = Path(temp_dir)
        
        # Create test files
        (sandbox_dir / "test.py").write_text("print('hello')")
        (sandbox_dir / "README.md").write_text("# Test")

        # Test valid plan
        valid_plan = [
            FilePlanItem("test.py", "print('hello')", []),
            FilePlanItem("README.md", "# Test", []),
        ]

        result = validate_fileplan_and_outputs(valid_plan, sandbox_dir)
        assert result["valid"] is True
        assert result["missing_files"] == []

        # Test plan with missing files
        (sandbox_dir / "test.py").unlink()  # Remove test.py

        result = validate_fileplan_and_outputs(valid_plan, sandbox_dir)
        assert result["valid"] is False
        assert "test.py" in result["missing_files"]


if __name__ == "__main__":
    pytest.main([__file__])
