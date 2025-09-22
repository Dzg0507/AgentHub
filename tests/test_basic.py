#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic tests that don't require API keys
"""
import unittest
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests that don't require API keys"""

    def test_imports(self):
        """Test that main modules can be imported"""
        try:
            from src.llm.llm_provider import LLMProvider
            from agents.factory import create_agents
            from services.pipeline import process_project_full_pipeline

            self.assertTrue(True, "All main modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import modules: {e}")

    def test_llm_provider_initialization(self):
        """Test LLM provider can be initialized"""
        try:
            from src.llm.llm_provider import LLMProvider

            provider = LLMProvider(provider="gemini")
            self.assertEqual(provider.provider, "gemini")
        except Exception as e:
            self.fail(f"Failed to initialize LLM provider: {e}")

    def test_agent_factory(self):
        """Test agent factory can create agents"""
        try:
            from agents.factory import create_agents

            agents = create_agents()
            self.assertIsInstance(agents, dict)
            self.assertIn("prompt_enhancer", agents)
            self.assertIn("file_planner", agents)
            self.assertIn("code_generator", agents)
        except Exception as e:
            self.fail(f"Failed to create agents: {e}")

    def test_project_structure(self):
        """Test that required directories exist"""
        required_dirs = [
            "src",
            "src/llm",
            "src/ui",
            "src/ui/streamlit",
            "agents",
            "services",
            "tests",
            "examples",
            "config",
            "scripts",
            "docs",
        ]

        for dir_path in required_dirs:
            self.assertTrue(
                Path(dir_path).exists(), f"Required directory {dir_path} does not exist"
            )

    def test_required_files(self):
        """Test that required files exist"""
        required_files = [
            "README.md",
            "LICENSE",
            "requirements.txt",
            "setup.py",
            ".gitignore",
            "src/llm/llm_provider.py",
            "agents/factory.py",
            "services/pipeline.py",
        ]

        for file_path in required_files:
            self.assertTrue(
                Path(file_path).exists(), f"Required file {file_path} does not exist"
            )


if __name__ == "__main__":
    unittest.main()
