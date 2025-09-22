#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for LLM Provider abstraction
"""
import unittest
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.llm_provider import LLMProvider, create_llm_provider, get_llm_response

class TestLLMProvider(unittest.TestCase):
    """Test cases for LLM Provider"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_messages = [
            {"role": "user", "content": "Hello, world!"}
        ]
    
    def test_provider_initialization(self):
        """Test provider initialization with different configurations"""
        # Test with default values
        provider = LLMProvider()
        self.assertEqual(provider.provider, "gemini")
        
        # Test with custom values
        provider = LLMProvider(
            provider="together",
            api_key="test_key",
            model="test_model"
        )
        self.assertEqual(provider.provider, "together")
        self.assertEqual(provider.api_key, "test_key")
        self.assertEqual(provider.model, "test_model")
    
    def test_available_models(self):
        """Test getting available models for different providers"""
        # Test Gemini models
        provider = LLMProvider(provider="gemini")
        models = provider.get_available_models()
        self.assertIn("gemini-1.5-flash", models)
        
        # Test Together models
        provider = LLMProvider(provider="together")
        models = provider.get_available_models()
        self.assertIn("togethercomputer/CodeLlama-34b-Instruct", models)
        
        # Test OpenRouter models
        provider = LLMProvider(provider="openrouter")
        models = provider.get_available_models()
        self.assertIn("openrouter/mistral-7b", models)
        
        # Test local models
        provider = LLMProvider(provider="local")
        models = provider.get_available_models()
        self.assertIn("llama2", models)
    
    @patch('llm_provider.requests.post')
    def test_together_api_call(self, mock_post):
        """Test Together.ai API call"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from Together!"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        provider = LLMProvider(provider="together", api_key="test_key")
        response = provider._chat_together(self.test_messages, 0.2, 100)
        
        self.assertEqual(response, "Hello from Together!")
        mock_post.assert_called_once()
    
    @patch('llm_provider.requests.post')
    def test_openrouter_api_call(self, mock_post):
        """Test OpenRouter API call"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from OpenRouter!"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        provider = LLMProvider(provider="openrouter", api_key="test_key")
        response = provider._chat_openrouter(self.test_messages, 0.2, 100)
        
        self.assertEqual(response, "Hello from OpenRouter!")
        mock_post.assert_called_once()
    
    @patch('llm_provider.requests.post')
    def test_local_api_call(self, mock_post):
        """Test local LLM API call"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from Local LLM!"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        provider = LLMProvider(provider="local", api_key="test_key")
        response = provider._chat_local(self.test_messages, 0.2, 100)
        
        self.assertEqual(response, "Hello from Local LLM!")
        mock_post.assert_called_once()
    
    def test_unknown_provider(self):
        """Test error handling for unknown provider"""
        provider = LLMProvider(provider="unknown")
        
        with self.assertRaises(ValueError):
            provider.chat(self.test_messages)
    
    def test_factory_function(self):
        """Test factory function for creating providers"""
        provider = create_llm_provider(provider="together")
        self.assertIsInstance(provider, LLMProvider)
        self.assertEqual(provider.provider, "together")
    
    @patch('llm_provider.create_llm_provider')
    def test_get_llm_response(self, mock_create):
        """Test get_llm_response convenience function"""
        # Mock provider
        mock_provider = MagicMock()
        mock_provider.chat.return_value = "Test response"
        mock_create.return_value = mock_provider
        
        response = get_llm_response(self.test_messages)
        
        self.assertEqual(response, "Test response")
        mock_create.assert_called_once()
        mock_provider.chat.assert_called_once_with(self.test_messages, 0.2, 2048)

class TestLLMProviderIntegration(unittest.TestCase):
    """Integration tests for LLM Provider"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
    
    @unittest.skipUnless(os.getenv("GOOGLE_API_KEY"), "Requires GOOGLE_API_KEY")
    def test_gemini_integration(self):
        """Test actual Gemini API integration (requires API key)"""
        provider = LLMProvider(provider="gemini")
        
        # Test connection
        self.assertTrue(provider.test_connection())
        
        # Test chat
        response = provider.chat(self.test_messages, temperature=0.1, max_tokens=50)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    @unittest.skipUnless(os.getenv("TOGETHER_API_KEY"), "Requires TOGETHER_API_KEY")
    def test_together_integration(self):
        """Test actual Together.ai API integration (requires API key)"""
        provider = LLMProvider(provider="together")
        
        # Test connection
        self.assertTrue(provider.test_connection())
        
        # Test chat
        response = provider.chat(self.test_messages, temperature=0.1, max_tokens=50)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

if __name__ == "__main__":
    unittest.main()
