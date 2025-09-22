#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Provider Abstraction Layer for Agent Control Hub
Supports multiple LLM providers with a unified interface
"""
import os
import requests
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider:
    """Unified LLM provider that supports multiple backends"""
    
    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM provider with configuration
        
        Args:
            provider: LLM provider name (together, openrouter, local, gemini)
            api_key: API key for the provider
            base_url: Base URL for the API
            model: Model name to use
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "gemini")
        
        # Get API key based on provider
        if api_key:
            self.api_key = api_key
        else:
            if self.provider == "gemini":
                self.api_key = os.getenv("GOOGLE_API_KEY")
            elif self.provider == "together":
                self.api_key = os.getenv("TOGETHER_API_KEY")
            elif self.provider == "openrouter":
                self.api_key = os.getenv("OPENROUTER_API_KEY")
            else:
                self.api_key = os.getenv("LLM_API_KEY")
        
        self.base_url = base_url or os.getenv("LLM_API_BASE")
        self.model = model or os.getenv("LLM_MODEL", "gemini-1.5-flash")
        
        # Set default URLs based on provider
        if not self.base_url:
            if self.provider == "together":
                self.base_url = "https://api.together.xyz/v1/chat/completions"
            elif self.provider == "openrouter":
                self.base_url = "https://openrouter.ai/api/v1/chat/completions"
            elif self.provider == "local":
                self.base_url = "http://localhost:11434/v1/chat/completions"
        
        logger.info(f"Initialized LLM provider: {self.provider} with model: {self.model}")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, 
             max_tokens: int = 2048) -> str:
        """
        Generate a chat completion using the configured provider
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response content as string
        """
        try:
            logger.info(f"LLM request to {self.provider}: {len(messages)} messages")
            
            if self.provider == "gemini":
                return self._chat_gemini(messages, temperature, max_tokens)
            elif self.provider == "together":
                return self._chat_together(messages, temperature, max_tokens)
            elif self.provider == "openrouter":
                return self._chat_openrouter(messages, temperature, max_tokens)
            elif self.provider == "local":
                return self._chat_local(messages, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    def _chat_gemini(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Generate response using Google Gemini API"""
        try:
            import google.generativeai as genai
            
            # Configure the API
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            
            # Prepare the conversation
            conversation = []
            for msg in messages:
                if msg['role'] == 'system':
                    conversation.append(f"System: {msg['content']}")
                elif msg['role'] == 'user':
                    conversation.append(f"User: {msg['content']}")
                elif msg['role'] == 'assistant':
                    conversation.append(f"Assistant: {msg['content']}")
            
            prompt = "\n".join(conversation)
            
            # Generate response
            response = model.generate_content(prompt)
            
            logger.info(f"Gemini response generated successfully")
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def _chat_together(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Generate response using Together.ai API"""
        url = self.base_url
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Together.ai response generated successfully")
        return result["choices"][0]["message"]["content"]

    def _chat_openrouter(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Generate response using OpenRouter API"""
        url = self.base_url
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"OpenRouter response generated successfully")
        return result["choices"][0]["message"]["content"]

    def _chat_local(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Generate response using local LLM (e.g., Ollama, LM Studio)"""
        url = self.base_url
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Local LLM response generated successfully")
        return result["choices"][0]["message"]["content"]

    def get_available_models(self) -> List[str]:
        """Get list of available models for the current provider"""
        if self.provider == "gemini":
            return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
        elif self.provider == "together":
            return [
                "togethercomputer/CodeLlama-34b-Instruct",
                "togethercomputer/CodeLlama-13b-Instruct",
                "meta-llama/Llama-2-70b-chat-hf",
                "mistralai/Mistral-7B-Instruct-v0.1"
            ]
        elif self.provider == "openrouter":
            return [
                "openrouter/mistral-7b",
                "openrouter/llama-2-7b",
                "openrouter/codellama-7b",
                "anthropic/claude-3-haiku"
            ]
        elif self.provider == "local":
            return ["llama2", "codellama", "mistral", "custom"]
        else:
            return []

    def test_connection(self) -> bool:
        """Test if the provider is accessible and working"""
        try:
            test_messages = [{"role": "user", "content": "Hello, this is a test."}]
            response = self.chat(test_messages, temperature=0.1, max_tokens=10)
            logger.info(f"Connection test successful for {self.provider}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed for {self.provider}: {e}")
            return False


def create_llm_provider(provider: Optional[str] = None, **kwargs) -> LLMProvider:
    """Factory function to create an LLM provider instance"""
    return LLMProvider(provider=provider, **kwargs)


def get_llm_response(messages: List[Dict[str, str]], temperature: float = 0.2, 
                    max_tokens: int = 2048, provider: Optional[str] = None) -> str:
    """
    Convenience function to get LLM response with intelligent fallback logic
    
    Args:
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        provider: Optional provider override
        
    Returns:
        Generated response content
    """
    # Define fallback chain based on provider
    fallback_chain = []
    
    if provider == "gemini":
        fallback_chain = ["gemini", "together", "openrouter", "local"]
    elif provider == "together":
        fallback_chain = ["together", "openrouter", "gemini", "local"]
    elif provider == "openrouter":
        fallback_chain = ["openrouter", "together", "gemini", "local"]
    elif provider == "local":
        fallback_chain = ["local", "gemini", "together", "openrouter"]
    else:
        # Default fallback chain
        fallback_chain = ["gemini", "together", "openrouter", "local"]
    
    last_error = None
    
    for i, fallback_provider in enumerate(fallback_chain):
        try:
            logger.info(f"Attempting LLM request with {fallback_provider} (attempt {i+1}/{len(fallback_chain)})")
            llm = create_llm_provider(provider=fallback_provider)
            response = llm.chat(messages, temperature, max_tokens)
            logger.info(f"Successfully got response from {fallback_provider}")
            return response
        except Exception as e:
            logger.warning(f"Provider {fallback_provider} failed: {e}")
            last_error = e
            continue
    
    # If all providers failed
    logger.error("All LLM providers failed")
    raise last_error or Exception("All LLM providers failed")


def get_hybrid_llm_response(messages: List[Dict[str, str]], temperature: float = 0.2, 
                           max_tokens: int = 2048, prefer_local: bool = True) -> str:
    """
    Get LLM response with hybrid local/cloud logic
    
    Args:
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        prefer_local: Whether to prefer local LLM first
        
    Returns:
        Generated response content
    """
    if prefer_local:
        # Try local first, then cloud
        fallback_chain = ["local", "gemini", "together", "openrouter"]
    else:
        # Try cloud first, then local
        fallback_chain = ["gemini", "together", "openrouter", "local"]
    
    return get_llm_response(messages, temperature, max_tokens, fallback_chain[0])


# Global instance for backward compatibility
_default_llm = None

def get_default_llm() -> LLMProvider:
    """Get the default LLM provider instance"""
    global _default_llm
    if _default_llm is None:
        _default_llm = create_llm_provider()
    return _default_llm
