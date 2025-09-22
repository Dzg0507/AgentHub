#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for LLM Provider abstraction
Tests different providers and fallback logic
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.llm_provider import LLMProvider, create_llm_provider, get_llm_response
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_provider(provider_name: str, model: str = None):
    """Test a specific LLM provider"""
    print(f"\n{'='*50}")
    print(f"Testing {provider_name.upper()} Provider")
    print(f"{'='*50}")

    try:
        # Create provider instance
        if model:
            llm = create_llm_provider(provider=provider_name, model=model)
        else:
            llm = create_llm_provider(provider=provider_name)

        print(f"Provider: {llm.provider}")
        print(f"Model: {llm.model}")
        print(f"Base URL: {llm.base_url}")

        # Test connection
        print("\nTesting connection...")
        if llm.test_connection():
            print("✅ Connection test passed")
        else:
            print("❌ Connection test failed")
            return False

        # Test simple chat
        print("\nTesting simple chat...")
        messages = [
            {
                "role": "user",
                "content": "Hello! Please respond with just 'Hello from {provider_name}!'",
            }
        ]

        response = llm.chat(messages, temperature=0.1, max_tokens=50)
        print(f"Response: {response}")
        print("✅ Chat test passed")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_fallback_logic():
    """Test fallback logic between providers"""
    print(f"\n{'='*50}")
    print("Testing Fallback Logic")
    print(f"{'='*50}")

    try:
        # Test with a provider that might fail
        messages = [{"role": "user", "content": "Test fallback"}]
        response = get_llm_response(messages, provider="nonexistent")
        print(f"Fallback response: {response}")
        print("✅ Fallback test passed")
        return True
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("LLM Provider Test Suite")
    print("=" * 50)

    # Test available providers
    providers_to_test = [
        ("gemini", "gemini-1.5-flash"),
        ("together", "togethercomputer/CodeLlama-34b-Instruct"),
        ("openrouter", "openrouter/mistral-7b"),
        ("local", "llama2"),
    ]

    results = {}

    for provider, model in providers_to_test:
        results[provider] = test_provider(provider, model)

    # Test fallback logic
    results["fallback"] = test_fallback_logic()

    # Print summary
    print(f"\n{'='*50}")
    print("Test Results Summary")
    print(f"{'='*50}")

    for provider, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{provider.upper()}: {status}")

    # Check if any provider works
    working_providers = [p for p, s in results.items() if s]
    if working_providers:
        print(f"\n✅ At least one provider is working: {', '.join(working_providers)}")
    else:
        print("\n❌ No providers are working. Check your API keys and configuration.")


if __name__ == "__main__":
    main()
