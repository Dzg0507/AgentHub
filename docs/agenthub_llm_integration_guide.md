# AgentHub LLM Integration Guide: Adding LLM Abstraction and Aggregator Support

This guide provides **step-by-step instructions and code snippets** for integrating LLM API abstraction and multi-provider (aggregator) support into your AgentHub system. The goal is to make it easy for an agent (or developer) to implement these improvements based on the recommendations.

---

## 1. LLM API Abstraction Layer

**Goal:**  
Allow AgentHub to use any compatible LLM provider (local, cloud, aggregator) with a single config change.

### a. Create an Abstraction File

Create a new file:  
```python name=llm_provider.py
import os
import requests

class LLMProvider:
    def __init__(self, provider=None, api_key=None, base_url=None, model=None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "together")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_API_BASE")
        self.model = model or os.getenv("LLM_MODEL", "togethercomputer/CodeLlama-34b-Instruct")

    def chat(self, messages, temperature=0.2, max_tokens=2048):
        if self.provider == "together":
            return self._chat_together(messages, temperature, max_tokens)
        elif self.provider == "openrouter":
            return self._chat_openrouter(messages, temperature, max_tokens)
        elif self.provider == "local":
            return self._chat_local(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _chat_together(self, messages, temperature, max_tokens):
        url = self.base_url or "https://api.together.xyz/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _chat_openrouter(self, messages, temperature, max_tokens):
        url = self.base_url or "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _chat_local(self, messages, temperature, max_tokens):
        # Example: local Llama server running OpenAI-compatible API (e.g. via LM Studio/Ollama)
        url = self.base_url or "http://localhost:11434/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
```

### b. Update Agent Creation to Use the Provider

In your agent creation code (e.g., `agent_hub_server.py`):

```python
from llm_provider import LLMProvider

# Initialize provider globally or per session
llm = LLMProvider()

def get_llm_response(messages, temperature=0.2, max_tokens=2048):
    return llm.chat(messages, temperature, max_tokens)

# Example usage in an agent:
system_prompt = "You are a code generation agent. Only use provided tools."
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Create a Python script that prints Hello World."},
]
response = get_llm_response(messages)
```

---

## 2. Configuring Multiple Providers

### a. .env Example

```
# .env
LLM_PROVIDER=together
LLM_API_KEY=your_api_key_here
LLM_API_BASE=https://api.together.xyz/v1/chat/completions
LLM_MODEL=togethercomputer/CodeLlama-34b-Instruct
```
- For OpenRouter: set `LLM_PROVIDER=openrouter`, update `LLM_API_BASE` and `LLM_MODEL`.
- For local Llama (e.g., via Ollama or LM Studio): set `LLM_PROVIDER=local`, `LLM_API_BASE=http://localhost:11434/v1/chat/completions`.

---

## 3. Adapting Prompting for Tool Use and Structure

**Tip:** Open LLMs may need strong, explicit prompts for tool use (file planning, tool calls, no code blocks).

**Prompt template example:**
```python
tool_use_prompt = """
You are an autonomous code agent.
- Respond ONLY with tool calls, e.g., write_file(path, content).
- No code blocks, no explanations, no markdown.
- Use only Python standard library.
- Plan project files first, then create them.
"""
messages = [
    {"role": "system", "content": tool_use_prompt},
    {"role": "user", "content": "Create a CLI calculator."}
]
response = get_llm_response(messages)
```

---

## 4. Adding Provider Selection to UI or Config

**Option 1:**  
Expose provider/model options in the web UI or settings page.

**Option 2:**  
Allow per-project overrides via API payload, e.g.:
```json
{
  "prompt": "Build a web scraper.",
  "llm_provider": "openrouter",
  "llm_model": "openrouter/mistral-7b"
}
```
Update backend to use these values when present.

---

## 5. Hybrid Local/Cloud Logic

**Sample pattern:**
```python
try:
    response = llm.chat(messages)
except Exception as e:
    # If cloud fails, fallback to local
    llm_local = LLMProvider(provider="local", base_url="http://localhost:11434/v1/chat/completions")
    response = llm_local.chat(messages)
```

---

## 6. Logging and Debugging

Add logging for all LLM API requests/responses:
```python
import logging
logging.basicConfig(level=logging.INFO)
logging.info(f"LLM request: {messages}")
logging.info(f"LLM response: {response}")
```

---

## 7. Summary Checklist

- [ ] Add `llm_provider.py` abstraction.
- [ ] Update agent code to use the new abstraction everywhere.
- [ ] Add `.env` config with LLM provider details.
- [ ] Test with Together, OpenRouter, and local Llama APIs.
- [ ] Refine prompt templates for tool calling.
- [ ] (Optional) Expose provider/model selection in UI or API.
- [ ] Add logging for all LLM calls.

---

## 8. References

- [Together.ai API Docs](https://docs.together.ai/docs/chat-completions)
- [OpenRouter API Docs](https://openrouter.ai/docs)
- [Ollama](https://ollama.com/) (local LLM runner)
- [LM Studio](https://lmstudio.ai/) (local OpenAI-compatible LLM server)

---

*Give this guide and the accompanying recommendations to an agent or developer for step-by-step, automatable integration!*