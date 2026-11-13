import os

import httpx


class LLMUnavailable(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, base_url=None, model=None, timeout=None):
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:1.5b")
        self.timeout = timeout or float(os.getenv("LLM_TIMEOUT", "120"))

    def is_reachable(self):
        try:
            return httpx.get(f"{self.base_url}/api/tags", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def chat(self, system, user):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 300},
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        except httpx.TimeoutException as e:
            raise LLMUnavailable(f"LLM timed out after {self.timeout}s") from e
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"cannot reach Ollama at {self.base_url}: {e}") from e
        if r.status_code != 200:
            raise LLMUnavailable(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
        try:
            content = r.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as e:
            raise LLMUnavailable("malformed response from Ollama") from e
        if not content or not content.strip():
            raise LLMUnavailable("LLM returned an empty answer")
        return content.strip()