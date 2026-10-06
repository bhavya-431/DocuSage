import json
from typing import AsyncGenerator
import httpx
from app.core.config import Settings, get_settings
from app.providers.base import LLMProvider


class GeminiLLMProvider:
    """Google Gemini LLM provider supporting Flash/Pro models and streaming."""

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-1.5-flash"):
        settings = get_settings()
        self._api_key = api_key or settings.GEMINI_API_KEY
        self._model_name = model_name

    async def validate_credentials(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, "GEMINI_API_KEY is not configured"
        if len(self._api_key.strip()) < 10:
            return False, "GEMINI_API_KEY appears invalid (too short)"
        return True, f"Gemini LLM configured for model {self._model_name}"

    async def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent?key={self._api_key}"
        payload: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(part.get("text", "") for part in parts)

    async def stream_response(
        self, prompt: str, system_prompt: str = ""
    ) -> AsyncGenerator[str, None]:
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:streamGenerateContent?alt=sse&key={self._api_key}"
        payload: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            parsed = json.loads(data_str)
                            candidates = parsed.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for part in parts:
                                    text = part.get("text", "")
                                    if text:
                                        yield text
                        except json.JSONDecodeError:
                            continue


class GroqLLMProvider:
    """Groq LLM provider supporting Llama models with high inference speed."""

    def __init__(self, api_key: str | None = None, model_name: str = "llama-3.3-70b-versatile"):
        settings = get_settings()
        self._api_key = api_key or settings.GROQ_API_KEY
        self._model_name = model_name

    async def validate_credentials(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, "GROQ_API_KEY is not configured"
        if not self._api_key.startswith("gsk_"):
            return False, "GROQ_API_KEY format appears invalid (should start with gsk_)"
        return True, f"Groq LLM configured for model {self._model_name}"

    async def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        if not self._api_key:
            raise ValueError("GROQ_API_KEY is not configured")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": 0.1,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def stream_response(
        self, prompt: str, system_prompt: str = ""
    ) -> AsyncGenerator[str, None]:
        if not self._api_key:
            raise ValueError("GROQ_API_KEY is not configured")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": 0.1,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_str)
                            delta = parsed["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue


class OpenRouterLLMProvider:
    """OpenRouter provider supporting unified access to models (including :free tiers)."""

    def __init__(self, api_key: str | None = None, model_name: str = "meta-llama/llama-3.2-3b-instruct:free"):
        settings = get_settings()
        self._api_key = api_key or settings.OPENROUTER_API_KEY
        self._model_name = model_name

    async def validate_credentials(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, "OPENROUTER_API_KEY is not configured"
        return True, f"OpenRouter LLM configured for model {self._model_name}"

    async def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/docusage/docusage",
            "X-Title": "DocuSage",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": 0.1,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def stream_response(
        self, prompt: str, system_prompt: str = ""
    ) -> AsyncGenerator[str, None]:
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/docusage/docusage",
            "X-Title": "DocuSage",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": 0.1,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_str)
                            delta = parsed["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Factory to retrieve configured LLM provider instance."""
    settings = settings or get_settings()
    provider_name = settings.LLM_PROVIDER.lower()

    if provider_name == "groq":
        return GroqLLMProvider(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL or "llama-3.3-70b-versatile",
        )
    if provider_name == "openrouter":
        return OpenRouterLLMProvider(
            api_key=settings.OPENROUTER_API_KEY,
            model_name=settings.LLM_MODEL or "meta-llama/llama-3.2-3b-instruct:free",
        )
    # Default to Gemini
    return GeminiLLMProvider(
        api_key=settings.GEMINI_API_KEY,
        model_name=settings.LLM_MODEL or "gemini-1.5-flash",
    )
