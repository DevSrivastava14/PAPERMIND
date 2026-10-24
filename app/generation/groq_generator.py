from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from app.generation.base import BaseGenerator


class GroqGenerator(BaseGenerator):
    """Generator implementation powered by Groq's OpenAI-compatible API."""

    DEFAULT_TIMEOUT: float = 30.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        timeout: Optional[float] = None,
    ) -> None:
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                try:
                    from dotenv import find_dotenv, load_dotenv

                    load_dotenv(find_dotenv(usecwd=True))
                    api_key = os.getenv("GROQ_API_KEY")
                except Exception:
                    pass

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is required for GroqGenerator. Set it in your environment or .env file."
            )

        self._api_key = api_key
        self._model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        self._base_url = base_url or os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        )
        self._temperature = temperature
        self._timeout = timeout if timeout is not None else float(
            os.getenv("GROQ_TIMEOUT", str(self.DEFAULT_TIMEOUT))
        )
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
