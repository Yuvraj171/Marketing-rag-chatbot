from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from google import genai


DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


class GeminiClientError(Exception):
    pass


@dataclass
class GeminiResponse:
    text: str
    model: str


class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        load_dotenv()

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

        if not self.api_key:
            raise GeminiClientError(
                "Missing GEMINI_API_KEY. Add it to your .env file before running."
            )

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as exc:
            raise GeminiClientError(f"Failed to initialize Gemini client: {exc}") from exc

    def generate_text(self, prompt: str) -> GeminiResponse:
        if not prompt or not prompt.strip():
            raise GeminiClientError("Prompt is empty. Cannot send empty prompt to Gemini.")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            text = getattr(response, "text", None)

            if not text or not text.strip():
                raise GeminiClientError(
                    "Gemini returned an empty response. This may be a model/API issue."
                )

            return GeminiResponse(text=text.strip(), model=self.model_name)

        except GeminiClientError:
            raise
        except Exception as exc:
            raise GeminiClientError(f"Gemini generation failed: {exc}") from exc