import os
import requests
import time
from typing import Dict, Any, Optional

from .base_provider import LLMProvider, LLMResponse


class TogetherProvider(LLMProvider):
    """LLM provider for Together API."""

    def __init__(self, model_name: str = "meta-llama/Llama-3-8b-chat-hf",
                 api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)

        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together API key not provided. Set TOGETHER_API_KEY environment variable or pass api_key parameter.")

        self.base_url = "https://api.together.xyz/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Default parameters
        self.default_params = {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9,
            "repetition_penalty": 1.0
        }
        self.default_params.update(kwargs)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Together API."""

        # Merge parameters
        params = self.default_params.copy()
        params.update(kwargs)

        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 512),
            "top_p": params.get("top_p", 0.9),
            "repetition_penalty": params.get("repetition_penalty", 1.0),
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["text"]

                # Extract usage info if available
                usage = data.get("usage", {})

                return LLMResponse(
                    content=content.strip(),
                    usage=usage,
                    metadata={"model": self.model_name, "provider": "together"}
                )
            else:
                raise ValueError("No choices returned from Together API")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Together API request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing Together API response: {e}")

    def is_available(self) -> bool:
        """Check if Together API is available."""
        try:
            # Test with a simple prompt
            test_response = requests.post(
                f"{self.base_url}/completions",
                headers=self.headers,
                json={
                    "model": self.model_name,
                    "prompt": "Test",
                    "max_tokens": 1,
                    "stream": False
                },
                timeout=10
            )
            return test_response.status_code == 200
        except:
            return False

    def list_available_models(self) -> list:
        """List available models from Together API."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            if "data" in data:
                return [model["id"] for model in data["data"]]
            return []
        except:
            return []