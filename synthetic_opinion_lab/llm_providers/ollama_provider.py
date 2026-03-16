import requests
import json
from typing import Dict, Any, Optional

from .base_provider import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """LLM provider for Ollama local models."""

    def __init__(self, model_name: str = "llama3", host: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)

        self.host = host.rstrip('/')
        self.api_url = f"{self.host}/api"

        # Default parameters
        self.default_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1
        }
        self.default_params.update(kwargs)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Ollama API."""

        # Merge parameters
        params = self.default_params.copy()
        params.update(kwargs)

        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 0.9),
                "top_k": params.get("top_k", 40),
                "repeat_penalty": params.get("repeat_penalty", 1.1),
                "num_predict": params.get("max_tokens", 512)
            }
        }

        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=60  # Longer timeout for local models
            )
            response.raise_for_status()

            data = response.json()

            if "response" in data:
                content = data["response"]

                # Extract usage/timing info if available
                usage = {}
                if "total_duration" in data:
                    usage["total_duration_ms"] = data["total_duration"] / 1_000_000  # Convert nanoseconds to ms
                if "load_duration" in data:
                    usage["load_duration_ms"] = data["load_duration"] / 1_000_000
                if "prompt_eval_count" in data:
                    usage["prompt_tokens"] = data["prompt_eval_count"]
                if "eval_count" in data:
                    usage["completion_tokens"] = data["eval_count"]

                return LLMResponse(
                    content=content.strip(),
                    usage=usage,
                    metadata={"model": self.model_name, "provider": "ollama"}
                )
            else:
                raise ValueError("No response returned from Ollama API")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing Ollama API response: {e}")

    def is_available(self) -> bool:
        """Check if Ollama is available and the model is loaded."""
        try:
            # First check if Ollama is running
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            # Check if our specific model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            return any(self.model_name in name for name in model_names)

        except:
            return False

    def list_available_models(self) -> list:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()

            data = response.json()
            if "models" in data:
                return [model["name"] for model in data["models"]]
            return []
        except:
            return []

    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """
        Pull/download a model to Ollama.

        Args:
            model_name: Name of model to pull. If None, uses self.model_name

        Returns:
            True if successful, False otherwise
        """
        model_to_pull = model_name or self.model_name

        try:
            payload = {"name": model_to_pull}
            response = requests.post(
                f"{self.api_url}/pull",
                json=payload,
                timeout=300  # 5 minutes for model download
            )
            return response.status_code == 200
        except:
            return False

    def ensure_model_available(self) -> bool:
        """
        Ensure the model is available, pulling it if necessary.

        Returns:
            True if model is available or was successfully pulled
        """
        if self.is_available():
            return True

        # Try to pull the model
        print(f"Model {self.model_name} not found. Attempting to pull...")
        return self.pull_model()

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model."""
        base_info = super().get_model_info()

        try:
            response = requests.post(
                f"{self.api_url}/show",
                json={"name": self.model_name},
                timeout=10
            )

            if response.status_code == 200:
                model_data = response.json()
                base_info.update({
                    "model_info": model_data.get("modelinfo", {}),
                    "parameters": model_data.get("parameters", {}),
                    "template": model_data.get("template", "")
                })
        except:
            pass  # Fail silently if model info can't be retrieved

        return base_info