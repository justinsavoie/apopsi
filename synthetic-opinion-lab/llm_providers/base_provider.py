from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json


@dataclass
class LLMResponse:
    """Standard response format for LLM providers."""
    content: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt string
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse object with generated content
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured correctly."""
        pass

    def generate_structured(self, prompt: str, expected_keys: List[str], **kwargs) -> Dict[str, Any]:
        """
        Generate structured output and attempt to parse as JSON.

        Args:
            prompt: Input prompt string
            expected_keys: List of expected keys in JSON response
            **kwargs: Provider-specific parameters

        Returns:
            Parsed JSON dictionary
        """
        # Add JSON formatting instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON containing keys: {expected_keys}"

        response = self.generate(json_prompt, **kwargs)

        try:
            # Try to parse as JSON
            parsed = json.loads(response.content)

            # Validate expected keys
            if not all(key in parsed for key in expected_keys):
                missing_keys = [key for key in expected_keys if key not in parsed]
                raise ValueError(f"Missing keys in response: {missing_keys}")

            return parsed

        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract JSON from response
            content = response.content.strip()

            # Look for JSON-like patterns
            start_idx = content.find('{')
            end_idx = content.rfind('}')

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                try:
                    json_str = content[start_idx:end_idx+1]
                    parsed = json.loads(json_str)

                    if all(key in parsed for key in expected_keys):
                        return parsed
                except json.JSONDecodeError:
                    pass

            # If all else fails, create a fallback response
            fallback = {}
            for key in expected_keys:
                if key == "response":
                    # Try to extract a number or default
                    words = content.split()
                    for word in words:
                        try:
                            fallback[key] = int(word)
                            break
                        except ValueError:
                            continue
                    else:
                        fallback[key] = 1  # Default response
                else:
                    fallback[key] = content[:100]  # Truncated content

            return fallback

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "provider": self.__class__.__name__,
            "config": self.config
        }