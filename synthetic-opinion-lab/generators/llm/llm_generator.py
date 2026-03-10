import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union
import time
from tqdm import tqdm

from ..base import ResponseGenerator
from ...personas.persona import Persona
from ...survey.schema import SurveySchema, Question, QuestionType
from ...llm_providers.base_provider import LLMProvider
from .prompt_templates import TemplateManager, AdaptivePromptStrategy


class LLMResponseGenerator(ResponseGenerator):
    """Generates survey responses using Large Language Models."""

    def __init__(self,
                 llm_provider: LLMProvider,
                 template_name: str = "standard",
                 batch_size: int = 10,
                 delay_between_requests: float = 0.1):
        """
        Initialize LLM response generator.

        Args:
            llm_provider: LLM provider instance
            template_name: Name of prompt template to use
            batch_size: Number of requests to process before pausing
            delay_between_requests: Delay in seconds between API requests
        """
        self.llm_provider = llm_provider
        self.template_manager = TemplateManager()
        self.adaptive_strategy = AdaptivePromptStrategy(self.template_manager)
        self.template_name = template_name
        self.batch_size = batch_size
        self.delay = delay_between_requests

    def generate(self, personas: List[Persona], survey: SurveySchema) -> pd.DataFrame:
        """
        Generate survey responses using LLM.

        Args:
            personas: List of persona objects
            survey: Survey schema defining questions

        Returns:
            DataFrame with rows=respondents, columns=survey questions
        """
        if not self.llm_provider.is_available():
            raise RuntimeError(f"LLM provider {self.llm_provider.__class__.__name__} is not available")

        # Initialize response dataframe
        n_personas = len(personas)
        responses = pd.DataFrame(index=range(n_personas))

        # Add persona identifiers
        responses['respondent_id'] = [p.id for p in personas]

        # Generate responses for each question
        for question in tqdm(survey.questions, desc="Processing questions"):
            question_responses = self._generate_question_responses(personas, question)
            responses[question.id] = question_responses

        return responses

    def _generate_question_responses(self, personas: List[Persona], question: Question) -> List[Any]:
        """Generate responses for a single question across all personas."""
        responses = []

        for i, persona in enumerate(tqdm(personas, desc=f"Question {question.id}", leave=False)):
            try:
                # Generate prompt
                if self.template_name == "adaptive":
                    prompt = self.adaptive_strategy.generate_adaptive_prompt(persona, question)
                else:
                    prompt = self.template_manager.generate_prompt(self.template_name, persona, question)

                # Generate response
                if question.options:
                    # Structured response for multiple choice
                    response_data = self.llm_provider.generate_structured(
                        prompt,
                        expected_keys=["response"]
                    )
                    raw_response = response_data.get("response", 1)

                    # Validate and convert response
                    response = self._validate_categorical_response(raw_response, len(question.options))
                else:
                    # Free text response
                    llm_response = self.llm_provider.generate(prompt)
                    response = llm_response.content.strip()

                responses.append(response)

                # Rate limiting
                if (i + 1) % self.batch_size == 0:
                    time.sleep(self.delay)

            except Exception as e:
                print(f"Error generating response for persona {persona.id}, question {question.id}: {e}")
                # Fallback response
                if question.options:
                    response = np.random.randint(1, len(question.options) + 1)
                else:
                    response = "No response"
                responses.append(response)

        return responses

    def _validate_categorical_response(self, raw_response: Any, num_options: int) -> int:
        """Validate and convert categorical responses to valid option numbers."""
        try:
            # Try to convert to integer
            if isinstance(raw_response, str):
                # Extract number from string
                import re
                numbers = re.findall(r'\d+', raw_response)
                if numbers:
                    response_num = int(numbers[0])
                else:
                    # If no numbers found, default to 1
                    response_num = 1
            else:
                response_num = int(raw_response)

            # Validate range
            if 1 <= response_num <= num_options:
                return response_num
            else:
                # Clamp to valid range
                return max(1, min(num_options, response_num))

        except (ValueError, TypeError):
            # Default to middle option if conversion fails
            return (num_options + 1) // 2

    def generate_single_response(self, persona: Persona, question: Question) -> Any:
        """Generate a response for a single persona-question pair."""
        try:
            if self.template_name == "adaptive":
                prompt = self.adaptive_strategy.generate_adaptive_prompt(persona, question)
            else:
                prompt = self.template_manager.generate_prompt(self.template_name, persona, question)

            if question.options:
                response_data = self.llm_provider.generate_structured(
                    prompt,
                    expected_keys=["response"]
                )
                raw_response = response_data.get("response", 1)
                return self._validate_categorical_response(raw_response, len(question.options))
            else:
                llm_response = self.llm_provider.generate(prompt)
                return llm_response.content.strip()

        except Exception as e:
            print(f"Error generating single response: {e}")
            if question.options:
                return np.random.randint(1, len(question.options) + 1)
            else:
                return "No response"

    def set_template(self, template_name: str) -> None:
        """Set the prompt template to use."""
        if template_name not in self.template_manager.templates and template_name != "adaptive":
            raise ValueError(f"Template '{template_name}' not found")
        self.template_name = template_name

    def add_custom_template(self, name: str, template_str: str) -> None:
        """Add a custom prompt template."""
        from .prompt_templates import PromptTemplate
        template = PromptTemplate(template_str)
        self.template_manager.add_template(name, template)

    def test_prompt(self, persona: Persona, question: Question) -> str:
        """Test prompt generation without calling LLM (for debugging)."""
        if self.template_name == "adaptive":
            return self.adaptive_strategy.generate_adaptive_prompt(persona, question)
        else:
            return self.template_manager.generate_prompt(self.template_name, persona, question)

    def estimate_cost(self, personas: List[Persona], survey: SurveySchema,
                     cost_per_token: float = 0.001) -> Dict[str, float]:
        """
        Estimate the cost of generating responses.

        Args:
            personas: List of personas
            survey: Survey schema
            cost_per_token: Estimated cost per token

        Returns:
            Dictionary with cost estimates
        """
        # Estimate tokens per prompt (very rough estimate)
        sample_persona = personas[0] if personas else None
        sample_question = survey.questions[0] if survey.questions else None

        if sample_persona and sample_question:
            sample_prompt = self.test_prompt(sample_persona, sample_question)
            # Rough estimate: 1 token per 4 characters
            tokens_per_prompt = len(sample_prompt) / 4
            tokens_per_response = 10  # Assume short responses

            total_prompts = len(personas) * len(survey.questions)
            total_input_tokens = total_prompts * tokens_per_prompt
            total_output_tokens = total_prompts * tokens_per_response

            estimated_cost = (total_input_tokens + total_output_tokens) * cost_per_token

            return {
                "total_prompts": total_prompts,
                "estimated_input_tokens": int(total_input_tokens),
                "estimated_output_tokens": int(total_output_tokens),
                "estimated_cost": estimated_cost,
                "cost_per_token": cost_per_token
            }
        else:
            return {"error": "Cannot estimate cost without personas and questions"}

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the generation process."""
        return {
            "llm_provider": self.llm_provider.__class__.__name__,
            "model_name": self.llm_provider.model_name,
            "template_name": self.template_name,
            "batch_size": self.batch_size,
            "delay_between_requests": self.delay
        }