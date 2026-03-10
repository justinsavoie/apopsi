import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union

from ..base import ResponseGenerator
from ...personas.persona import Persona
from ...survey.schema import SurveySchema, QuestionType
from .item_models import ItemParameters, TwoPLModel, GradedResponseModel, ItemParameterGenerator


class IRTResponseGenerator(ResponseGenerator):
    """Generates survey responses using Item Response Theory models."""

    def __init__(self, item_parameters: Optional[Dict[str, ItemParameters]] = None):
        """
        Initialize IRT generator.

        Args:
            item_parameters: Dict mapping question IDs to ItemParameters.
                           If None, parameters will be auto-generated.
        """
        self.item_parameters = item_parameters or {}
        self.models = {}  # Will store IRT models for each question

    def generate(self, personas: List[Persona], survey: SurveySchema) -> pd.DataFrame:
        """
        Generate survey responses using IRT models.

        Args:
            personas: List of persona objects
            survey: Survey schema defining questions

        Returns:
            DataFrame with rows=respondents, columns=survey questions
        """
        # Ensure we have item parameters for all questions
        self._ensure_item_parameters(survey)

        # Create IRT models for each question
        self._create_irt_models(survey)

        # Initialize response dataframe
        n_personas = len(personas)
        responses = pd.DataFrame(index=range(n_personas))

        # Add persona identifiers
        responses['respondent_id'] = [p.id for p in personas]

        # Generate responses for each question
        for question in survey.questions:
            question_id = question.id
            model = self.models[question_id]

            # Extract relevant trait for this question
            # For now, use ideology as primary trait, but this could be more sophisticated
            theta_values = [self._get_theta_for_question(p, question) for p in personas]

            # Generate responses
            question_responses = []
            for theta in theta_values:
                response = model.sample_response(theta)
                question_responses.append(response)

            responses[question_id] = question_responses

        return responses

    def _ensure_item_parameters(self, survey: SurveySchema) -> None:
        """Ensure we have item parameters for all questions in survey."""
        for question in survey.questions:
            if question.id not in self.item_parameters:
                # Auto-generate parameters based on question type
                if question.type == QuestionType.BINARY:
                    params = ItemParameterGenerator.generate_binary_items(1)[0]
                elif question.type == QuestionType.LIKERT:
                    n_options = len(question.options) if question.options else 5
                    params = ItemParameterGenerator.generate_likert_items(1, n_options)[0]
                elif question.type == QuestionType.CATEGORICAL:
                    # Treat as ordered categorical for now
                    n_options = len(question.options) if question.options else 3
                    params = ItemParameterGenerator.generate_likert_items(1, n_options)[0]
                else:
                    # Default to binary
                    params = ItemParameterGenerator.generate_binary_items(1)[0]

                self.item_parameters[question.id] = params

    def _create_irt_models(self, survey: SurveySchema) -> None:
        """Create IRT model objects for each question."""
        for question in survey.questions:
            question_id = question.id
            params = self.item_parameters[question_id]

            if question.type == QuestionType.BINARY:
                self.models[question_id] = TwoPLModel(params)
            elif question.type in [QuestionType.LIKERT, QuestionType.CATEGORICAL]:
                self.models[question_id] = GradedResponseModel(params)
            else:
                # Default to binary
                self.models[question_id] = TwoPLModel(params)

    def _get_theta_for_question(self, persona: Persona, question) -> float:
        """
        Extract the relevant latent trait for a given question.

        This is a simplified mapping. In practice, you might have more
        sophisticated ways to map questions to traits.
        """
        # For now, use a weighted combination of traits based on question content
        traits = persona.traits

        # Simple heuristic: use ideology for most political questions
        # In practice, you'd want a more sophisticated mapping
        question_text_lower = question.text.lower()

        if any(word in question_text_lower for word in ['government', 'policy', 'tax']):
            # Government-related questions: combine ideology and trust_in_government
            return 0.7 * traits.ideology + 0.3 * traits.trust_in_government

        elif any(word in question_text_lower for word in ['immigration', 'trade', 'global']):
            # Globalization questions: use cosmopolitanism
            return traits.cosmopolitanism

        elif any(word in question_text_lower for word in ['authority', 'law', 'order']):
            # Authority questions: use authoritarianism
            return traits.authoritarianism

        elif any(word in question_text_lower for word in ['elite', 'establishment']):
            # Populism questions
            return traits.populism

        else:
            # Default: use ideology
            return traits.ideology

    def set_item_parameters(self, question_id: str, parameters: ItemParameters) -> None:
        """Set item parameters for a specific question."""
        self.item_parameters[question_id] = parameters

    def get_item_parameters(self) -> Dict[str, ItemParameters]:
        """Get all item parameters."""
        return self.item_parameters.copy()

    def calibrate_from_data(self, real_data: pd.DataFrame, survey: SurveySchema) -> None:
        """
        Calibrate item parameters from real survey data.

        This is a placeholder for future IRT calibration functionality.
        In practice, you would use packages like mirt (R) or similar.
        """
        # TODO: Implement IRT calibration
        # This would involve estimating item parameters from real response data
        # For now, we'll just generate reasonable defaults
        self._ensure_item_parameters(survey)

    def calculate_information(self, theta_range: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Calculate test information function across theta range.

        Returns information curves for each item and total test information.
        """
        if theta_range is None:
            theta_range = np.linspace(-3, 3, 100)

        information = {}

        for question_id, model in self.models.items():
            if isinstance(model, TwoPLModel):
                # Information for 2PL model
                alpha = model.item_params.discrimination
                beta = model.item_params.difficulty

                info_values = []
                for theta in theta_range:
                    p = model.probability(theta, 1)
                    info = alpha**2 * p * (1 - p)
                    info_values.append(info)

                information[question_id] = np.array(info_values)

            elif isinstance(model, GradedResponseModel):
                # Information for graded response model (simplified)
                alpha = model.item_params.discrimination

                info_values = []
                for theta in theta_range:
                    # Calculate information using category probabilities
                    total_info = 0
                    n_categories = len(model.item_params.thresholds) + 1

                    for k in range(n_categories):
                        p_k = model.probability(theta, k)
                        if p_k > 0:
                            # Approximate information contribution
                            total_info += alpha**2 * p_k * (1 - p_k)

                    info_values.append(total_info)

                information[question_id] = np.array(info_values)

        # Calculate total test information
        total_info = np.sum([info for info in information.values()], axis=0)
        information['total_test'] = total_info

        return information