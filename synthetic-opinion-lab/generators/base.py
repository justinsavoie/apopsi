from abc import ABC, abstractmethod
from typing import List
import pandas as pd

from ..personas.persona import Persona
from ..survey.schema import SurveySchema


class ResponseGenerator(ABC):
    """Base class for all response generators."""

    @abstractmethod
    def generate(self, personas: List[Persona], survey: SurveySchema) -> pd.DataFrame:
        """
        Generate survey responses for given personas.

        Args:
            personas: List of persona objects
            survey: Survey schema defining questions

        Returns:
            DataFrame with rows=respondents, columns=survey questions
        """
        pass