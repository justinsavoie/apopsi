"""
Synthetic Opinion Lab: A framework for generating synthetic public opinion survey data.

This package provides tools for:
- Ingesting real survey data from multiple formats (SPSS, CSV, Stata)
- Creating synthetic personas with demographics and latent traits
- Generating synthetic responses using IRT, LLM, or agent-based approaches
- Evaluating synthetic data quality against real survey data
"""

__version__ = "0.1.0"

# Main pipeline functions
from .pipelines.experiment_runner import run_simple_experiment, compare_all_generators
from .pipelines.survey_replication_pipeline import SurveyReplicationPipeline

# Core components
from .survey.ingestion import SurveyIngester
from .personas.persona_generator import PersonaGenerator
from .generators.irt.irt_generator import IRTResponseGenerator
from .generators.llm.llm_generator import LLMResponseGenerator
from .generators.agents.simulation import AgentBasedSimulator
from .evaluation.evaluation_pipeline import EvaluationPipeline

# LLM providers
from .llm_providers.together_provider import TogetherProvider
from .llm_providers.ollama_provider import OllamaProvider

__all__ = [
    # Pipeline functions
    "run_simple_experiment",
    "compare_all_generators",
    "SurveyReplicationPipeline",

    # Core components
    "SurveyIngester",
    "PersonaGenerator",
    "IRTResponseGenerator",
    "LLMResponseGenerator",
    "AgentBasedSimulator",
    "EvaluationPipeline",

    # LLM providers
    "TogetherProvider",
    "OllamaProvider"
]