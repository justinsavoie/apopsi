import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import time
from datetime import datetime

from ..survey.ingestion import SurveyIngester
from ..survey.schema import SurveySchema
from ..personas.persona_generator import PersonaGenerator, DemographicDistribution
from ..personas.trait_models import TraitModel
from ..generators.base import ResponseGenerator
from ..generators.irt.irt_generator import IRTResponseGenerator
from ..generators.llm.llm_generator import LLMResponseGenerator
from ..generators.agents.simulation import AgentBasedSimulator
from ..llm_providers.together_provider import TogetherProvider
from ..llm_providers.ollama_provider import OllamaProvider
from ..evaluation.evaluation_pipeline import EvaluationPipeline


class ExperimentRunner:
    """Main pipeline for running synthetic opinion generation experiments."""

    def __init__(self, experiment_name: str, output_dir: str = "./experiments/results"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_log = []
        self.results = {}

    def run_experiment(self,
                      survey_file: str,
                      generator_type: str,
                      n_personas: int = 1000,
                      generator_config: Optional[Dict[str, Any]] = None,
                      persona_config: Optional[Dict[str, Any]] = None,
                      evaluation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a complete synthetic opinion generation experiment.

        Args:
            survey_file: Path to survey data file (SPSS, CSV, Stata)
            generator_type: Type of generator ("irt", "llm", "agent_simulation")
            n_personas: Number of synthetic personas to generate
            generator_config: Configuration for the response generator
            persona_config: Configuration for persona generation
            evaluation_config: Configuration for evaluation

        Returns:
            Dictionary with experiment results
        """
        start_time = time.time()
        self._log("Starting experiment", {"generator_type": generator_type, "n_personas": n_personas})

        try:
            # 1. Ingest Survey Data
            self._log("Ingesting survey data")
            real_data, survey_schema = self._ingest_survey(survey_file)

            # 2. Generate Personas
            self._log("Generating personas")
            personas = self._generate_personas(n_personas, persona_config or {})

            # 3. Generate Synthetic Responses
            self._log("Generating synthetic responses")
            synthetic_data = self._generate_responses(
                personas, survey_schema, generator_type, generator_config or {}
            )

            # 4. Evaluate Results
            self._log("Evaluating synthetic data quality")
            evaluation_results = self._evaluate_synthetic_data(
                real_data, synthetic_data, survey_schema, evaluation_config or {}
            )

            # 5. Save Results
            self._log("Saving experiment results")
            experiment_results = {
                "metadata": {
                    "experiment_name": self.experiment_name,
                    "generator_type": generator_type,
                    "n_personas": n_personas,
                    "survey_file": survey_file,
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "duration_seconds": time.time() - start_time
                },
                "generator_config": generator_config,
                "persona_config": persona_config,
                "evaluation_config": evaluation_config,
                "survey_schema": survey_schema.to_dict(),
                "evaluation_results": evaluation_results,
                "experiment_log": self.experiment_log
            }

            self._save_experiment_results(experiment_results, real_data, synthetic_data)

            self._log("Experiment completed successfully")
            return experiment_results

        except Exception as e:
            self._log("Experiment failed", {"error": str(e)})
            raise

    def _log(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log experiment progress."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "details": details or {}
        }
        self.experiment_log.append(log_entry)
        print(f"[{log_entry['timestamp']}] {message}")

    def _ingest_survey(self, survey_file: str) -> tuple[pd.DataFrame, SurveySchema]:
        """Ingest survey data from file."""
        try:
            real_data, survey_schema = SurveyIngester.ingest(survey_file)
            self._log("Survey ingestion completed", {
                "n_responses": len(real_data),
                "n_questions": len(survey_schema.questions)
            })
            return real_data, survey_schema
        except Exception as e:
            raise RuntimeError(f"Failed to ingest survey data: {e}")

    def _generate_personas(self, n_personas: int, config: Dict[str, Any]) -> List:
        """Generate synthetic personas."""
        try:
            # Use configuration or defaults
            demographic_type = config.get("demographic_distribution", "canada_census")
            trait_model_type = config.get("trait_model", "default_political")

            # Create demographic distribution
            if demographic_type == "canada_census":
                demo_dist = DemographicDistribution.canada_census()
            elif demographic_type == "us_census":
                demo_dist = DemographicDistribution.us_census()
            else:
                raise ValueError(f"Unknown demographic distribution: {demographic_type}")

            # Create trait model
            if trait_model_type == "default_political":
                trait_model = TraitModel.default_political_model()
            elif trait_model_type == "uncorrelated":
                trait_model = TraitModel.uncorrelated_model()
            else:
                raise ValueError(f"Unknown trait model: {trait_model_type}")

            # Generate personas
            generator = PersonaGenerator(demo_dist, trait_model)
            personas = generator.generate(n_personas, config.get("random_state"))

            self._log("Persona generation completed", {"n_personas": len(personas)})
            return personas

        except Exception as e:
            raise RuntimeError(f"Failed to generate personas: {e}")

    def _generate_responses(self, personas: List, survey_schema: SurveySchema,
                           generator_type: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic responses using specified generator."""
        try:
            if generator_type == "irt":
                generator = IRTResponseGenerator(config.get("item_parameters"))

            elif generator_type == "llm":
                # Set up LLM provider
                provider_type = config.get("provider", "ollama")
                model_name = config.get("model", "llama3")

                if provider_type == "ollama":
                    llm_provider = OllamaProvider(model_name)
                elif provider_type == "together":
                    api_key = config.get("api_key")
                    llm_provider = TogetherProvider(model_name, api_key)
                else:
                    raise ValueError(f"Unknown LLM provider: {provider_type}")

                generator = LLMResponseGenerator(
                    llm_provider=llm_provider,
                    template_name=config.get("template", "standard"),
                    batch_size=config.get("batch_size", 10),
                    delay_between_requests=config.get("delay", 0.1)
                )

            elif generator_type == "agent_simulation":
                generator = AgentBasedSimulator(
                    n_timesteps=config.get("n_timesteps", 50),
                    network_type=config.get("network_type", "small_world"),
                    network_params=config.get("network_params", {}),
                    media_influence_strength=config.get("media_influence_strength", 0.1)
                )

            else:
                raise ValueError(f"Unknown generator type: {generator_type}")

            # Generate responses
            synthetic_data = generator.generate(personas, survey_schema)

            self._log("Response generation completed", {
                "n_responses": len(synthetic_data),
                "n_questions": len([col for col in synthetic_data.columns if col != "respondent_id"])
            })

            return synthetic_data

        except Exception as e:
            raise RuntimeError(f"Failed to generate responses: {e}")

    def _evaluate_synthetic_data(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                                survey_schema: SurveySchema, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate synthetic data quality."""
        try:
            evaluator = EvaluationPipeline()

            # Prepare variable types for evaluation
            variable_types = {}
            ordinal_mappings = {}

            for question in survey_schema.questions:
                if question.type.value == "binary":
                    variable_types[question.id] = "categorical"
                elif question.type.value == "likert":
                    variable_types[question.id] = "ordinal"
                    if question.options:
                        ordinal_mappings[question.id] = question.options
                elif question.type.value == "categorical":
                    variable_types[question.id] = "categorical"
                elif question.type.value == "numeric":
                    variable_types[question.id] = "continuous"

            # Run evaluation
            evaluation_results = evaluator.run_full_evaluation(
                real_data,
                synthetic_data,
                variable_types=variable_types,
                ordinal_mappings=ordinal_mappings,
                regression_tests=config.get("regression_tests")
            )

            self._log("Evaluation completed")
            return evaluation_results

        except Exception as e:
            self._log("Evaluation failed", {"error": str(e)})
            return {"error": str(e)}

    def _save_experiment_results(self, experiment_results: Dict[str, Any],
                                real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """Save experiment results to files."""
        experiment_dir = self.output_dir / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        with open(experiment_dir / "experiment_results.json", 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)

        # Save datasets
        real_data.to_parquet(experiment_dir / "real_data.parquet")
        synthetic_data.to_parquet(experiment_dir / "synthetic_data.parquet")

        # Generate evaluation report
        try:
            if "evaluation_results" in experiment_results and "error" not in experiment_results["evaluation_results"]:
                evaluator = EvaluationPipeline()
                evaluator.results = experiment_results["evaluation_results"]
                evaluator.generate_html_report(str(experiment_dir / "evaluation_report.html"))
                evaluator.export_results(str(experiment_dir / "evaluation_results.json"))

        except Exception as e:
            self._log("Failed to generate evaluation report", {"error": str(e)})

        self._log("Results saved", {"output_directory": str(experiment_dir)})

    def compare_generators(self, survey_file: str, generator_configs: List[Dict[str, Any]],
                          n_personas: int = 1000, persona_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare multiple generators on the same survey.

        Args:
            survey_file: Path to survey data
            generator_configs: List of generator configurations, each with 'type' and other params
            n_personas: Number of personas to generate
            persona_config: Persona generation configuration

        Returns:
            Comparison results across all generators
        """
        comparison_results = {
            "survey_file": survey_file,
            "n_personas": n_personas,
            "generators": {},
            "comparison_summary": {}
        }

        # Run experiment for each generator
        for i, gen_config in enumerate(generator_configs):
            gen_name = gen_config.get("name", f"generator_{i+1}")
            gen_type = gen_config["type"]

            self._log(f"Running experiment with {gen_name} ({gen_type})")

            try:
                # Create sub-experiment
                sub_experiment = ExperimentRunner(f"{self.experiment_name}_{gen_name}")
                results = sub_experiment.run_experiment(
                    survey_file=survey_file,
                    generator_type=gen_type,
                    n_personas=n_personas,
                    generator_config=gen_config.get("config", {}),
                    persona_config=persona_config,
                    evaluation_config=gen_config.get("evaluation_config", {})
                )

                comparison_results["generators"][gen_name] = results

            except Exception as e:
                self._log(f"Failed to run {gen_name}", {"error": str(e)})
                comparison_results["generators"][gen_name] = {"error": str(e)}

        # Create comparison summary
        comparison_results["comparison_summary"] = self._create_comparison_summary(
            comparison_results["generators"]
        )

        # Save comparison results
        comparison_dir = self.output_dir / f"{self.experiment_name}_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        with open(comparison_dir / "comparison_results.json", 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)

        self._log("Generator comparison completed")
        return comparison_results

    def _create_comparison_summary(self, generator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary comparing different generators."""
        summary = {"quality_scores": {}, "performance_metrics": {}}

        for gen_name, results in generator_results.items():
            if "error" in results:
                continue

            # Extract quality scores
            eval_results = results.get("evaluation_results", {})
            overall_quality = eval_results.get("overall_quality", {})

            if "summary" in overall_quality:
                summary["quality_scores"][gen_name] = overall_quality["summary"]

            # Extract performance metrics
            metadata = results.get("metadata", {})
            summary["performance_metrics"][gen_name] = {
                "duration_seconds": metadata.get("duration_seconds", 0),
                "n_personas": metadata.get("n_personas", 0)
            }

        # Rank generators by overall score
        if summary["quality_scores"]:
            ranked_generators = sorted(
                summary["quality_scores"].items(),
                key=lambda x: x[1].get("overall_score", 0),
                reverse=True
            )
            summary["ranking"] = [gen_name for gen_name, _ in ranked_generators]

        return summary


# Convenience functions for common use cases

def run_simple_experiment(survey_file: str, generator_type: str, n_personas: int = 1000,
                         output_dir: str = "./experiments/results") -> Dict[str, Any]:
    """
    Run a simple experiment with default configurations.

    Args:
        survey_file: Path to survey data
        generator_type: "irt", "llm", or "agent_simulation"
        n_personas: Number of personas to generate
        output_dir: Output directory

    Returns:
        Experiment results
    """
    experiment_name = f"simple_{generator_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runner = ExperimentRunner(experiment_name, output_dir)

    return runner.run_experiment(
        survey_file=survey_file,
        generator_type=generator_type,
        n_personas=n_personas
    )


def compare_all_generators(survey_file: str, n_personas: int = 1000,
                          output_dir: str = "./experiments/results") -> Dict[str, Any]:
    """
    Compare all available generators on a survey.

    Args:
        survey_file: Path to survey data
        n_personas: Number of personas to generate
        output_dir: Output directory

    Returns:
        Comparison results
    """
    experiment_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runner = ExperimentRunner(experiment_name, output_dir)

    generator_configs = [
        {"name": "irt", "type": "irt", "config": {}},
        {"name": "llm_ollama", "type": "llm", "config": {"provider": "ollama", "model": "llama3"}},
        {"name": "agent_simulation", "type": "agent_simulation", "config": {"n_timesteps": 25}}
    ]

    return runner.compare_generators(
        survey_file=survey_file,
        generator_configs=generator_configs,
        n_personas=n_personas
    )