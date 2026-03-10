import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from .experiment_runner import ExperimentRunner
from ..survey.ingestion import SurveyIngester
from ..evaluation.evaluation_pipeline import EvaluationPipeline


class SurveyReplicationPipeline:
    """Specialized pipeline for replicating existing survey results."""

    def __init__(self, output_dir: str = "./experiments/replication"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def replicate_survey(self,
                        survey_file: str,
                        target_sample_size: Optional[int] = None,
                        replication_strategy: str = "adaptive",
                        quality_threshold: float = 0.8,
                        max_iterations: int = 5) -> Dict[str, Any]:
        """
        Automatically replicate a survey using the best-performing generator.

        Args:
            survey_file: Path to original survey data
            target_sample_size: Target synthetic sample size (if None, match original)
            replication_strategy: "adaptive", "irt_only", "llm_only", "agent_only"
            quality_threshold: Minimum quality score to accept
            max_iterations: Maximum iterations to try improving quality

        Returns:
            Replication results including best synthetic dataset
        """
        # Ingest original survey to determine target sample size
        real_data, survey_schema = SurveyIngester.ingest(survey_file)

        if target_sample_size is None:
            target_sample_size = len(real_data)

        replication_log = []
        replication_log.append({
            "step": "initialization",
            "original_sample_size": len(real_data),
            "target_sample_size": target_sample_size,
            "n_questions": len(survey_schema.questions)
        })

        if replication_strategy == "adaptive":
            return self._adaptive_replication(
                survey_file, target_sample_size, quality_threshold,
                max_iterations, replication_log
            )
        else:
            return self._single_generator_replication(
                survey_file, target_sample_size, replication_strategy, replication_log
            )

    def _adaptive_replication(self, survey_file: str, target_sample_size: int,
                             quality_threshold: float, max_iterations: int,
                             replication_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use adaptive strategy to find best replication approach."""

        # Step 1: Quick comparison of all generators with smaller sample
        quick_sample_size = min(500, target_sample_size)
        replication_log.append({"step": "quick_comparison", "sample_size": quick_sample_size})

        comparison_results = self._compare_generators_quick(survey_file, quick_sample_size)

        # Identify best performing generator
        best_generator = self._select_best_generator(comparison_results)
        replication_log.append({"step": "best_generator_selected", "generator": best_generator})

        # Step 2: Full-scale replication with best generator
        replication_log.append({"step": "full_scale_replication", "sample_size": target_sample_size})

        full_results = self._run_full_replication(
            survey_file, target_sample_size, best_generator
        )

        # Step 3: Quality improvement iterations if needed
        iteration = 0
        current_quality = self._extract_quality_score(full_results)

        while current_quality < quality_threshold and iteration < max_iterations:
            iteration += 1
            replication_log.append({
                "step": "quality_improvement",
                "iteration": iteration,
                "current_quality": current_quality
            })

            # Try parameter tuning or alternative approaches
            improved_results = self._improve_replication_quality(
                survey_file, target_sample_size, best_generator, full_results
            )

            improved_quality = self._extract_quality_score(improved_results)

            if improved_quality > current_quality:
                full_results = improved_results
                current_quality = improved_quality

        # Compile final results
        final_results = {
            "replication_strategy": "adaptive",
            "best_generator": best_generator,
            "final_quality_score": current_quality,
            "target_sample_size": target_sample_size,
            "iterations_completed": iteration,
            "quality_threshold_met": current_quality >= quality_threshold,
            "replication_log": replication_log,
            "comparison_results": comparison_results,
            "final_experiment_results": full_results
        }

        self._save_replication_results(final_results, survey_file)
        return final_results

    def _compare_generators_quick(self, survey_file: str, sample_size: int) -> Dict[str, Any]:
        """Quick comparison of generators with reduced sample size."""
        experiment_name = f"quick_comparison_{Path(survey_file).stem}"
        runner = ExperimentRunner(experiment_name, str(self.output_dir))

        generator_configs = [
            {"name": "irt", "type": "irt"},
            {"name": "llm", "type": "llm", "config": {"provider": "ollama", "model": "llama3"}},
            {"name": "agent", "type": "agent_simulation", "config": {"n_timesteps": 25}}
        ]

        return runner.compare_generators(
            survey_file=survey_file,
            generator_configs=generator_configs,
            n_personas=sample_size
        )

    def _select_best_generator(self, comparison_results: Dict[str, Any]) -> str:
        """Select the best performing generator from comparison results."""
        if "comparison_summary" not in comparison_results:
            return "irt"  # Default fallback

        summary = comparison_results["comparison_summary"]

        if "ranking" in summary and summary["ranking"]:
            return summary["ranking"][0]

        # Fallback: choose based on quality scores
        quality_scores = summary.get("quality_scores", {})
        if quality_scores:
            best_gen = max(quality_scores.items(),
                          key=lambda x: x[1].get("overall_score", 0))
            return best_gen[0]

        return "irt"  # Ultimate fallback

    def _run_full_replication(self, survey_file: str, target_sample_size: int,
                             generator_type: str) -> Dict[str, Any]:
        """Run full replication with specified generator."""
        experiment_name = f"full_replication_{Path(survey_file).stem}_{generator_type}"
        runner = ExperimentRunner(experiment_name, str(self.output_dir))

        # Configure generator based on type
        generator_config = self._get_optimized_config(generator_type)

        return runner.run_experiment(
            survey_file=survey_file,
            generator_type=generator_type,
            n_personas=target_sample_size,
            generator_config=generator_config
        )

    def _get_optimized_config(self, generator_type: str) -> Dict[str, Any]:
        """Get optimized configuration for each generator type."""
        if generator_type == "irt":
            return {}  # Use defaults
        elif generator_type == "llm":
            return {
                "provider": "ollama",
                "model": "llama3",
                "template": "personalized",
                "batch_size": 5,
                "delay": 0.1
            }
        elif generator_type == "agent_simulation" or generator_type == "agent":
            return {
                "n_timesteps": 50,
                "network_type": "demographic",
                "network_params": {"connection_probability": 0.15, "homophily_strength": 0.8},
                "media_influence_strength": 0.1
            }
        else:
            return {}

    def _improve_replication_quality(self, survey_file: str, target_sample_size: int,
                                    generator_type: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to improve replication quality through parameter tuning."""
        # This is a simplified approach - in practice, you might implement
        # more sophisticated hyperparameter optimization

        experiment_name = f"improved_replication_{Path(survey_file).stem}_{generator_type}"
        runner = ExperimentRunner(experiment_name, str(self.output_dir))

        # Try alternative configurations
        if generator_type == "llm":
            improved_config = {
                "provider": "ollama",
                "model": "llama3",
                "template": "adaptive",  # Try adaptive template
                "batch_size": 3,  # Smaller batches for more variation
                "delay": 0.2
            }
        elif generator_type == "agent_simulation" or generator_type == "agent":
            improved_config = {
                "n_timesteps": 75,  # More timesteps
                "network_type": "small_world",
                "network_params": {"k": 8, "p": 0.15},  # Denser network
                "media_influence_strength": 0.15
            }
        else:
            # For IRT, we could try different item parameter distributions
            improved_config = {}

        return runner.run_experiment(
            survey_file=survey_file,
            generator_type=generator_type,
            n_personas=target_sample_size,
            generator_config=improved_config
        )

    def _extract_quality_score(self, experiment_results: Dict[str, Any]) -> float:
        """Extract overall quality score from experiment results."""
        try:
            eval_results = experiment_results.get("evaluation_results", {})
            overall_quality = eval_results.get("overall_quality", {})
            summary = overall_quality.get("summary", {})
            return summary.get("overall_score", 0.0)
        except:
            return 0.0

    def _single_generator_replication(self, survey_file: str, target_sample_size: int,
                                     generator_type: str, replication_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Replicate survey using a single specified generator."""
        # Remove strategy suffix if present
        if generator_type.endswith("_only"):
            generator_type = generator_type[:-5]

        replication_log.append({"step": "single_generator_replication", "generator": generator_type})

        results = self._run_full_replication(survey_file, target_sample_size, generator_type)
        quality_score = self._extract_quality_score(results)

        final_results = {
            "replication_strategy": f"{generator_type}_only",
            "generator": generator_type,
            "final_quality_score": quality_score,
            "target_sample_size": target_sample_size,
            "replication_log": replication_log,
            "experiment_results": results
        }

        self._save_replication_results(final_results, survey_file)
        return final_results

    def _save_replication_results(self, results: Dict[str, Any], survey_file: str):
        """Save replication results to output directory."""
        survey_name = Path(survey_file).stem
        output_file = self.output_dir / f"replication_{survey_name}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def batch_replicate_surveys(self, survey_files: List[str],
                               replication_configs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Replicate multiple surveys in batch.

        Args:
            survey_files: List of survey file paths
            replication_configs: Optional list of configs for each survey

        Returns:
            Batch replication results
        """
        batch_results = {
            "surveys_processed": len(survey_files),
            "results": {},
            "summary": {}
        }

        if replication_configs is None:
            replication_configs = [{}] * len(survey_files)

        for i, survey_file in enumerate(survey_files):
            survey_name = Path(survey_file).stem
            config = replication_configs[i] if i < len(replication_configs) else {}

            try:
                print(f"Replicating survey {i+1}/{len(survey_files)}: {survey_name}")

                replication_result = self.replicate_survey(
                    survey_file=survey_file,
                    target_sample_size=config.get("target_sample_size"),
                    replication_strategy=config.get("replication_strategy", "adaptive"),
                    quality_threshold=config.get("quality_threshold", 0.8),
                    max_iterations=config.get("max_iterations", 5)
                )

                batch_results["results"][survey_name] = replication_result

            except Exception as e:
                print(f"Failed to replicate {survey_name}: {e}")
                batch_results["results"][survey_name] = {"error": str(e)}

        # Create batch summary
        batch_results["summary"] = self._create_batch_summary(batch_results["results"])

        # Save batch results
        batch_output_file = self.output_dir / "batch_replication_results.json"
        with open(batch_output_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)

        return batch_results

    def _create_batch_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of batch replication results."""
        summary = {
            "successful_replications": 0,
            "failed_replications": 0,
            "average_quality_score": 0.0,
            "quality_scores": [],
            "generator_usage": {}
        }

        for survey_name, result in results.items():
            if "error" in result:
                summary["failed_replications"] += 1
            else:
                summary["successful_replications"] += 1

                quality = result.get("final_quality_score", 0.0)
                summary["quality_scores"].append(quality)

                generator = result.get("best_generator") or result.get("generator", "unknown")
                summary["generator_usage"][generator] = summary["generator_usage"].get(generator, 0) + 1

        if summary["quality_scores"]:
            summary["average_quality_score"] = float(np.mean(summary["quality_scores"]))

        return summary