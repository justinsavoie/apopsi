#!/usr/bin/env python3
"""
Generate synthetic DC21 data with IRT, LLM, and Agent generators.
Compare against real n=7,576 respondents on 21 political attitude variables.
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from synthetic_opinion_lab.survey.schema import SurveySchema, Question, QuestionType
from synthetic_opinion_lab.personas.persona_generator import PersonaGenerator, DemographicDistribution
from synthetic_opinion_lab.personas.trait_models import TraitModel
from synthetic_opinion_lab.generators.irt.irt_generator import IRTResponseGenerator
from synthetic_opinion_lab.generators.llm.llm_generator import LLMResponseGenerator
from synthetic_opinion_lab.generators.agents.simulation import AgentBasedSimulator
from synthetic_opinion_lab.llm_providers.together_provider import TogetherProvider
from synthetic_opinion_lab.evaluation.evaluation_pipeline import EvaluationPipeline

OUTPUT_DIR = Path("dc21_output/comparison")
N_PERSONAS_LLM = 100     # LLM is slower / costs money
N_PERSONAS_IRT = 1000    # IRT is fast
N_PERSONAS_AGENT = 500   # Agent is moderate
MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"
API_KEY = open("to-key.txt").read().strip()


def load_data():
    real_data = pd.read_parquet("dc21_output/clean_data.parquet")
    with open("dc21_output/survey_schema.json") as f:
        schema_raw = json.load(f)

    questions = [
        Question(
            id=q["id"],
            text=q["text"],
            type=QuestionType(q["type"]),
            options=q.get("options"),
        )
        for q in schema_raw["questions"]
    ]
    survey_schema = SurveySchema(survey_name="DC21", questions=questions)

    # Add a respondent_id column for the evaluator
    real_data = real_data.copy()
    real_data["respondent_id"] = range(len(real_data))

    print(f"Real data: {len(real_data)} rows, {len(questions)} questions")
    return real_data, survey_schema


def generate_personas(n, seed=42):
    demo_dist = DemographicDistribution.canada_census()
    trait_model = TraitModel.default_political_model()
    generator = PersonaGenerator(demo_dist, trait_model)
    personas = generator.generate(n, random_state=seed)
    print(f"Generated {len(personas)} personas")
    return personas


def run_generator(name, generator, personas, survey_schema):
    print(f"\n--- Running {name} generator ({len(personas)} personas) ---")
    t0 = time.time()
    synthetic = generator.generate(personas, survey_schema)
    elapsed = time.time() - t0
    print(f"{name} done in {elapsed:.1f}s, shape: {synthetic.shape}")
    return synthetic, elapsed


def evaluate(real_data, synthetic_data, survey_schema):
    evaluator = EvaluationPipeline()
    # Use the question IDs from schema to classify types
    variable_types = {}
    for q in survey_schema.questions:
        if q.type.value in ("likert", "categorical", "binary"):
            variable_types[q.id] = "categorical"
        else:
            variable_types[q.id] = "continuous"

    # Align columns: evaluator needs matching columns
    common_cols = [c for c in synthetic_data.columns if c in real_data.columns and c != "respondent_id"]
    real_sub = real_data[common_cols].copy()
    syn_sub = synthetic_data[common_cols].copy()

    results = evaluator.run_full_evaluation(
        real_sub, syn_sub,
        variable_types={k: v for k, v in variable_types.items() if k in common_cols},
    )
    score = (results.get("overall_quality", {})
                    .get("summary", {})
                    .get("overall_score", None))
    return results, score, evaluator


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_data, survey_schema = load_data()

    generators = {
        "irt": (
            IRTResponseGenerator(),
            N_PERSONAS_IRT,
        ),
        "llm_qwen": (
            LLMResponseGenerator(
                llm_provider=TogetherProvider(MODEL, api_key=API_KEY),
                template_name="standard",
                batch_size=10,
                delay_between_requests=0.1,
            ),
            N_PERSONAS_LLM,
        ),
        "agent_simulation": (
            AgentBasedSimulator(
                n_timesteps=25,
                network_type="small_world",
                network_params={},
                media_influence_strength=0.1,
            ),
            N_PERSONAS_AGENT,
        ),
    }

    comparison = {
        "survey": "DC21",
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "generators": {},
    }

    for name, (generator, n_personas) in generators.items():
        try:
            personas = generate_personas(n_personas)
            synthetic, elapsed = run_generator(name, generator, personas, survey_schema)
            synthetic.to_parquet(OUTPUT_DIR / f"synthetic_{name}.parquet")

            eval_results, score, evaluator = evaluate(real_data, synthetic, survey_schema)

            try:
                evaluator.generate_html_report(str(OUTPUT_DIR / f"report_{name}.html"))
            except Exception as e:
                print(f"  HTML report failed: {e}")

            comparison["generators"][name] = {
                "n_personas": n_personas,
                "overall_score": score,
                "duration_seconds": elapsed,
                "evaluation": eval_results,
            }
            print(f"{name} quality score: {score}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR in {name}: {e}")
            comparison["generators"][name] = {"error": str(e)}

    scores = {
        k: v["overall_score"]
        for k, v in comparison["generators"].items()
        if "overall_score" in v and v["overall_score"] is not None
    }
    comparison["ranking"] = sorted(scores, key=scores.get, reverse=True)

    print("\n=== Ranking ===")
    for rank, name in enumerate(comparison["ranking"], 1):
        print(f"  {rank}. {name}: {scores[name]:.3f}")

    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
