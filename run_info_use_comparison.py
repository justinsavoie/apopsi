#!/usr/bin/env python3
"""
Run compare_all_generators on TIDES W9 INFO_USE_* questions using pre-ingested data.
LLM generator uses qwen3.5:9b via Ollama.
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'synthetic-opinion-lab'))

import pandas as pd

from synthetic_opinion_lab.survey.schema import SurveySchema, Question, QuestionType
from synthetic_opinion_lab.personas.persona_generator import PersonaGenerator, DemographicDistribution
from synthetic_opinion_lab.personas.trait_models import TraitModel
from synthetic_opinion_lab.generators.irt.irt_generator import IRTResponseGenerator
from synthetic_opinion_lab.generators.llm.llm_generator import LLMResponseGenerator
from synthetic_opinion_lab.generators.agents.simulation import AgentBasedSimulator
from synthetic_opinion_lab.llm_providers.together_provider import TogetherProvider
from synthetic_opinion_lab.evaluation.evaluation_pipeline import EvaluationPipeline

OUTPUT_DIR = Path("./tides_output/info_use_comparison")
N_PERSONAS = 50
MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"
API_KEY = open("to-key.txt").read().strip()

def load_data():
    real_data = pd.read_parquet("tides_output/clean_data.parquet")

    with open("tides_output/survey_schema.json") as f:
        schema_raw = json.load(f)

    info_qs = [q for q in schema_raw["questions"] if q["id"].startswith("INFO_USE_")]

    questions = [
        Question(
            id=q["id"],
            text=q.get("extended_text") or q["text"],
            type=QuestionType(q["type"]),
            options=q.get("options"),
        )
        for q in info_qs
    ]

    survey_schema = SurveySchema(survey_name="TIDES_W9_INFO_USE", questions=questions)

    keep_cols = ["UniqueID"] + [q.id for q in questions]
    real_subset = real_data[keep_cols].dropna().copy()
    real_subset = real_subset.rename(columns={"UniqueID": "respondent_id"})

    print(f"Real data: {len(real_subset)} rows, {len(questions)} questions")
    return real_subset, survey_schema


def generate_personas():
    demo_dist = DemographicDistribution.canada_census()
    trait_model = TraitModel.default_political_model()
    generator = PersonaGenerator(demo_dist, trait_model)
    personas = generator.generate(N_PERSONAS, random_state=42)
    print(f"Generated {len(personas)} personas")
    return personas


def run_generator(name, generator, personas, survey_schema):
    print(f"\n--- Running {name} generator ---")
    t0 = time.time()
    synthetic = generator.generate(personas, survey_schema)
    elapsed = time.time() - t0
    print(f"{name} done in {elapsed:.1f}s, shape: {synthetic.shape}")
    return synthetic, elapsed


def evaluate(real_data, synthetic_data, survey_schema):
    evaluator = EvaluationPipeline()
    variable_types = {q.id: "categorical" for q in survey_schema.questions}
    results = evaluator.run_full_evaluation(
        real_data, synthetic_data,
        variable_types=variable_types,
    )
    score = (results.get("overall_quality", {})
                    .get("summary", {})
                    .get("overall_score", None))
    return results, score, evaluator


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_data, survey_schema = load_data()
    personas = generate_personas()

    generators = {
        "irt": IRTResponseGenerator(),
        "llm_qwen": LLMResponseGenerator(
            llm_provider=TogetherProvider(MODEL, api_key=API_KEY),
            template_name="standard",
            batch_size=10,
            delay_between_requests=0.1,
        ),
        "agent_simulation": AgentBasedSimulator(
            n_timesteps=25,
            network_type="small_world",
            network_params={},
            media_influence_strength=0.1,
        ),
    }

    comparison = {
        "survey": "TIDES_W9_INFO_USE",
        "n_personas": N_PERSONAS,
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "generators": {},
    }

    for name, generator in generators.items():
        try:
            synthetic, elapsed = run_generator(name, generator, personas, survey_schema)
            eval_results, score, evaluator = evaluate(real_data, synthetic, survey_schema)

            synthetic.to_parquet(OUTPUT_DIR / f"synthetic_{name}.parquet")
            evaluator.generate_html_report(str(OUTPUT_DIR / f"report_{name}.html"))

            comparison["generators"][name] = {
                "overall_score": score,
                "duration_seconds": elapsed,
                "evaluation": eval_results,
            }
            print(f"{name} quality score: {score}")

        except Exception as e:
            print(f"ERROR in {name}: {e}")
            comparison["generators"][name] = {"error": str(e)}

    # Ranking
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
