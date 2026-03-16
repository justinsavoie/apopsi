#!/usr/bin/env python3
"""
Ingest DC2019, generate synthetic data with IRT/LLM/Agent, evaluate.
DC2019 variables have no prefix. 'Don't know' = 5 (recode to NaN).
"""
import sys, os, json, time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import pyreadstat

from synthetic_opinion_lab.survey.schema import SurveySchema, Question, QuestionType
from synthetic_opinion_lab.personas.persona_generator import PersonaGenerator, DemographicDistribution
from synthetic_opinion_lab.personas.trait_models import TraitModel
from synthetic_opinion_lab.generators.irt.irt_generator import IRTResponseGenerator
from synthetic_opinion_lab.generators.llm.llm_generator import LLMResponseGenerator
from synthetic_opinion_lab.generators.agents.simulation import AgentBasedSimulator
from synthetic_opinion_lab.llm_providers.together_provider import TogetherProvider
from synthetic_opinion_lab.evaluation.evaluation_pipeline import EvaluationPipeline

OUTPUT_DIR = Path("dc19_output")
N_PERSONAS_LLM   = 100
N_PERSONAS_IRT   = 1000
N_PERSONAS_AGENT = 500
MODEL   = "Qwen/Qwen2.5-7B-Instruct-Turbo"
API_KEY = open("to-key.txt").read().strip()

# Core variables available in DC2019
CORE_VARS = [
    "fed_gov_satisfaction",
    "confidence_inst_1", "confidence_inst_2", "confidence_inst_3", "confidence_inst_4",
    "pos_govt_care", "pos_govt_say", "pos_govt_complicated",
    "pos_assimilation", "pos_take_jobs", "pos_look_after", "immigration_level",
    "pos_family_values", "pos_equal_rights", "pos_women_home",
    "inequality_gap", "pos_govt_jobs",
    "party_id", "vote_intention",
]

FULL_TEXTS = {
    "fed_gov_satisfaction":  "How satisfied are you with the performance of the federal government?",
    "confidence_inst_1":     "Please indicate how much confidence you have in the following: The courts and justice system",
    "confidence_inst_2":     "Please indicate how much confidence you have in the following: The federal government",
    "confidence_inst_3":     "Please indicate how much confidence you have in the following: The media/news",
    "confidence_inst_4":     "Please indicate how much confidence you have in the following: Your provincial government",
    "pos_govt_care":         "The government does not care much about what people like me think.",
    "pos_govt_say":          "People like me don't have any say about what the government does.",
    "pos_govt_complicated":  "Sometimes, politics and government seem so complicated that a person like me cannot really understand what is going on.",
    "pos_assimilation":      "Too many recent immigrants just don't want to fit in to Canadian society.",
    "pos_take_jobs":         "Immigrants take jobs away from other Canadians.",
    "pos_look_after":        "We should look after Canadians born in this country first and others second.",
    "immigration_level":     "Do you think Canada should admit more immigrants, fewer immigrants, or about the same number as now?",
    "pos_family_values":     "This country would have many fewer problems if there was more emphasis on traditional family values.",
    "pos_equal_rights":      "We have gone too far in pushing equal rights in this country.",
    "pos_women_home":        "Society would be better off if fewer women worked outside the home.",
    "inequality_gap":        "How much do you think should be done to reduce the gap between the rich and poor in Canada?",
    "pos_govt_jobs":         "The government should leave it entirely to the private sector to create jobs.",
    "party_id":              "In federal politics, do you usually think of yourself as a Liberal, Conservative, NDP, Bloc Québécois, or Green supporter?",
    "vote_intention":        "Which party do you intend to vote for in the federal election?",
}


def ingest():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, meta = pyreadstat.read_dta("data/Democracy Checkup 2019 Waves 1_4 v2.0.dta")
    print(f"DC2019 raw shape: {df.shape}")

    # Save full column inventory
    inventory = [{"id": c, "label": meta.column_names_to_labels.get(c,""),
                  "value_labels": {str(k): v for k,v in meta.variable_value_labels.get(c,{}).items()}}
                 for c in df.columns]
    with open(OUTPUT_DIR / "column_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)

    missing = [v for v in CORE_VARS if v not in df.columns]
    if missing:
        print(f"WARNING - vars not found: {missing}")

    existing = [v for v in CORE_VARS if v in df.columns]
    subset = df[existing].copy()

    # Recode "Don't know" (coded as 5 for 4-opt items, 6 for 5-opt, etc.) and -99 to NaN
    # In DC2019 all selected vars: max valid value is n_categories, DK = n_categories+1
    # Safest: drop the DK label values
    for col in existing:
        vl = meta.variable_value_labels.get(col, {})
        dk_keys = [float(k) for k,v in vl.items() if "don't know" in str(v).lower() or "prefer not" in str(v).lower()]
        for dk in dk_keys:
            subset[col] = subset[col].replace(dk, np.nan)
    subset = subset.replace(-99, np.nan)

    subset = subset.dropna(how="all")
    cc = subset.dropna()
    print(f"Subset shape: {subset.shape} | Complete cases: {len(cc)}")
    cc.to_parquet(OUTPUT_DIR / "clean_data.parquet", index=False)

    # Build schema
    questions = []
    for col in existing:
        vl = meta.variable_value_labels.get(col, {})
        dk_keys = {float(k) for k,v in vl.items() if "don't know" in str(v).lower() or "prefer not" in str(v).lower()}
        options = [v for k,v in sorted(vl.items(), key=lambda x: float(x[0])) if float(k) not in dk_keys] if vl else None

        labels_str = " ".join(str(v).lower() for v in (options or []))
        if options and any(k in labels_str for k in ["strongly","agree","satisfied","confident","very"]):
            qtype = QuestionType.LIKERT
        elif options and len(options) == 2:
            qtype = QuestionType.BINARY
        elif options:
            qtype = QuestionType.CATEGORICAL
        else:
            qtype = QuestionType.NUMERIC

        questions.append(Question(id=col, text=FULL_TEXTS.get(col, meta.column_names_to_labels.get(col,col)),
                                  type=qtype, options=options))

    schema = SurveySchema(survey_name="DC19", questions=questions)
    schema.to_json(str(OUTPUT_DIR / "survey_schema.json"))
    print(f"Schema: {len(questions)} questions")
    for q in questions:
        print(f"  [{q.type.value:12s}] {q.id}: {len(q.options) if q.options else 0} opts")
    return cc, schema


def generate_personas(n, seed=42):
    g = PersonaGenerator(DemographicDistribution.canada_census(), TraitModel.default_political_model())
    return g.generate(n, random_state=seed)


def run_generator(name, generator, personas, schema):
    print(f"\n--- {name} ({len(personas)} personas) ---")
    t0 = time.time()
    syn = generator.generate(personas, schema)
    elapsed = time.time() - t0
    print(f"{name} done in {elapsed:.1f}s, shape: {syn.shape}")
    return syn, elapsed


def evaluate(real, syn, schema):
    ev = EvaluationPipeline()
    vt = {q.id: "categorical" for q in schema.questions}
    common = [c for c in syn.columns if c in real.columns and c != "respondent_id"]
    res = ev.run_full_evaluation(real[common], syn[common], variable_types={k:v for k,v in vt.items() if k in common})
    score = res.get("overall_quality",{}).get("summary",{}).get("overall_score")
    return res, score, ev


def main():
    real, schema = ingest()
    real = real.copy(); real["respondent_id"] = range(len(real))

    OUT = OUTPUT_DIR / "comparison"
    OUT.mkdir(parents=True, exist_ok=True)

    generators = {
        "irt":              (IRTResponseGenerator(), N_PERSONAS_IRT),
        "llm_qwen":         (LLMResponseGenerator(TogetherProvider(MODEL, api_key=API_KEY),
                              template_name="standard", batch_size=10, delay_between_requests=0.1), N_PERSONAS_LLM),
        "agent_simulation": (AgentBasedSimulator(n_timesteps=25, network_type="small_world",
                              network_params={}, media_influence_strength=0.1), N_PERSONAS_AGENT),
    }

    comparison = {"survey":"DC19","model":MODEL,"timestamp":datetime.now().isoformat(),"generators":{}}

    for name, (gen, n) in generators.items():
        try:
            personas = generate_personas(n)
            syn, elapsed = run_generator(name, gen, personas, schema)
            syn.to_parquet(OUT / f"synthetic_{name}.parquet")
            res, score, ev = evaluate(real, syn, schema)
            try: ev.generate_html_report(str(OUT / f"report_{name}.html"))
            except Exception as e: print(f"  HTML report error: {e}")
            comparison["generators"][name] = {"n_personas":n,"overall_score":score,"duration_seconds":elapsed}
            print(f"{name} score: {score}")
        except Exception as e:
            import traceback; traceback.print_exc()
            comparison["generators"][name] = {"error": str(e)}

    scores = {k:v["overall_score"] for k,v in comparison["generators"].items() if "overall_score" in v and v["overall_score"]}
    comparison["ranking"] = sorted(scores, key=scores.get, reverse=True)
    print("\n=== Ranking ===")
    for i, n in enumerate(comparison["ranking"], 1):
        print(f"  {i}. {n}: {scores[n]:.3f}")
    with open(OUT / "comparison_results.json","w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nDone → {OUT}/")


if __name__ == "__main__":
    main()
