#!/usr/bin/env python3
"""
Ingest 2019 Canadian Election Study, generate synthetic data, evaluate.
CPS always-shown variables + PES core module.
Raw data pre-exported via R (arrow) due to encoding issues in the .dta.
'Don't know / Prefer not to answer' recoded to NaN.
"""
import sys, os, json, time
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

OUTPUT_DIR = Path("ces19_output")
N_PERSONAS_LLM   = 100
N_PERSONAS_IRT   = 1000
N_PERSONAS_AGENT = 500
MODEL   = "Qwen/Qwen2.5-7B-Instruct-Turbo"
API_KEY = open("to-key.txt").read().strip()

CORE_VARS = [
    "cps19_demsat",
    "cps19_govt_confusing",
    "cps19_govt_say",
    "cps19_fed_gov_sat",
    "pes19_govtcare",
    "pes19_populism_3",
    "pes19_populism_7",
    "pes19_populism_8",
    "pes19_conf_inst1_1",
    "pes19_conf_inst1_2",
    "pes19_conf_inst1_3",
    "pes19_conf_inst2_1",
    "pes19_fitin",
    "pes19_immigjobs",
    "pes19_famvalues",
    "pes19_equalrights",
    "pes19_newerlife",
    "pes19_inequal",
    "pes19_pidtrad",
    "pes19_votechoice2019",
]

# DK code = highest value for each variable
DK_CODES = {
    "cps19_demsat": 5,
    "cps19_govt_confusing": 5,
    "cps19_govt_say": 5,
    "cps19_fed_gov_sat": 5,
    "pes19_govtcare": 6,
    "pes19_populism_3": 6,
    "pes19_populism_7": 6,
    "pes19_populism_8": 6,
    "pes19_conf_inst1_1": 5,
    "pes19_conf_inst1_2": 5,
    "pes19_conf_inst1_3": 5,
    "pes19_conf_inst2_1": 5,
    "pes19_fitin": 6,
    "pes19_immigjobs": 6,
    "pes19_famvalues": 6,
    "pes19_equalrights": 6,
    "pes19_newerlife": 6,
    "pes19_inequal": 6,
    "pes19_pidtrad": 9,
    "pes19_votechoice2019": 9,
}

FULL_TEXTS = {
    "cps19_demsat":         "On the whole, how satisfied are you with the way democracy works in Canada?",
    "cps19_govt_confusing": "Sometimes, politics and government seem so complicated that a person like me cannot really understand what is going on.",
    "cps19_govt_say":       "People like me don't have any say about what the government does.",
    "cps19_fed_gov_sat":    "How satisfied are you with the performance of the federal government under Justin Trudeau?",
    "pes19_govtcare":       "The government does not care much about what people like me think.",
    "pes19_populism_3":     "Most politicians do not care about the people.",
    "pes19_populism_7":     "The people, and not politicians, should make our most important policy decisions.",
    "pes19_populism_8":     "Most politicians care only about the interests of the rich and powerful.",
    "pes19_conf_inst1_1":   "Please indicate how much confidence you have in the following: The federal government",
    "pes19_conf_inst1_2":   "Please indicate how much confidence you have in the following: Your provincial government",
    "pes19_conf_inst1_3":   "Please indicate how much confidence you have in the following: The media/news",
    "pes19_conf_inst2_1":   "Please indicate how much confidence you have in the following: The courts",
    "pes19_fitin":          "Too many recent immigrants just don't want to fit in to Canadian society.",
    "pes19_immigjobs":      "Immigrants take jobs away from other Canadians.",
    "pes19_famvalues":      "This country would have many fewer problems if there was more emphasis on traditional family values.",
    "pes19_equalrights":    "We have gone too far in pushing equal rights in this country.",
    "pes19_newerlife":      "Newer lifestyles are contributing to the breakdown of our society.",
    "pes19_inequal":        "Is income inequality a big problem in Canada?",
    "pes19_pidtrad":        "In federal politics, do you usually think of yourself as a Liberal, Conservative, NDP, Bloc Québécois, Green, or People's Party supporter?",
    "pes19_votechoice2019": "Which party did you vote for in the October 21, 2019 federal election?",
}

OPTIONS = {
    "cps19_demsat":         ["Very satisfied", "Fairly satisfied", "Not very satisfied", "Not at all satisfied"],
    "cps19_govt_confusing": ["Strongly disagree", "Somewhat disagree", "Somewhat agree", "Strongly agree"],
    "cps19_govt_say":       ["Strongly disagree", "Somewhat disagree", "Somewhat agree", "Strongly agree"],
    "cps19_fed_gov_sat":    ["Very satisfied", "Fairly satisfied", "Not very satisfied", "Not at all satisfied"],
    "pes19_govtcare":       ["Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", "Somewhat agree", "Strongly agree"],
    "pes19_populism_3":     ["Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", "Somewhat agree", "Strongly agree"],
    "pes19_populism_7":     ["Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", "Somewhat agree", "Strongly agree"],
    "pes19_populism_8":     ["Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", "Somewhat agree", "Strongly agree"],
    "pes19_conf_inst1_1":   ["A great deal", "Quite a lot", "Not very much", "None at all"],
    "pes19_conf_inst1_2":   ["A great deal", "Quite a lot", "Not very much", "None at all"],
    "pes19_conf_inst1_3":   ["A great deal", "Quite a lot", "Not very much", "None at all"],
    "pes19_conf_inst2_1":   ["A great deal", "Quite a lot", "Not very much", "None at all"],
    "pes19_fitin":          ["Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", "Somewhat agree", "Strongly agree"],
    "pes19_immigjobs":      ["Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", "Somewhat agree", "Strongly agree"],
    "pes19_famvalues":      ["Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", "Somewhat agree", "Strongly agree"],
    "pes19_equalrights":    ["Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", "Somewhat agree", "Strongly agree"],
    "pes19_newerlife":      ["Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", "Somewhat agree", "Strongly agree"],
    "pes19_inequal":        ["Definitely yes", "Probably yes", "Not sure", "Probably not", "Definitely not"],
    "pes19_pidtrad":        ["Liberal", "Conservative", "NDP", "Bloc Québécois", "Green", "People's Party", "Another party", "None of these"],
    "pes19_votechoice2019": ["Liberal Party", "Conservative Party", "NDP", "Bloc Québécois", "Green Party", "People's Party", "Another party", "Spoiled ballot"],
}

QTYPES = {
    "cps19_demsat":         QuestionType.LIKERT,
    "cps19_govt_confusing": QuestionType.LIKERT,
    "cps19_govt_say":       QuestionType.LIKERT,
    "cps19_fed_gov_sat":    QuestionType.LIKERT,
    "pes19_govtcare":       QuestionType.LIKERT,
    "pes19_populism_3":     QuestionType.LIKERT,
    "pes19_populism_7":     QuestionType.LIKERT,
    "pes19_populism_8":     QuestionType.LIKERT,
    "pes19_conf_inst1_1":   QuestionType.LIKERT,
    "pes19_conf_inst1_2":   QuestionType.LIKERT,
    "pes19_conf_inst1_3":   QuestionType.LIKERT,
    "pes19_conf_inst2_1":   QuestionType.LIKERT,
    "pes19_fitin":          QuestionType.LIKERT,
    "pes19_immigjobs":      QuestionType.LIKERT,
    "pes19_famvalues":      QuestionType.LIKERT,
    "pes19_equalrights":    QuestionType.LIKERT,
    "pes19_newerlife":      QuestionType.LIKERT,
    "pes19_inequal":        QuestionType.LIKERT,
    "pes19_pidtrad":        QuestionType.CATEGORICAL,
    "pes19_votechoice2019": QuestionType.CATEGORICAL,
}


def ingest():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(OUTPUT_DIR / "raw_data.parquet")
    print(f"CES19 raw shape: {df.shape}")

    subset = df[CORE_VARS].copy()

    # Recode DK → NaN
    for col in CORE_VARS:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
        dk = DK_CODES.get(col)
        if dk:
            subset[col] = subset[col].replace(dk, np.nan)

    subset = subset.dropna(how="all")
    cc = subset.dropna()
    print(f"Subset shape: {subset.shape} | Complete cases: {len(cc):,}")
    cc.to_parquet(OUTPUT_DIR / "clean_data.parquet", index=False)

    # Build schema
    questions = []
    for col in CORE_VARS:
        questions.append(Question(
            id=col,
            text=FULL_TEXTS[col],
            type=QTYPES[col],
            options=OPTIONS[col],
        ))

    schema = SurveySchema(survey_name="CES19", questions=questions)
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
    res = ev.run_full_evaluation(
        real[common], syn[common],
        variable_types={k: v for k, v in vt.items() if k in common},
    )
    score = res.get("overall_quality", {}).get("summary", {}).get("overall_score")
    return res, score, ev


def main():
    real, schema = ingest()
    real = real.copy()
    real["respondent_id"] = range(len(real))

    OUT = OUTPUT_DIR / "comparison"
    OUT.mkdir(parents=True, exist_ok=True)

    generators = {
        "irt": (IRTResponseGenerator(), N_PERSONAS_IRT),
        "llm_qwen": (
            LLMResponseGenerator(
                TogetherProvider(MODEL, api_key=API_KEY),
                template_name="standard", batch_size=10, delay_between_requests=0.1,
            ),
            N_PERSONAS_LLM,
        ),
        "agent_simulation": (
            AgentBasedSimulator(
                n_timesteps=25, network_type="small_world",
                network_params={}, media_influence_strength=0.1,
            ),
            N_PERSONAS_AGENT,
        ),
    }

    comparison = {"survey": "CES19", "model": MODEL,
                  "timestamp": datetime.now().isoformat(), "generators": {}}

    for name, (gen, n) in generators.items():
        try:
            personas = generate_personas(n)
            syn, elapsed = run_generator(name, gen, personas, schema)
            syn.to_parquet(OUT / f"synthetic_{name}.parquet")
            res, score, ev = evaluate(real, syn, schema)
            try:
                ev.generate_html_report(str(OUT / f"report_{name}.html"))
            except Exception as e:
                print(f"  HTML report error: {e}")
            comparison["generators"][name] = {"n_personas": n, "overall_score": score, "duration_seconds": elapsed}
            print(f"{name} score: {score}")
        except Exception as e:
            import traceback; traceback.print_exc()
            comparison["generators"][name] = {"error": str(e)}

    scores = {k: v["overall_score"] for k, v in comparison["generators"].items()
              if "overall_score" in v and v["overall_score"]}
    comparison["ranking"] = sorted(scores, key=scores.get, reverse=True)
    print("\n=== Ranking ===")
    for i, n in enumerate(comparison["ranking"], 1):
        print(f"  {i}. {n}: {scores[n]:.3f}")
    with open(OUT / "comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nDone → {OUT}/")


if __name__ == "__main__":
    main()
