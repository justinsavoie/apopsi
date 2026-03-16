#!/usr/bin/env python3
"""
Ingest 2021 Democracy Checkup (DC21) data and build a focused schema
for synthetic data validation.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import pyreadstat
from pathlib import Path

from synthetic_opinion_lab.survey.ingestion import SurveyIngester
from synthetic_opinion_lab.survey.schema import SurveySchema, Question, QuestionType

DATA_FILE = "data/DC 2021 v1.dta"
OUTPUT_DIR = Path("dc21_output")

# The 25 substantive variables selected for validation
# (democratic attitudes, institutional confidence, populism, efficacy,
#  immigration, social conservatism, economic positions, party ID)
SELECTED_VARS = [
    # Democratic satisfaction
    "dc21_democratic_sat",
    "dc21_fed_gov_satisfa",

    # Institutional confidence (7-item battery)
    "dc21_confidence_inst_1",  # Parliament
    "dc21_confidence_inst_2",  # Federal government
    "dc21_confidence_inst_3",  # Courts / justice system
    "dc21_confidence_inst_4",  # Police
    "dc21_confidence_inst_5",  # Media / news
    "dc21_confidence_inst_6",  # Scientists / experts
    "dc21_confidence_inst_7",  # Corporations / business

    # Populist attitudes
    "dc21_pos_experts",
    "dc21_pos_lose_touch",
    "dc21_pos_pol_lie",
    "dc21_pos_career_pol",

    # Political efficacy
    "dc21_pos_govt_care",
    "dc21_pos_govt_say",
    "dc21_pos_govt_comp",

    # Immigration attitudes
    "dc21_pos_assim",
    "dc21_pos_take_jobs",
    "dc21_pos_look_after",
    "dc21_imm_level",

    # Social conservatism
    "dc21_pos_equal_1",
    "dc21_pos_women_home",
    "dc21_pos_family_val",
    "dc21_newerlife",

    # Economic positions
    "dc21_inequality_gap",
    "dc21_pos_govt_jobs",

    # Party ID / vote (for regression models)
    "dc21_party_id",
    "dc21_vote_choice",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Ingesting DC21...")
    df, meta = pyreadstat.read_dta(DATA_FILE)
    print(f"  Shape: {df.shape}")

    # Print all variable names to inspect what's actually in the file
    print("\nAll column names:")
    for col in df.columns:
        label = meta.column_names_to_labels.get(col, "")
        print(f"  {col}: {label}")

    # Save full column inventory
    inventory = [
        {
            "id": col,
            "label": meta.column_names_to_labels.get(col, ""),
            "value_labels": {str(k): v for k, v in meta.variable_value_labels.get(col, {}).items()},
            "n_unique": int(df[col].nunique()),
            "n_missing": int(df[col].isna().sum()),
        }
        for col in df.columns
    ]
    with open(OUTPUT_DIR / "column_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"\nColumn inventory saved to {OUTPUT_DIR}/column_inventory.json")

    # Check which selected vars actually exist
    existing = [v for v in SELECTED_VARS if v in df.columns]
    missing  = [v for v in SELECTED_VARS if v not in df.columns]
    print(f"\nSelected vars found: {len(existing)}/{len(SELECTED_VARS)}")
    if missing:
        print(f"Missing vars: {missing}")

    # Save raw data as parquet (full dataset for reference)
    df.to_parquet(OUTPUT_DIR / "raw_data.parquet", index=False)

    # Build focused subset
    if not existing:
        print("No selected variables found — inspect column_inventory.json and update SELECTED_VARS.")
        return

    subset = df[existing].copy()
    print(f"\nSubset shape: {subset.shape}")

    # Drop rows with all-NA across selected vars
    subset = subset.dropna(how="all")
    print(f"After dropping all-NA rows: {subset.shape}")

    subset.to_parquet(OUTPUT_DIR / "clean_data.parquet", index=False)

    # Build survey schema for selected vars
    questions = []
    for col in existing:
        label = meta.column_names_to_labels.get(col, col)
        value_labels = meta.variable_value_labels.get(col, {})

        if value_labels:
            # Sort by numeric key
            sorted_labels = dict(sorted(value_labels.items(), key=lambda x: float(x[0])))
            options = list(sorted_labels.values())

            labels_lower = " ".join(str(v).lower() for v in options)
            likert_patterns = ["strongly", "agree", "disagree", "support", "oppose",
                               "very", "somewhat", "not at all", "confident", "satisfied"]
            if any(p in labels_lower for p in likert_patterns):
                qtype = QuestionType.LIKERT
            elif len(options) == 2:
                qtype = QuestionType.BINARY
            else:
                qtype = QuestionType.CATEGORICAL
        else:
            options = None
            qtype = QuestionType.NUMERIC

        questions.append(Question(
            id=col,
            text=label,
            type=qtype,
            options=options,
        ))

    schema = SurveySchema(survey_name="DC21", questions=questions)
    schema.to_json(str(OUTPUT_DIR / "survey_schema.json"))

    print(f"\nSurvey schema saved: {len(questions)} questions")
    for q in questions:
        opts = f"{len(q.options)} opts" if q.options else "numeric"
        print(f"  [{q.type.value:12s}] {q.id}: {q.text[:60]}  ({opts})")

    print(f"\nOutputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
