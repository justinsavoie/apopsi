#!/usr/bin/env python3
"""
Example usage of the Synthetic Opinion Lab framework.

This script demonstrates how to use the framework to generate synthetic survey data.
"""

import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'synthetic-opinion-lab'))

from synthetic_opinion_lab import (
    run_simple_experiment,
    compare_all_generators,
    PersonaGenerator,
    IRTResponseGenerator,
    EvaluationPipeline
)
from synthetic_opinion_lab.survey.schema import SurveySchema, Question, QuestionType
from synthetic_opinion_lab.personas.trait_models import TraitModel
from synthetic_opinion_lab.personas.persona_generator import DemographicDistribution

import pandas as pd
import numpy as np


def create_example_survey_data():
    """Create a simple example survey dataset."""
    print("Creating example survey data...")

    # Define survey questions
    questions = [
        Question(
            id="Q1",
            text="Do you support the carbon tax?",
            type=QuestionType.LIKERT,
            options=["Strongly oppose", "Oppose", "Neutral", "Support", "Strongly support"]
        ),
        Question(
            id="Q2",
            text="Do you trust the government?",
            type=QuestionType.LIKERT,
            options=["Not at all", "A little", "Somewhat", "Quite a bit", "Completely"]
        ),
        Question(
            id="Q3",
            text="Should immigration be increased?",
            type=QuestionType.BINARY,
            options=["No", "Yes"]
        )
    ]

    # Create survey schema
    survey_schema = SurveySchema(
        survey_name="Example Political Survey",
        questions=questions
    )

    # Generate some realistic fake data for the "real" survey
    np.random.seed(42)
    n_real = 800

    real_data = pd.DataFrame({
        'respondent_id': [f"real_{i}" for i in range(n_real)],
        'Q1': np.random.choice([1, 2, 3, 4, 5], n_real, p=[0.2, 0.25, 0.3, 0.15, 0.1]),
        'Q2': np.random.choice([1, 2, 3, 4, 5], n_real, p=[0.15, 0.2, 0.35, 0.2, 0.1]),
        'Q3': np.random.choice([1, 2], n_real, p=[0.6, 0.4])
    })

    return real_data, survey_schema


def example_persona_generation():
    """Demonstrate persona generation."""
    print("\n=== Persona Generation Example ===")

    # Create persona generator with Canadian demographics
    generator = PersonaGenerator.default_canadian()

    # Generate 10 personas
    personas = generator.generate(10, random_state=42)

    print(f"Generated {len(personas)} personas:")
    for i, persona in enumerate(personas[:3]):  # Show first 3
        print(f"\n{i+1}. {persona.summary()}")
        print(f"   Traits: ideology={persona.traits.ideology:.2f}, "
              f"authoritarianism={persona.traits.authoritarianism:.2f}")
        print(f"   Narrative: {persona.narrative[:100]}...")

    return personas


def example_irt_generation():
    """Demonstrate IRT response generation."""
    print("\n=== IRT Response Generation Example ===")

    # Create example data
    real_data, survey_schema = create_example_survey_data()

    # Generate personas
    generator = PersonaGenerator.default_canadian()
    personas = generator.generate(100, random_state=42)

    # Generate responses using IRT
    irt_generator = IRTResponseGenerator()
    synthetic_data = irt_generator.generate(personas, survey_schema)

    print(f"Generated {len(synthetic_data)} synthetic responses using IRT")
    print("First 5 responses:")
    print(synthetic_data.head())

    # Quick evaluation
    evaluator = EvaluationPipeline()

    # Prepare variable types
    variable_types = {"Q1": "ordinal", "Q2": "ordinal", "Q3": "categorical"}
    ordinal_mappings = {
        "Q1": ["Strongly oppose", "Oppose", "Neutral", "Support", "Strongly support"],
        "Q2": ["Not at all", "A little", "Somewhat", "Quite a bit", "Completely"]
    }

    eval_results = evaluator.run_full_evaluation(
        real_data, synthetic_data,
        variable_types=variable_types,
        ordinal_mappings=ordinal_mappings
    )

    overall_quality = eval_results.get("overall_quality", {})
    if "summary" in overall_quality:
        score = overall_quality["summary"].get("overall_score", 0)
        print(f"Overall quality score: {score:.3f}")

    return synthetic_data, eval_results


def example_simple_experiment():
    """Demonstrate running a simple experiment."""
    print("\n=== Simple Experiment Example ===")

    # Create a temporary CSV file for the experiment
    real_data, survey_schema = create_example_survey_data()
    temp_file = "temp_survey.csv"
    real_data.to_csv(temp_file, index=False)

    try:
        # Run simple IRT experiment
        print("Running simple IRT experiment...")
        results = run_simple_experiment(
            survey_file=temp_file,
            generator_type="irt",
            n_personas=200,
            output_dir="./example_output"
        )

        print("Experiment completed!")
        print(f"Quality score: {results.get('evaluation_results', {}).get('overall_quality', {}).get('summary', {}).get('overall_score', 'N/A')}")

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    """Run all examples."""
    print("Synthetic Opinion Lab - Example Usage")
    print("=" * 50)

    try:
        # Example 1: Persona Generation
        personas = example_persona_generation()

        # Example 2: IRT Generation and Evaluation
        synthetic_data, eval_results = example_irt_generation()

        # Example 3: Simple Experiment Pipeline
        example_simple_experiment()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nNext steps:")
        print("1. Try with your own survey data files (.sav, .csv, .dta)")
        print("2. Experiment with LLM generators (requires Ollama or Together API)")
        print("3. Try agent-based simulations for opinion dynamics")
        print("4. Use the comparison functions to test multiple approaches")

    except Exception as e:
        print(f"\nExample failed with error: {e}")
        print("Make sure you have all required dependencies installed:")
        print("pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly pyreadstat requests")


if __name__ == "__main__":
    main()