#!/usr/bin/env python3
import sys, json
sys.path.insert(0, '.')
import pandas as pd, numpy as np
from synthetic_opinion_lab.survey.schema import SurveySchema, Question, QuestionType
from synthetic_opinion_lab.personas.persona_generator import PersonaGenerator, DemographicDistribution
from synthetic_opinion_lab.personas.trait_models import TraitModel
from synthetic_opinion_lab.generators.irt.irt_generator import IRTResponseGenerator

with open('dc21_output/survey_schema.json') as f:
    schema_raw = json.load(f)

questions = [Question(id=q['id'],text=q['text'],type=QuestionType(q['type']),options=q.get('options')) for q in schema_raw['questions']]
survey = SurveySchema(survey_name='DC21', questions=questions)

personas = PersonaGenerator(DemographicDistribution.canada_census(), TraitModel.default_political_model()).generate(50, random_state=42)

irt = IRTResponseGenerator()
syn = irt.generate(personas, survey)
print('Shape:', syn.shape)
print(syn.head(3).to_string())
print()
for col in [c for c in syn.columns if c != 'respondent_id']:
    opts = next(q.options for q in questions if q.id == col)
    n_opts = len(opts)
    valid = syn[col].between(1, n_opts).sum()
    print(f'{col}: valid={valid}/50, range={syn[col].min():.0f}-{syn[col].max():.0f}')
