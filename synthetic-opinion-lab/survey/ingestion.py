import pandas as pd
import pyreadstat
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

from .schema import SurveySchema, Question, QuestionType


class SurveyIngester:
    """Handles ingestion of survey data from multiple formats."""

    @staticmethod
    def ingest(filepath: str, survey_name: Optional[str] = None) -> Tuple[pd.DataFrame, SurveySchema]:
        """
        Main ingestion method that routes to appropriate handler based on file extension.

        Returns:
            Tuple of (cleaned_dataframe, survey_schema)
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Survey file not found: {filepath}")

        if survey_name is None:
            survey_name = path.stem

        if path.suffix.lower() == '.sav':
            return SurveyIngester._ingest_spss(filepath, survey_name)
        elif path.suffix.lower() == '.dta':
            return SurveyIngester._ingest_stata(filepath, survey_name)
        elif path.suffix.lower() in ['.csv', '.tsv']:
            return SurveyIngester._ingest_csv(filepath, survey_name)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    @staticmethod
    def _ingest_spss(filepath: str, survey_name: str) -> Tuple[pd.DataFrame, SurveySchema]:
        """Ingest SPSS (.sav) files."""
        df, meta = pyreadstat.read_sav(filepath)

        # Clean the dataframe
        df_clean = SurveyIngester._clean_dataframe(df)

        # Extract schema from metadata
        schema = SurveyIngester._extract_schema_from_spss_meta(meta, survey_name)

        return df_clean, schema

    @staticmethod
    def _ingest_stata(filepath: str, survey_name: str) -> Tuple[pd.DataFrame, SurveySchema]:
        """Ingest Stata (.dta) files."""
        df, meta = pyreadstat.read_dta(filepath)

        # Clean the dataframe
        df_clean = SurveyIngester._clean_dataframe(df)

        # Extract schema from metadata
        schema = SurveyIngester._extract_schema_from_stata_meta(meta, survey_name)

        return df_clean, schema

    @staticmethod
    def _ingest_csv(filepath: str, survey_name: str) -> Tuple[pd.DataFrame, SurveySchema]:
        """Ingest CSV/TSV files."""
        sep = '\t' if filepath.endswith('.tsv') else ','
        df = pd.read_csv(filepath, sep=sep)

        # Clean the dataframe
        df_clean = SurveyIngester._clean_dataframe(df)

        # Infer schema from data
        schema = SurveyIngester._infer_schema_from_dataframe(df_clean, survey_name)

        return df_clean, schema

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Apply general cleaning to dataframe."""
        df_clean = df.copy()

        # Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all')
        df_clean = df_clean.dropna(axis=1, how='all')

        # Standardize column names (remove spaces, special chars)
        df_clean.columns = [col.strip().replace(' ', '_').replace('-', '_')
                           for col in df_clean.columns]

        return df_clean

    @staticmethod
    def _extract_schema_from_spss_meta(meta: pyreadstat.metadata_container, survey_name: str) -> SurveySchema:
        """Extract schema from SPSS metadata."""
        questions = []

        for col_name in meta.column_names:
            # Get variable label (question text)
            var_label = meta.column_names_to_labels.get(col_name, col_name)

            # Get value labels (response options)
            value_labels = meta.variable_value_labels.get(col_name, {})

            # Determine question type and options
            question_type, options = SurveyIngester._determine_question_type(value_labels)

            question = Question(
                id=col_name,
                text=var_label,
                type=question_type,
                options=options
            )
            questions.append(question)

        return SurveySchema(survey_name=survey_name, questions=questions)

    @staticmethod
    def _extract_schema_from_stata_meta(meta: pyreadstat.metadata_container, survey_name: str) -> SurveySchema:
        """Extract schema from Stata metadata."""
        questions = []

        for col_name in meta.column_names:
            # Get variable label (question text)
            var_label = meta.column_names_to_labels.get(col_name, col_name)

            # Get value labels (response options)
            value_labels = meta.variable_value_labels.get(col_name, {})

            # Determine question type and options
            question_type, options = SurveyIngester._determine_question_type(value_labels)

            question = Question(
                id=col_name,
                text=var_label,
                type=question_type,
                options=options
            )
            questions.append(question)

        return SurveySchema(survey_name=survey_name, questions=questions)

    @staticmethod
    def _infer_schema_from_dataframe(df: pd.DataFrame, survey_name: str) -> SurveySchema:
        """Infer schema from DataFrame when no metadata is available."""
        questions = []

        for col_name in df.columns:
            col_data = df[col_name].dropna()

            # Determine question type based on data patterns
            if col_data.dtype == 'bool' or set(col_data.unique()) <= {0, 1, True, False}:
                question_type = QuestionType.BINARY
                options = ["No", "Yes"]
            elif col_data.dtype in ['int64', 'float64'] and len(col_data.unique()) <= 10:
                # Likely ordinal/categorical
                unique_vals = sorted(col_data.unique())
                if len(unique_vals) <= 5 and min(unique_vals) == 1:
                    # Likely Likert scale
                    question_type = QuestionType.LIKERT
                    options = [f"Option {i}" for i in unique_vals]
                else:
                    question_type = QuestionType.CATEGORICAL
                    options = [str(val) for val in unique_vals]
            elif col_data.dtype in ['int64', 'float64']:
                question_type = QuestionType.NUMERIC
                options = None
            else:
                # String/object type
                unique_vals = col_data.unique()
                if len(unique_vals) <= 10:
                    question_type = QuestionType.CATEGORICAL
                    options = list(unique_vals)
                else:
                    question_type = QuestionType.CATEGORICAL
                    options = None

            question = Question(
                id=col_name,
                text=col_name.replace('_', ' ').title(),
                type=question_type,
                options=options
            )
            questions.append(question)

        return SurveySchema(survey_name=survey_name, questions=questions)

    @staticmethod
    def _determine_question_type(value_labels: Dict[Any, str]) -> Tuple[QuestionType, Optional[List[str]]]:
        """Determine question type from value labels."""
        if not value_labels:
            return QuestionType.NUMERIC, None

        # Check for binary
        if len(value_labels) == 2:
            return QuestionType.BINARY, list(value_labels.values())

        # Check for Likert patterns
        labels_lower = [str(label).lower() for label in value_labels.values()]
        likert_patterns = [
            'strongly', 'agree', 'disagree', 'support', 'oppose',
            'very', 'extremely', 'somewhat', 'not at all'
        ]

        if any(pattern in ' '.join(labels_lower) for pattern in likert_patterns):
            return QuestionType.LIKERT, list(value_labels.values())

        # Default to categorical
        return QuestionType.CATEGORICAL, list(value_labels.values())

    @staticmethod
    def save_outputs(df: pd.DataFrame, schema: SurveySchema, output_dir: str) -> Dict[str, str]:
        """Save cleaned data and schema to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save cleaned data as parquet
        data_path = output_path / "clean_data.parquet"
        df.to_parquet(data_path, index=False)

        # Save schema as JSON
        schema_path = output_path / "survey_schema.json"
        schema.to_json(str(schema_path))

        return {
            "data_path": str(data_path),
            "schema_path": str(schema_path)
        }