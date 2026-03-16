from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json


class QuestionType(Enum):
    BINARY = "binary"
    LIKERT = "likert"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


@dataclass
class Question:
    id: str
    text: str
    type: QuestionType
    options: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "text": self.text,
            "type": self.type.value
        }
        if self.options:
            result["options"] = self.options
        if self.min_value is not None:
            result["min_value"] = self.min_value
        if self.max_value is not None:
            result["max_value"] = self.max_value
        return result


@dataclass
class SurveySchema:
    survey_name: str
    questions: List[Question]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "survey_name": self.survey_name,
            "questions": [q.to_dict() for q in self.questions]
        }

    def to_json(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SurveySchema':
        questions = []
        for q_data in data["questions"]:
            question = Question(
                id=q_data["id"],
                text=q_data["text"],
                type=QuestionType(q_data["type"]),
                options=q_data.get("options"),
                min_value=q_data.get("min_value"),
                max_value=q_data.get("max_value")
            )
            questions.append(question)

        return cls(
            survey_name=data["survey_name"],
            questions=questions
        )

    @classmethod
    def from_json(cls, filepath: str) -> 'SurveySchema':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)