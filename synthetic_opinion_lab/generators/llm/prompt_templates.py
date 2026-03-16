from typing import Dict, List, Optional, Any
from jinja2 import Template
from ...personas.persona import Persona
from ...survey.schema import Question, QuestionType


class PromptTemplate:
    """Base class for survey prompt templates."""

    def __init__(self, template_str: str):
        self.template = Template(template_str)

    def render(self, persona: Persona, question: Question) -> str:
        """Render the template with persona and question data."""
        return self.template.render(
            persona=persona,
            question=question,
            demographics=persona.demographics,
            traits=persona.traits,
            narrative=persona.narrative
        )


class StandardSurveyTemplate(PromptTemplate):
    """Standard template for survey questions."""

    def __init__(self):
        template_str = """You are participating in a survey research study. Please respond as the following person would:

{{ narrative }}

Question: {{ question.text }}

{% if question.options -%}
Please choose from the following options by responding with ONLY the option number (1, 2, 3, etc.):
{% for option in question.options -%}
{{ loop.index }}. {{ option }}
{% endfor %}
{%- else -%}
Please provide a brief response.
{%- endif %}

Your response (number only):"""
        super().__init__(template_str)


class PersonalizedSurveyTemplate(PromptTemplate):
    """More detailed template that includes demographic context."""

    def __init__(self):
        template_str = """You are responding to a survey. Here is your background:

Personal Profile:
- Age: {{ demographics.age }}
- Gender: {{ demographics.gender }}
- Education: {{ demographics.education }}
- Income level: {{ demographics.income }}
- Location: {{ demographics.region }} ({{ 'urban' if demographics.urban else 'rural' }})

Personality and Views:
{{ narrative }}

Based on your background and views, please respond to this question:

{{ question.text }}

{% if question.options -%}
Choose the option that best represents your view by responding with ONLY the number:
{% for option in question.options -%}
{{ loop.index }}. {{ option }}
{% endfor %}

Think carefully about how someone with your background and views would genuinely respond.

Your answer (number only):
{%- else -%}
Provide a brief response that reflects your background and views:
{%- endif %}"""
        super().__init__(template_str)


class ContextualSurveyTemplate(PromptTemplate):
    """Template that provides context about the survey topic."""

    def __init__(self, survey_context: str = ""):
        self.survey_context = survey_context
        context_clause = (' about ' + survey_context) if survey_context else ''
        template_str = "You are participating in a research survey" + context_clause + """.

Your background:
{{ narrative }}

Key characteristics:
- {{ demographics.age }}-year-old {{ demographics.gender }}
- {{ demographics.education }} education, {{ demographics.income }} income
- Lives in {{ demographics.region }} ({{ 'urban' if demographics.urban else 'rural' }} area)

Please answer the following question honestly based on your background and perspectives:

{{ question.text }}

{% if question.options -%}
Options:
{% for option in question.options -%}
{{ loop.index }}. {{ option }}
{% endfor %}

Respond with the number (1-{{ question.options|length }}) that best matches your view:
{%- else -%}
Your response:
{%- endif %}"""
        super().__init__(template_str)


class TemplateManager:
    """Manages different prompt templates for survey generation."""

    def __init__(self):
        self.templates = {
            "standard": StandardSurveyTemplate(),
            "personalized": PersonalizedSurveyTemplate(),
            "contextual": ContextualSurveyTemplate()
        }

    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a template by name."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        return self.templates[template_name]

    def add_template(self, name: str, template: PromptTemplate) -> None:
        """Add a custom template."""
        self.templates[name] = template

    def create_contextual_template(self, survey_context: str) -> ContextualSurveyTemplate:
        """Create a contextual template with specific survey context."""
        return ContextualSurveyTemplate(survey_context)

    def generate_prompt(self, template_name: str, persona: Persona, question: Question) -> str:
        """Generate a prompt using the specified template."""
        template = self.get_template(template_name)
        return template.render(persona, question)


class AdaptivePromptStrategy:
    """Adapts prompts based on question type and persona characteristics."""

    def __init__(self, template_manager: TemplateManager):
        self.template_manager = template_manager

    def select_template(self, persona: Persona, question: Question) -> str:
        """Select the best template based on persona and question characteristics."""

        # For political questions, use personalized template
        political_keywords = ['government', 'policy', 'vote', 'party', 'politics', 'tax', 'immigration']
        if any(keyword in question.text.lower() for keyword in political_keywords):
            return "personalized"

        # For attitude questions, use standard template
        attitude_keywords = ['agree', 'support', 'opinion', 'feel', 'think']
        if any(keyword in question.text.lower() for keyword in attitude_keywords):
            return "standard"

        # Default to standard
        return "standard"

    def generate_adaptive_prompt(self, persona: Persona, question: Question) -> str:
        """Generate an adaptive prompt based on persona and question."""
        template_name = self.select_template(persona, question)
        return self.template_manager.generate_prompt(template_name, persona, question)