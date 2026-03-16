import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import json

from .persona import Persona, Demographics, LatentTraits
from .trait_models import TraitGenerator, TraitModel


@dataclass
class DemographicDistribution:
    """Defines demographic distributions for persona generation."""
    age_distribution: Dict[str, float]  # age_group -> probability
    gender_distribution: Dict[str, float]
    education_distribution: Dict[str, float]
    income_distribution: Dict[str, float]
    region_distribution: Dict[str, float]
    urban_probability: float

    @classmethod
    def canada_census(cls) -> 'DemographicDistribution':
        """Approximate Canadian census demographics."""
        return cls(
            age_distribution={
                "18-29": 0.20,
                "30-44": 0.25,
                "45-64": 0.35,
                "65+": 0.20
            },
            gender_distribution={
                "male": 0.49,
                "female": 0.51
            },
            education_distribution={
                "high_school": 0.25,
                "college": 0.35,
                "university": 0.30,
                "graduate": 0.10
            },
            income_distribution={
                "low": 0.25,
                "middle": 0.50,
                "high": 0.20,
                "very_high": 0.05
            },
            region_distribution={
                "British Columbia": 0.13,
                "Alberta": 0.12,
                "Saskatchewan": 0.03,
                "Manitoba": 0.04,
                "Ontario": 0.39,
                "Quebec": 0.23,
                "Atlantic": 0.06
            },
            urban_probability=0.81
        )

    @classmethod
    def us_census(cls) -> 'DemographicDistribution':
        """Approximate US census demographics."""
        return cls(
            age_distribution={
                "18-29": 0.22,
                "30-44": 0.24,
                "45-64": 0.34,
                "65+": 0.20
            },
            gender_distribution={
                "male": 0.49,
                "female": 0.51
            },
            education_distribution={
                "high_school": 0.28,
                "college": 0.32,
                "university": 0.30,
                "graduate": 0.10
            },
            income_distribution={
                "low": 0.25,
                "middle": 0.45,
                "high": 0.25,
                "very_high": 0.05
            },
            region_distribution={
                "Northeast": 0.17,
                "Midwest": 0.21,
                "South": 0.38,
                "West": 0.24
            },
            urban_probability=0.82
        )


class PersonaGenerator:
    """Generates synthetic personas with demographics, traits, and narratives."""

    def __init__(self,
                 demographic_dist: DemographicDistribution,
                 trait_model: TraitModel):
        self.demographic_dist = demographic_dist
        self.trait_generator = TraitGenerator(trait_model)

    @classmethod
    def default_canadian(cls) -> 'PersonaGenerator':
        """Default generator with Canadian demographics and political traits."""
        return cls(
            demographic_dist=DemographicDistribution.canada_census(),
            trait_model=TraitModel.default_political_model()
        )

    @classmethod
    def default_american(cls) -> 'PersonaGenerator':
        """Default generator with US demographics and political traits."""
        return cls(
            demographic_dist=DemographicDistribution.us_census(),
            trait_model=TraitModel.default_political_model()
        )

    def generate(self, n_personas: int, random_state: Optional[int] = None) -> List[Persona]:
        """Generate n personas with demographics, traits, and narratives."""
        if random_state is not None:
            np.random.seed(random_state)

        # Generate traits
        traits_list = self.trait_generator.generate_traits(n_personas, random_state)

        # Generate demographics
        demographics_list = self._generate_demographics(n_personas)

        # Create personas
        personas = []
        for i in range(n_personas):
            persona_id = f"persona_{i+1:06d}"

            # Generate narrative
            narrative = self._generate_narrative(demographics_list[i], traits_list[i])

            persona = Persona(
                id=persona_id,
                demographics=demographics_list[i],
                traits=traits_list[i],
                narrative=narrative
            )
            personas.append(persona)

        return personas

    def _generate_demographics(self, n_personas: int) -> List[Demographics]:
        """Generate demographics for n personas."""
        demographics_list = []

        for _ in range(n_personas):
            # Sample age
            age_group = self._sample_from_distribution(self.demographic_dist.age_distribution)
            age = self._sample_age_from_group(age_group)

            # Sample other demographics
            gender = self._sample_from_distribution(self.demographic_dist.gender_distribution)
            education = self._sample_from_distribution(self.demographic_dist.education_distribution)
            income = self._sample_from_distribution(self.demographic_dist.income_distribution)
            region = self._sample_from_distribution(self.demographic_dist.region_distribution)
            urban = np.random.random() < self.demographic_dist.urban_probability

            demographics = Demographics(
                age=age,
                gender=gender,
                education=education,
                income=income,
                region=region,
                urban=urban
            )
            demographics_list.append(demographics)

        return demographics_list

    def _sample_from_distribution(self, distribution: Dict[str, float]) -> str:
        """Sample a category from a probability distribution."""
        categories = list(distribution.keys())
        probabilities = list(distribution.values())
        return np.random.choice(categories, p=probabilities)

    def _sample_age_from_group(self, age_group: str) -> int:
        """Sample specific age from age group."""
        if age_group == "18-29":
            return np.random.randint(18, 30)
        elif age_group == "30-44":
            return np.random.randint(30, 45)
        elif age_group == "45-64":
            return np.random.randint(45, 65)
        elif age_group == "65+":
            return np.random.randint(65, 85)
        else:
            return 35  # fallback

    def _generate_narrative(self, demographics: Demographics, traits: LatentTraits) -> str:
        """Generate narrative description for persona."""
        # Create basic demographic description
        pronoun = "He" if demographics.gender == "male" else "She"
        location_desc = "urban" if demographics.urban else "rural"

        # Map education to readable form
        education_map = {
            "high_school": "high school",
            "college": "college",
            "university": "university",
            "graduate": "graduate school"
        }
        education_str = education_map.get(demographics.education, demographics.education)

        # Map income to readable form
        income_map = {
            "low": "lower-income",
            "middle": "middle-income",
            "high": "upper-middle-income",
            "very_high": "high-income"
        }
        income_str = income_map.get(demographics.income, demographics.income)

        # Generate personality description based on traits
        personality_traits = []

        # Ideology
        if traits.ideology > 0.5:
            personality_traits.append("progressive")
        elif traits.ideology < -0.5:
            personality_traits.append("conservative")
        else:
            personality_traits.append("moderate")

        # Authoritarianism
        if traits.authoritarianism > 0.5:
            personality_traits.append("values order and authority")
        elif traits.authoritarianism < -0.5:
            personality_traits.append("values individual freedom")

        # Trust in government
        if traits.trust_in_government > 0.5:
            personality_traits.append("trusts government institutions")
        elif traits.trust_in_government < -0.5:
            personality_traits.append("skeptical of government")

        # Populism
        if traits.populism > 0.5:
            personality_traits.append("distrusts political elites")

        # Cosmopolitanism
        if traits.cosmopolitanism > 0.5:
            personality_traits.append("globally minded")
        elif traits.cosmopolitanism < -0.5:
            personality_traits.append("nationally focused")

        personality_str = ", ".join(personality_traits)

        narrative = (f"{pronoun} is a {demographics.age}-year-old {demographics.gender} "
                    f"from {location_desc} {demographics.region}. {pronoun} has a "
                    f"{education_str} education and is {income_str}. {pronoun} is {personality_str}.")

        return narrative

    def save_personas(self, personas: List[Persona], filepath: str) -> None:
        """Save personas to JSON file."""
        personas_data = [p.to_dict() for p in personas]
        with open(filepath, 'w') as f:
            json.dump(personas_data, f, indent=2)

    @staticmethod
    def load_personas(filepath: str) -> List[Persona]:
        """Load personas from JSON file."""
        with open(filepath, 'r') as f:
            personas_data = json.load(f)
        return [Persona.from_dict(data) for data in personas_data]