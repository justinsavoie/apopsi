from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class Demographics:
    age: int
    gender: str
    education: str
    income: str
    region: str
    urban: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatentTraits:
    ideology: float  # -2 to +2, conservative to liberal
    authoritarianism: float  # -2 to +2, libertarian to authoritarian
    trust_in_government: float  # -2 to +2, low to high trust
    populism: float  # -2 to +2, low to high populism
    cosmopolitanism: float  # -2 to +2, nationalist to cosmopolitan

    def __post_init__(self):
        """Validate trait values are within range."""
        for field_name, value in asdict(self).items():
            if not -2 <= value <= 2:
                raise ValueError(f"Trait {field_name} must be between -2 and +2, got {value}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Persona:
    id: str
    demographics: Demographics
    traits: LatentTraits
    narrative: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "demographics": self.demographics.to_dict(),
            "traits": self.traits.to_dict(),
            "narrative": self.narrative
        }

    def to_json(self, filepath: str) -> None:
        """Save persona to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Persona':
        """Create persona from dictionary."""
        demographics = Demographics(**data["demographics"])
        traits = LatentTraits(**data["traits"])
        return cls(
            id=data["id"],
            demographics=demographics,
            traits=traits,
            narrative=data["narrative"]
        )

    @classmethod
    def from_json(cls, filepath: str) -> 'Persona':
        """Load persona from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_trait_vector(self) -> Dict[str, float]:
        """Get traits as a flat dictionary for modeling."""
        return self.traits.to_dict()

    def get_demographic_vector(self) -> Dict[str, Any]:
        """Get demographics as a flat dictionary for modeling."""
        return self.demographics.to_dict()

    def summary(self) -> str:
        """Generate a brief summary of the persona."""
        demo = self.demographics
        return (f"Persona {self.id}: {demo.age}yo {demo.gender} from {demo.region}, "
               f"{demo.education} education, {demo.income} income, "
               f"{'urban' if demo.urban else 'rural'}")