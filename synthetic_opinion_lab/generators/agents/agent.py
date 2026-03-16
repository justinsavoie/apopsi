import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import uuid

from ...personas.persona import Persona


@dataclass
class OpinionState:
    """Represents an agent's current opinion state."""
    opinions: Dict[str, float]  # topic -> opinion strength (-1 to 1)
    certainty: Dict[str, float]  # topic -> certainty level (0 to 1)
    last_updated: Dict[str, int]  # topic -> timestep of last update

    def __post_init__(self):
        # Ensure all dictionaries have the same keys
        topics = set(self.opinions.keys()) | set(self.certainty.keys()) | set(self.last_updated.keys())
        for topic in topics:
            if topic not in self.opinions:
                self.opinions[topic] = 0.0
            if topic not in self.certainty:
                self.certainty[topic] = 0.5
            if topic not in self.last_updated:
                self.last_updated[topic] = 0

    def update_opinion(self, topic: str, new_opinion: float, timestep: int, decay_factor: float = 0.1):
        """Update opinion on a topic with some inertia."""
        if topic not in self.opinions:
            self.opinions[topic] = 0.0
            self.certainty[topic] = 0.5

        # Weighted update based on certainty
        current_opinion = self.opinions[topic]
        current_certainty = self.certainty[topic]

        # Higher certainty means less change
        weight = 1.0 - (current_certainty * 0.7)  # Max 70% resistance to change
        updated_opinion = current_opinion + weight * (new_opinion - current_opinion) * decay_factor

        self.opinions[topic] = np.clip(updated_opinion, -1.0, 1.0)
        self.last_updated[topic] = timestep

        # Increase certainty slightly after updates
        self.certainty[topic] = min(1.0, current_certainty + 0.01)


@dataclass
class Memory:
    """Simple memory system for agents."""
    events: deque = field(default_factory=lambda: deque(maxlen=100))
    social_interactions: deque = field(default_factory=lambda: deque(maxlen=50))

    def add_event(self, event: Dict[str, Any]):
        """Add an event to memory."""
        self.events.append(event)

    def add_social_interaction(self, agent_id: str, topic: str, influence: float, timestep: int):
        """Record a social interaction."""
        interaction = {
            "agent_id": agent_id,
            "topic": topic,
            "influence": influence,
            "timestep": timestep
        }
        self.social_interactions.append(interaction)

    def get_recent_interactions(self, topic: str, window: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions about a specific topic."""
        return [
            interaction for interaction in list(self.social_interactions)
            if interaction["topic"] == topic and
               len(self.social_interactions) - list(self.social_interactions).index(interaction) <= window
        ]


class OpinionAgent:
    """Agent representing a synthetic person with opinions that evolve over time."""

    def __init__(self, persona: Persona, initial_topics: List[str]):
        self.id = str(uuid.uuid4())
        self.persona = persona
        self.connections: List[str] = []  # IDs of connected agents

        # Initialize opinion state
        initial_opinions = {}
        initial_certainty = {}
        initial_timestamps = {}

        for topic in initial_topics:
            # Initialize opinions based on persona traits
            opinion = self._trait_to_opinion(topic)
            initial_opinions[topic] = opinion
            initial_certainty[topic] = np.random.uniform(0.3, 0.7)  # Random initial certainty
            initial_timestamps[topic] = 0

        self.opinion_state = OpinionState(
            opinions=initial_opinions,
            certainty=initial_certainty,
            last_updated=initial_timestamps
        )

        self.memory = Memory()

        # Agent parameters
        self.susceptibility = self._calculate_susceptibility()
        self.influence_strength = self._calculate_influence_strength()

    def _trait_to_opinion(self, topic: str) -> float:
        """Convert persona traits to initial opinion on a topic."""
        traits = self.persona.traits
        topic_lower = topic.lower()

        # --- Institutional trust / government ---
        if "government" in topic_lower or "policy" in topic_lower or "gov" in topic_lower:
            return np.clip(0.6 * traits.ideology + 0.4 * traits.trust_in_government, -1.0, 1.0)

        # --- Immigration / trade / globalisation ---
        elif "immigration" in topic_lower or "trade" in topic_lower:
            return np.clip(traits.cosmopolitanism, -1.0, 1.0)

        # --- Authority / law ---
        elif "authority" in topic_lower or "law" in topic_lower:
            return np.clip(traits.authoritarianism, -1.0, 1.0)

        # --- Populist / anti-elite ---
        elif "elite" in topic_lower or "establishment" in topic_lower:
            return np.clip(-traits.populism, -1.0, 1.0)

        # --- Info source: social media (cosmopolitan, liberal, younger) ---
        elif "socmed" in topic_lower or "social_med" in topic_lower:
            return np.clip(0.5 * traits.cosmopolitanism - 0.3 * traits.authoritarianism, -1.0, 1.0)

        # --- Info source: friends / family (informal, populist distrust of institutions) ---
        elif "friendfam" in topic_lower or "friend" in topic_lower:
            return np.clip(0.4 * traits.populism - 0.3 * traits.trust_in_government, -1.0, 1.0)

        # --- Info source: scientists / experts (institutional trust, education) ---
        elif "sci" in topic_lower:
            return np.clip(0.7 * traits.trust_in_government - 0.3 * traits.populism, -1.0, 1.0)

        # --- Info source: politicians (trust in institutions) ---
        elif "pol" in topic_lower:
            return np.clip(traits.trust_in_government, -1.0, 1.0)

        # --- Info source: Canadian mainstream news (establishment trust, ideology) ---
        elif "cannews" in topic_lower:
            return np.clip(0.5 * traits.trust_in_government + 0.3 * traits.ideology, -1.0, 1.0)

        # --- Info source: US or international news (cosmopolitan) ---
        elif "usnews" in topic_lower or "intlnews" in topic_lower:
            return np.clip(traits.cosmopolitanism, -1.0, 1.0)

        # --- Info source: local news (traditionalist, authoritarian) ---
        elif "localnews" in topic_lower or "local" in topic_lower:
            return np.clip(0.4 * traits.authoritarianism - 0.2 * traits.cosmopolitanism, -1.0, 1.0)

        # --- Info source: independent / alternative media (populist, anti-establishment) ---
        elif "indie" in topic_lower:
            return np.clip(0.6 * traits.populism - 0.4 * traits.trust_in_government, -1.0, 1.0)

        # --- Info source: AI chatbots (cosmopolitan, liberal, tech-forward) ---
        elif "ai" in topic_lower:
            return np.clip(0.4 * traits.cosmopolitanism + 0.3 * traits.ideology, -1.0, 1.0)

        # --- Default: ideology ---
        else:
            return np.clip(traits.ideology, -1.0, 1.0)

    def _calculate_susceptibility(self) -> float:
        """Calculate how susceptible agent is to opinion change."""
        traits = self.persona.traits

        # Lower trust and higher populism = higher susceptibility
        base_susceptibility = 0.5
        trust_factor = -0.3 * traits.trust_in_government  # Lower trust = higher susceptibility
        populism_factor = 0.2 * traits.populism  # Higher populism = higher susceptibility

        susceptibility = base_susceptibility + trust_factor + populism_factor
        return np.clip(susceptibility, 0.1, 0.9)

    def _calculate_influence_strength(self) -> float:
        """Calculate how influential this agent is."""
        demo = self.persona.demographics

        # Higher education and income = higher influence
        education_map = {"high_school": 0.3, "college": 0.5, "university": 0.7, "graduate": 0.9}
        income_map = {"low": 0.3, "middle": 0.5, "high": 0.7, "very_high": 0.9}

        education_influence = education_map.get(demo.education, 0.5)
        income_influence = income_map.get(demo.income, 0.5)

        # Age factor (middle-aged people have more influence)
        age_influence = 0.5
        if 30 <= demo.age <= 60:
            age_influence = 0.7
        elif demo.age < 25 or demo.age > 70:
            age_influence = 0.3

        influence = np.mean([education_influence, income_influence, age_influence])
        return np.clip(influence, 0.1, 1.0)

    def add_connection(self, agent_id: str):
        """Add a social connection to another agent."""
        if agent_id not in self.connections:
            self.connections.append(agent_id)

    def remove_connection(self, agent_id: str):
        """Remove a social connection."""
        if agent_id in self.connections:
            self.connections.remove(agent_id)

    def get_opinion(self, topic: str) -> Tuple[float, float]:
        """Get current opinion and certainty on a topic."""
        opinion = self.opinion_state.opinions.get(topic, 0.0)
        certainty = self.opinion_state.certainty.get(topic, 0.5)
        return opinion, certainty

    def receive_social_influence(self, topic: str, neighbor_opinion: float, neighbor_influence: float,
                                neighbor_id: str, timestep: int):
        """Process social influence from a neighbor."""
        current_opinion, current_certainty = self.get_opinion(topic)

        # Calculate influence based on opinion similarity and susceptibility
        opinion_difference = abs(current_opinion - neighbor_opinion)
        similarity_factor = 1.0 - opinion_difference  # More similar = more influence

        # Total influence considering agent's susceptibility
        influence = self.susceptibility * neighbor_influence * similarity_factor * 0.1

        # Calculate opinion shift
        if neighbor_opinion > current_opinion:
            opinion_shift = influence
        else:
            opinion_shift = -influence

        new_opinion = current_opinion + opinion_shift
        self.opinion_state.update_opinion(topic, new_opinion, timestep)

        # Record the interaction
        self.memory.add_social_interaction(neighbor_id, topic, influence, timestep)

    def receive_media_influence(self, topic: str, media_signal: float, media_strength: float, timestep: int):
        """Process influence from media exposure."""
        current_opinion, current_certainty = self.get_opinion(topic)

        # Media influence modified by susceptibility
        influence = self.susceptibility * media_strength * 0.05  # Weaker than social influence

        # Calculate opinion shift toward media signal
        opinion_shift = influence * (media_signal - current_opinion)
        new_opinion = current_opinion + opinion_shift

        self.opinion_state.update_opinion(topic, new_opinion, timestep)

        # Record media event
        self.memory.add_event({
            "type": "media_exposure",
            "topic": topic,
            "signal": media_signal,
            "influence": influence,
            "timestep": timestep
        })

    def respond_to_survey_question(self, topic: str, options: List[str]) -> int:
        """Convert opinion to survey response."""
        opinion, certainty = self.get_opinion(topic)

        # Map opinion (-1 to 1) to response options
        n_options = len(options)

        # Scale opinion to option range
        # -1 maps to option 1, +1 maps to last option
        scaled_opinion = (opinion + 1) / 2  # Convert to 0-1 range
        option_index = int(scaled_opinion * (n_options - 1))

        # Add some noise based on certainty (lower certainty = more noise)
        noise_factor = (1 - certainty) * 0.3
        if np.random.random() < noise_factor:
            # Random walk by ±1 option
            direction = np.random.choice([-1, 1])
            option_index = np.clip(option_index + direction, 0, n_options - 1)

        return option_index + 1  # Convert to 1-based indexing

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of agent's current state."""
        return {
            "agent_id": self.id,
            "persona_id": self.persona.id,
            "opinions": self.opinion_state.opinions.copy(),
            "certainty": self.opinion_state.certainty.copy(),
            "susceptibility": self.susceptibility,
            "influence_strength": self.influence_strength,
            "n_connections": len(self.connections),
            "memory_events": len(self.memory.events),
            "memory_interactions": len(self.memory.social_interactions)
        }