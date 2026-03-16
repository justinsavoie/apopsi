import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from tqdm import tqdm

from ..base import ResponseGenerator
from ...personas.persona import Persona
from ...survey.schema import SurveySchema, Question, QuestionType
from .agent import OpinionAgent
from .network import SocialNetwork


class AgentBasedSimulator(ResponseGenerator):
    """Agent-based simulation for opinion dynamics."""

    def __init__(self,
                 n_timesteps: int = 50,
                 network_type: str = "small_world",
                 network_params: Optional[Dict[str, Any]] = None,
                 media_influence_strength: float = 0.1,
                 convergence_threshold: float = 0.01):
        """
        Initialize agent-based simulator.

        Args:
            n_timesteps: Number of simulation timesteps
            network_type: Type of network ("small_world", "demographic", "preferential_attachment")
            network_params: Parameters for network creation
            media_influence_strength: Strength of media influence
            convergence_threshold: Threshold for detecting convergence
        """
        self.n_timesteps = n_timesteps
        self.network_type = network_type
        self.network_params = network_params or {}
        self.media_influence_strength = media_influence_strength
        self.convergence_threshold = convergence_threshold

        self.agents: List[OpinionAgent] = []
        self.network: Optional[SocialNetwork] = None
        self.simulation_history: List[Dict[str, Any]] = []

    def generate(self, personas: List[Persona], survey: SurveySchema) -> pd.DataFrame:
        """
        Generate survey responses using agent-based simulation.

        Args:
            personas: List of persona objects
            survey: Survey schema defining questions

        Returns:
            DataFrame with rows=respondents, columns=survey questions
        """
        # Extract topics from survey questions
        topics = [q.id for q in survey.questions]

        # Initialize agents
        self._initialize_agents(personas, topics)

        # Create social network
        self._create_network()

        # Run simulation
        self._run_simulation(topics)

        # Generate survey responses
        responses = self._generate_survey_responses(survey)

        return responses

    def _initialize_agents(self, personas: List[Persona], topics: List[str]) -> None:
        """Initialize opinion agents from personas."""
        self.agents = []
        for persona in personas:
            agent = OpinionAgent(persona, topics)
            self.agents.append(agent)

    def _create_network(self) -> None:
        """Create the social network structure."""
        self.network = SocialNetwork(self.agents)

        if self.network_type == "small_world":
            k = self.network_params.get("k", 6)
            p = self.network_params.get("p", 0.1)
            self.network.create_small_world_network(k=k, p=p)

        elif self.network_type == "demographic":
            connection_prob = self.network_params.get("connection_probability", 0.1)
            homophily = self.network_params.get("homophily_strength", 0.7)
            self.network.create_demographic_network(connection_prob, homophily)

        elif self.network_type == "preferential_attachment":
            m = self.network_params.get("m", 3)
            self.network.create_preferential_attachment_network(m=m)

        else:
            raise ValueError(f"Unknown network type: {self.network_type}")

    def _run_simulation(self, topics: List[str]) -> None:
        """Run the opinion dynamics simulation."""
        self.simulation_history = []

        for timestep in tqdm(range(self.n_timesteps), desc="Running simulation"):
            timestep_data = {"timestep": timestep, "agents": {}}

            # Store initial state
            for agent in self.agents:
                timestep_data["agents"][agent.id] = {
                    "opinions": agent.opinion_state.opinions.copy(),
                    "certainty": agent.opinion_state.certainty.copy()
                }

            # Social influence step
            self._social_influence_step(topics, timestep)

            # Media influence step (optional)
            if self.media_influence_strength > 0:
                self._media_influence_step(topics, timestep)

            # Record final state
            for agent in self.agents:
                timestep_data["agents"][agent.id].update({
                    "final_opinions": agent.opinion_state.opinions.copy(),
                    "final_certainty": agent.opinion_state.certainty.copy()
                })

            self.simulation_history.append(timestep_data)

            # Check for convergence
            if timestep > 10 and self._check_convergence(topics):
                print(f"Simulation converged at timestep {timestep}")
                break

    def _social_influence_step(self, topics: List[str], timestep: int) -> None:
        """Execute social influence among connected agents."""
        for agent in self.agents:
            neighbors = self.network.get_neighbors(agent.id)

            for topic in topics:
                if not neighbors:
                    continue

                # Get opinions from all neighbors
                neighbor_influences = []
                for neighbor_id in neighbors:
                    neighbor = self.network.agents[neighbor_id]
                    neighbor_opinion, _ = neighbor.get_opinion(topic)
                    neighbor_influences.append((neighbor_opinion, neighbor.influence_strength, neighbor_id))

                # Apply influence from each neighbor
                for neighbor_opinion, neighbor_influence, neighbor_id in neighbor_influences:
                    agent.receive_social_influence(topic, neighbor_opinion, neighbor_influence,
                                                 neighbor_id, timestep)

    def _media_influence_step(self, topics: List[str], timestep: int) -> None:
        """Apply media influence to all agents."""
        for topic in topics:
            # Generate media signal (could be more sophisticated)
            media_signal = self._generate_media_signal(topic, timestep)

            for agent in self.agents:
                agent.receive_media_influence(topic, media_signal, self.media_influence_strength, timestep)

    def _generate_media_signal(self, topic: str, timestep: int) -> float:
        """Generate media signal for a topic at given timestep."""
        # Simple implementation: random walk around center
        # In practice, this could be based on real events or news cycles

        # For political topics, media might lean slightly in one direction
        if "government" in topic.lower() or "policy" in topic.lower():
            center = np.random.choice([-0.3, 0.3])  # Slight political lean
        else:
            center = 0.0

        # Add some noise and temporal evolution
        noise = np.random.normal(0, 0.2)
        temporal_component = 0.1 * np.sin(timestep / 10)  # Slow cyclical changes

        signal = center + noise + temporal_component
        return np.clip(signal, -1.0, 1.0)

    def _check_convergence(self, topics: List[str]) -> bool:
        """Check if opinions have converged."""
        if len(self.simulation_history) < 2:
            return False

        current_state = self.simulation_history[-1]
        previous_state = self.simulation_history[-2]

        total_change = 0.0
        n_comparisons = 0

        for agent_id in current_state["agents"]:
            for topic in topics:
                current_opinion = current_state["agents"][agent_id]["final_opinions"][topic]
                previous_opinion = previous_state["agents"][agent_id]["final_opinions"][topic]

                change = abs(current_opinion - previous_opinion)
                total_change += change
                n_comparisons += 1

        average_change = total_change / n_comparisons if n_comparisons > 0 else 0.0
        return average_change < self.convergence_threshold

    def _generate_survey_responses(self, survey: SurveySchema) -> pd.DataFrame:
        """Generate survey responses from final agent opinions."""
        n_agents = len(self.agents)
        responses = pd.DataFrame(index=range(n_agents))

        # Add agent/persona identifiers
        responses['respondent_id'] = [agent.persona.id for agent in self.agents]

        # Generate responses for each question
        for question in survey.questions:
            question_responses = []

            for agent in self.agents:
                if question.options:
                    # Convert opinion to categorical response
                    response = agent.respond_to_survey_question(question.id, question.options)
                else:
                    # For open-ended questions, convert opinion to text
                    opinion, certainty = agent.get_opinion(question.id)
                    if opinion > 0.5:
                        response = "Strongly agree"
                    elif opinion > 0:
                        response = "Somewhat agree"
                    elif opinion < -0.5:
                        response = "Strongly disagree"
                    elif opinion < 0:
                        response = "Somewhat disagree"
                    else:
                        response = "Neutral"

                question_responses.append(response)

            responses[question.id] = question_responses

        return responses

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the simulation."""
        if not self.simulation_history:
            return {"error": "No simulation has been run"}

        network_stats = self.network.get_network_stats() if self.network else {}

        # Calculate opinion volatility
        volatilities = {}
        for topic in self.agents[0].opinion_state.opinions.keys():
            topic_changes = []
            for i in range(1, len(self.simulation_history)):
                timestep_change = 0.0
                for agent_id in self.simulation_history[i]["agents"]:
                    current = self.simulation_history[i]["agents"][agent_id]["final_opinions"][topic]
                    previous = self.simulation_history[i-1]["agents"][agent_id]["final_opinions"][topic]
                    timestep_change += abs(current - previous)
                topic_changes.append(timestep_change / len(self.agents))
            volatilities[topic] = np.mean(topic_changes)

        return {
            "simulation_params": {
                "n_timesteps": self.n_timesteps,
                "network_type": self.network_type,
                "network_params": self.network_params,
                "media_influence_strength": self.media_influence_strength
            },
            "network_stats": network_stats,
            "n_agents": len(self.agents),
            "simulation_length": len(self.simulation_history),
            "opinion_volatility": volatilities,
            "final_opinions": {
                agent.id: agent.opinion_state.opinions.copy()
                for agent in self.agents
            }
        }

    def export_simulation_data(self, filepath: str) -> None:
        """Export simulation history to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "summary": self.get_simulation_summary(),
                "history": self.simulation_history
            }, f, indent=2)

    def get_opinion_trajectories(self) -> Dict[str, pd.DataFrame]:
        """Get opinion trajectories for all agents and topics."""
        trajectories = {}

        for topic in self.agents[0].opinion_state.opinions.keys():
            # Create dataframe: timesteps x agents
            data = []

            for timestep_data in self.simulation_history:
                timestep_opinions = []
                for agent in self.agents:
                    agent_id = agent.id
                    opinion = timestep_data["agents"][agent_id]["final_opinions"][topic]
                    timestep_opinions.append(opinion)
                data.append(timestep_opinions)

            columns = [f"agent_{i}" for i in range(len(self.agents))]
            trajectories[topic] = pd.DataFrame(data, columns=columns)

        return trajectories