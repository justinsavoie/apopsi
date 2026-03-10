import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from .agent import OpinionAgent
from ...personas.persona import Persona


class SocialNetwork:
    """Manages the social network structure for opinion agents."""

    def __init__(self, agents: List[OpinionAgent]):
        self.agents = {agent.id: agent for agent in agents}
        self.graph = nx.Graph()

        # Add all agents as nodes
        for agent in agents:
            self.graph.add_node(agent.id, agent=agent)

    def create_demographic_network(self, connection_probability: float = 0.1,
                                 homophily_strength: float = 0.7) -> None:
        """
        Create network connections based on demographic similarity.

        Args:
            connection_probability: Base probability of connection
            homophily_strength: How much demographics influence connection probability
        """
        agents_list = list(self.agents.values())

        for i, agent1 in enumerate(agents_list):
            for agent2 in agents_list[i+1:]:
                # Calculate demographic similarity
                similarity = self._calculate_demographic_similarity(agent1.persona, agent2.persona)

                # Adjust connection probability based on similarity
                adjusted_prob = connection_probability + (homophily_strength * similarity * connection_probability)
                adjusted_prob = min(adjusted_prob, 0.9)  # Cap at 90%

                if np.random.random() < adjusted_prob:
                    self._add_connection(agent1.id, agent2.id)

    def create_small_world_network(self, k: int = 6, p: float = 0.1) -> None:
        """
        Create a small-world network using Watts-Strogatz model.

        Args:
            k: Number of nearest neighbors to connect initially
            p: Probability of rewiring each edge
        """
        agent_ids = list(self.agents.keys())
        n_agents = len(agent_ids)

        if n_agents < k + 1:
            k = max(1, n_agents - 1)

        # Create regular ring lattice
        for i, agent_id in enumerate(agent_ids):
            for j in range(1, k // 2 + 1):
                neighbor_idx = (i + j) % n_agents
                self._add_connection(agent_id, agent_ids[neighbor_idx])

        # Rewire edges with probability p
        edges_to_rewire = []
        for edge in list(self.graph.edges()):
            if np.random.random() < p:
                edges_to_rewire.append(edge)

        for edge in edges_to_rewire:
            self.graph.remove_edge(*edge)
            # Remove from agent connections
            self.agents[edge[0]].remove_connection(edge[1])
            self.agents[edge[1]].remove_connection(edge[0])

            # Add new random connection
            available_targets = [aid for aid in agent_ids if aid != edge[0] and
                               aid not in self.agents[edge[0]].connections]
            if available_targets:
                new_target = np.random.choice(available_targets)
                self._add_connection(edge[0], new_target)

    def create_preferential_attachment_network(self, m: int = 3) -> None:
        """
        Create network using preferential attachment (Barabási-Albert model).

        Args:
            m: Number of edges to attach from a new node to existing nodes
        """
        agent_ids = list(self.agents.keys())

        # Start with a small complete graph
        initial_size = min(m + 1, len(agent_ids))
        for i in range(initial_size):
            for j in range(i + 1, initial_size):
                self._add_connection(agent_ids[i], agent_ids[j])

        # Add remaining nodes with preferential attachment
        for i in range(initial_size, len(agent_ids)):
            new_agent_id = agent_ids[i]

            # Calculate attachment probabilities based on current degrees
            degrees = dict(self.graph.degree())
            total_degree = sum(degrees.values())

            if total_degree == 0:
                # Fallback: connect to random existing nodes
                existing_agents = agent_ids[:i]
                targets = np.random.choice(existing_agents, min(m, len(existing_agents)), replace=False)
            else:
                # Preferential attachment
                probabilities = [degrees[aid] / total_degree for aid in agent_ids[:i]]
                targets = np.random.choice(agent_ids[:i], min(m, i), p=probabilities, replace=False)

            for target_id in targets:
                self._add_connection(new_agent_id, target_id)

    def _add_connection(self, agent1_id: str, agent2_id: str) -> None:
        """Add bidirectional connection between two agents."""
        self.graph.add_edge(agent1_id, agent2_id)
        self.agents[agent1_id].add_connection(agent2_id)
        self.agents[agent2_id].add_connection(agent1_id)

    def _calculate_demographic_similarity(self, persona1: Persona, persona2: Persona) -> float:
        """Calculate similarity between two personas based on demographics."""
        demo1, demo2 = persona1.demographics, persona2.demographics

        similarities = []

        # Age similarity
        age_diff = abs(demo1.age - demo2.age)
        age_similarity = max(0, 1 - age_diff / 50)  # Normalize by max reasonable age difference
        similarities.append(age_similarity)

        # Categorical similarities
        categorical_matches = [
            demo1.gender == demo2.gender,
            demo1.education == demo2.education,
            demo1.income == demo2.income,
            demo1.region == demo2.region,
            demo1.urban == demo2.urban
        ]
        similarities.extend([1.0 if match else 0.0 for match in categorical_matches])

        return np.mean(similarities)

    def get_neighbors(self, agent_id: str) -> List[str]:
        """Get the neighbor IDs of an agent."""
        return list(self.graph.neighbors(agent_id))

    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the network structure."""
        return {
            "n_agents": len(self.agents),
            "n_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "average_degree": np.mean([d for n, d in self.graph.degree()]),
            "clustering_coefficient": nx.average_clustering(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "n_components": nx.number_connected_components(self.graph)
        }

    def get_agent_centralities(self) -> Dict[str, Dict[str, float]]:
        """Calculate centrality measures for all agents."""
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)

        centralities = {}
        for agent_id in self.agents.keys():
            centralities[agent_id] = {
                "degree": degree_centrality[agent_id],
                "betweenness": betweenness_centrality[agent_id],
                "closeness": closeness_centrality[agent_id]
            }

        return centralities

    def visualize_network(self, layout: str = "spring", figsize: Tuple[int, int] = (12, 8)) -> Any:
        """
        Visualize the network (requires matplotlib).

        Returns matplotlib figure object.
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize)

            # Choose layout
            if layout == "spring":
                pos = nx.spring_layout(self.graph)
            elif layout == "circular":
                pos = nx.circular_layout(self.graph)
            elif layout == "random":
                pos = nx.random_layout(self.graph)
            else:
                pos = nx.spring_layout(self.graph)

            # Color nodes by some demographic characteristic
            node_colors = []
            for agent_id in self.graph.nodes():
                agent = self.agents[agent_id]
                # Color by education level
                education_map = {"high_school": 0.2, "college": 0.4, "university": 0.6, "graduate": 0.8}
                color = education_map.get(agent.persona.demographics.education, 0.5)
                node_colors.append(color)

            nx.draw(self.graph, pos, node_color=node_colors, node_size=50,
                   with_labels=False, alpha=0.7, ax=ax)

            ax.set_title("Social Network Visualization")
            plt.tight_layout()

            return fig

        except ImportError:
            print("matplotlib not available for visualization")
            return None

    def get_influence_flows(self, topic: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get influence flow information for a specific topic.

        Returns dictionary mapping agent IDs to their influence relationships.
        """
        influence_flows = {}

        for agent_id, agent in self.agents.items():
            flows = []
            for neighbor_id in self.get_neighbors(agent_id):
                neighbor = self.agents[neighbor_id]

                # Get opinions
                agent_opinion, _ = agent.get_opinion(topic)
                neighbor_opinion, _ = neighbor.get_opinion(topic)

                flows.append({
                    "target_agent": neighbor_id,
                    "agent_opinion": agent_opinion,
                    "target_opinion": neighbor_opinion,
                    "opinion_difference": abs(agent_opinion - neighbor_opinion),
                    "agent_influence": agent.influence_strength,
                    "target_susceptibility": neighbor.susceptibility
                })

            influence_flows[agent_id] = flows

        return influence_flows