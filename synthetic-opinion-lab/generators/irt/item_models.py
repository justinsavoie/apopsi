import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from scipy.special import expit  # logistic function


@dataclass
class ItemParameters:
    """Parameters for an IRT item."""
    discrimination: float  # alpha parameter
    difficulty: float  # beta parameter (or threshold for binary)
    thresholds: Optional[List[float]] = None  # for graded response model

    def __post_init__(self):
        if self.discrimination <= 0:
            raise ValueError("Discrimination parameter must be positive")


class IRTModel:
    """Base class for IRT models."""

    def __init__(self, item_params: ItemParameters):
        self.item_params = item_params

    def probability(self, theta: float, response: int) -> float:
        """Calculate probability of response given theta."""
        raise NotImplementedError


class TwoPLModel(IRTModel):
    """Two-parameter logistic model for binary items."""

    def probability(self, theta: float, response: int) -> float:
        """
        Calculate P(X=1|theta) for binary response.

        P(X=1|theta) = logistic(alpha * (theta - beta))
        """
        if response not in [0, 1]:
            raise ValueError("Binary response must be 0 or 1")

        alpha = self.item_params.discrimination
        beta = self.item_params.difficulty

        p_correct = expit(alpha * (theta - beta))

        return p_correct if response == 1 else (1 - p_correct)

    def sample_response(self, theta: float, random_state: Optional[int] = None) -> int:
        """Sample a binary response given theta."""
        if random_state is not None:
            np.random.seed(random_state)

        p_correct = self.probability(theta, 1)
        return int(np.random.random() < p_correct)


class GradedResponseModel(IRTModel):
    """Graded response model for ordinal items (e.g., Likert scales)."""

    def __init__(self, item_params: ItemParameters):
        super().__init__(item_params)
        if item_params.thresholds is None:
            raise ValueError("Graded response model requires threshold parameters")

        # Verify thresholds are in ascending order
        thresholds = item_params.thresholds
        if not all(thresholds[i] <= thresholds[i+1] for i in range(len(thresholds)-1)):
            raise ValueError("Thresholds must be in ascending order")

    def probability(self, theta: float, response: int) -> float:
        """
        Calculate P(X=k|theta) for ordinal response k.

        Uses cumulative probability differences.
        """
        alpha = self.item_params.discrimination
        thresholds = self.item_params.thresholds

        if response < 0 or response >= len(thresholds) + 1:
            raise ValueError(f"Response must be in range [0, {len(thresholds)}]")

        # Calculate cumulative probabilities
        # P*(k) = P(X >= k | theta)
        cum_probs = []
        for threshold in thresholds:
            cum_prob = expit(alpha * (theta - threshold))
            cum_probs.append(cum_prob)

        # Add boundary probabilities
        cum_probs = [1.0] + cum_probs + [0.0]

        # Category probability = P*(k) - P*(k+1)
        prob = cum_probs[response] - cum_probs[response + 1]

        return max(prob, 1e-10)  # Avoid numerical issues

    def sample_response(self, theta: float, random_state: Optional[int] = None) -> int:
        """Sample an ordinal response given theta."""
        if random_state is not None:
            np.random.seed(random_state)

        alpha = self.item_params.discrimination
        thresholds = self.item_params.thresholds

        # Calculate cumulative probabilities
        cum_probs = [expit(alpha * (theta - t)) for t in thresholds]

        # Sample response category
        random_value = np.random.random()

        # Find the response category
        for k in range(len(thresholds) + 1):
            if k == 0:
                prob_k = 1.0 - cum_probs[0]
            elif k == len(thresholds):
                prob_k = cum_probs[-1]
            else:
                prob_k = cum_probs[k-1] - cum_probs[k]

            if random_value <= prob_k:
                return k
            random_value -= prob_k

        return len(thresholds)  # fallback


class ItemParameterGenerator:
    """Generates realistic item parameters for IRT models."""

    @staticmethod
    def generate_binary_items(n_items: int,
                            discrimination_range: tuple = (0.5, 2.5),
                            difficulty_range: tuple = (-2.0, 2.0),
                            random_state: Optional[int] = None) -> List[ItemParameters]:
        """Generate parameters for binary items."""
        if random_state is not None:
            np.random.seed(random_state)

        items = []
        for _ in range(n_items):
            discrimination = np.random.uniform(*discrimination_range)
            difficulty = np.random.uniform(*difficulty_range)
            items.append(ItemParameters(discrimination, difficulty))

        return items

    @staticmethod
    def generate_likert_items(n_items: int,
                            n_categories: int = 5,
                            discrimination_range: tuple = (0.5, 2.5),
                            threshold_spacing: float = 1.5,
                            random_state: Optional[int] = None) -> List[ItemParameters]:
        """Generate parameters for Likert-scale items."""
        if random_state is not None:
            np.random.seed(random_state)

        items = []
        for _ in range(n_items):
            discrimination = np.random.uniform(*discrimination_range)

            # Generate ordered thresholds
            base_threshold = np.random.uniform(-2.0, 2.0)
            thresholds = []
            for k in range(n_categories - 1):
                threshold = base_threshold + (k - (n_categories-2)/2) * threshold_spacing
                thresholds.append(threshold)

            # Ensure proper ordering with some random variation
            thresholds = sorted(thresholds)

            items.append(ItemParameters(
                discrimination=discrimination,
                difficulty=np.mean(thresholds),  # center point
                thresholds=thresholds
            ))

        return items