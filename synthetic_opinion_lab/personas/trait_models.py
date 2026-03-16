import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import multivariate_normal
from .persona import LatentTraits


@dataclass
class TraitModel:
    """Configuration for trait generation."""
    trait_names: List[str]
    means: np.ndarray
    covariance_matrix: np.ndarray

    def __post_init__(self):
        """Validate dimensions match."""
        n_traits = len(self.trait_names)
        if len(self.means) != n_traits:
            raise ValueError(f"Means length {len(self.means)} != trait count {n_traits}")
        if self.covariance_matrix.shape != (n_traits, n_traits):
            raise ValueError(f"Covariance matrix shape {self.covariance_matrix.shape} != ({n_traits}, {n_traits})")

    @classmethod
    def default_political_model(cls) -> 'TraitModel':
        """Default political trait model with realistic correlations."""
        trait_names = ["ideology", "authoritarianism", "trust_in_government", "populism", "cosmopolitanism"]

        # Centered means
        means = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Realistic covariance structure
        # ideology and cosmopolitanism are positively correlated
        # authoritarianism and populism are positively correlated
        # trust_in_government has weak correlations
        cov_matrix = np.array([
            [1.0,  -0.3,  0.1,  -0.2,  0.6],  # ideology
            [-0.3,  1.0,  0.2,   0.4, -0.3],  # authoritarianism
            [0.1,   0.2,  1.0,  -0.3,  0.1],  # trust_in_government
            [-0.2,  0.4, -0.3,   1.0, -0.4],  # populism
            [0.6,  -0.3,  0.1,  -0.4,  1.0]   # cosmopolitanism
        ])

        return cls(trait_names=trait_names, means=means, covariance_matrix=cov_matrix)

    @classmethod
    def uncorrelated_model(cls) -> 'TraitModel':
        """Independent trait model (identity covariance)."""
        trait_names = ["ideology", "authoritarianism", "trust_in_government", "populism", "cosmopolitanism"]
        means = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        cov_matrix = np.eye(5)

        return cls(trait_names=trait_names, means=means, covariance_matrix=cov_matrix)


class TraitGenerator:
    """Generates latent traits for personas."""

    def __init__(self, trait_model: TraitModel):
        self.model = trait_model
        self.distribution = multivariate_normal(
            mean=trait_model.means,
            cov=trait_model.covariance_matrix
        )

    def generate_traits(self, n_personas: int, random_state: Optional[int] = None) -> List[LatentTraits]:
        """Generate latent traits for n personas."""
        if random_state is not None:
            np.random.seed(random_state)

        # Sample from multivariate normal
        samples = self.distribution.rvs(size=n_personas)

        # Ensure 2D array even for single sample
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        # Clip to valid range [-2, 2] and create LatentTraits objects
        traits_list = []
        for sample in samples:
            clipped_sample = np.clip(sample, -2, 2)

            traits = LatentTraits(
                ideology=float(clipped_sample[0]),
                authoritarianism=float(clipped_sample[1]),
                trust_in_government=float(clipped_sample[2]),
                populism=float(clipped_sample[3]),
                cosmopolitanism=float(clipped_sample[4])
            )
            traits_list.append(traits)

        return traits_list

    def get_correlation_matrix(self) -> np.ndarray:
        """Get the correlation matrix from the covariance matrix."""
        cov = self.model.covariance_matrix
        std_devs = np.sqrt(np.diag(cov))
        corr_matrix = cov / np.outer(std_devs, std_devs)
        return corr_matrix

    def sample_single_trait(self, trait_name: str, conditional_traits: Optional[Dict[str, float]] = None) -> float:
        """Sample a single trait, optionally conditional on other traits."""
        if conditional_traits is None:
            # Marginal sampling
            trait_idx = self.model.trait_names.index(trait_name)
            mean = self.model.means[trait_idx]
            var = self.model.covariance_matrix[trait_idx, trait_idx]
            value = np.random.normal(mean, np.sqrt(var))
        else:
            # Conditional sampling (simplified - assumes multivariate normal)
            # In practice, you might want more sophisticated conditional sampling
            raise NotImplementedError("Conditional sampling not yet implemented")

        return float(np.clip(value, -2, 2))