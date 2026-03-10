import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
from scipy.stats import ks_2samp, chi2_contingency


def kullback_leibler_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate KL divergence between two probability distributions.

    Args:
        p: Target distribution
        q: Approximate distribution
        epsilon: Small constant to avoid log(0)

    Returns:
        KL divergence D(P||Q)
    """
    # Normalize to ensure they are probability distributions
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Calculate KL divergence
    return np.sum(p * np.log(p / q))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        JS divergence (symmetric, always finite)
    """
    p = np.array(p)
    q = np.array(q)

    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Calculate M = 0.5 * (P + Q)
    m = 0.5 * (p + q)

    # JS divergence = 0.5 * (KL(P||M) + KL(Q||M))
    js_div = 0.5 * (kullback_leibler_divergence(p, m) + kullback_leibler_divergence(q, m))

    return js_div


def earth_movers_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Earth Mover's Distance (Wasserstein-1) between two 1D distributions.

    Args:
        x: Samples from first distribution
        y: Samples from second distribution

    Returns:
        Wasserstein distance
    """
    return wasserstein_distance(x, y)


class DistributionComparator:
    """Compares distributions between real and synthetic data."""

    def __init__(self):
        self.results = {}

    def compare_categorical_distributions(self, real_data: pd.Series, synthetic_data: pd.Series,
                                        variable_name: str) -> Dict[str, Any]:
        """
        Compare categorical variable distributions.

        Args:
            real_data: Real survey responses for a categorical variable
            synthetic_data: Synthetic responses for the same variable
            variable_name: Name of the variable

        Returns:
            Dictionary with comparison metrics
        """
        # Get value counts
        real_counts = real_data.value_counts().sort_index()
        synthetic_counts = synthetic_data.value_counts().sort_index()

        # Align categories (handle missing categories)
        all_categories = sorted(set(real_counts.index) | set(synthetic_counts.index))
        real_aligned = np.array([real_counts.get(cat, 0) for cat in all_categories])
        synthetic_aligned = np.array([synthetic_counts.get(cat, 0) for cat in all_categories])

        # Convert to probabilities
        real_probs = real_aligned / np.sum(real_aligned)
        synthetic_probs = synthetic_aligned / np.sum(synthetic_aligned)

        # Calculate metrics
        results = {
            "variable": variable_name,
            "type": "categorical",
            "categories": all_categories,
            "real_distribution": real_probs.tolist(),
            "synthetic_distribution": synthetic_probs.tolist(),
            "real_counts": real_aligned.tolist(),
            "synthetic_counts": synthetic_aligned.tolist(),
        }

        # KL divergence (both directions)
        results["kl_divergence_real_to_synthetic"] = kullback_leibler_divergence(real_probs, synthetic_probs)
        results["kl_divergence_synthetic_to_real"] = kullback_leibler_divergence(synthetic_probs, real_probs)

        # Jensen-Shannon divergence (symmetric)
        results["jensen_shannon_divergence"] = jensen_shannon_divergence(real_probs, synthetic_probs)

        # Chi-square test
        if np.all(real_aligned > 0) and np.all(synthetic_aligned > 0):
            chi2_stat, p_value = chi2_contingency([real_aligned, synthetic_aligned])[:2]
            results["chi_square_statistic"] = chi2_stat
            results["chi_square_p_value"] = p_value
        else:
            results["chi_square_statistic"] = None
            results["chi_square_p_value"] = None

        # Total variation distance
        results["total_variation_distance"] = 0.5 * np.sum(np.abs(real_probs - synthetic_probs))

        return results

    def compare_continuous_distributions(self, real_data: pd.Series, synthetic_data: pd.Series,
                                       variable_name: str) -> Dict[str, Any]:
        """
        Compare continuous variable distributions.

        Args:
            real_data: Real survey responses for a continuous variable
            synthetic_data: Synthetic responses for the same variable
            variable_name: Name of the variable

        Returns:
            Dictionary with comparison metrics
        """
        real_values = real_data.dropna().values
        synthetic_values = synthetic_data.dropna().values

        results = {
            "variable": variable_name,
            "type": "continuous",
            "real_n": len(real_values),
            "synthetic_n": len(synthetic_values),
        }

        # Basic statistics
        results["real_mean"] = float(np.mean(real_values))
        results["real_std"] = float(np.std(real_values))
        results["real_median"] = float(np.median(real_values))
        results["real_min"] = float(np.min(real_values))
        results["real_max"] = float(np.max(real_values))

        results["synthetic_mean"] = float(np.mean(synthetic_values))
        results["synthetic_std"] = float(np.std(synthetic_values))
        results["synthetic_median"] = float(np.median(synthetic_values))
        results["synthetic_min"] = float(np.min(synthetic_values))
        results["synthetic_max"] = float(np.max(synthetic_values))

        # Statistical tests
        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = ks_2samp(real_values, synthetic_values)
        results["ks_statistic"] = ks_stat
        results["ks_p_value"] = ks_p_value

        # Wasserstein (Earth Mover's) distance
        results["wasserstein_distance"] = earth_movers_distance(real_values, synthetic_values)

        # Difference in moments
        results["mean_difference"] = results["synthetic_mean"] - results["real_mean"]
        results["std_difference"] = results["synthetic_std"] - results["real_std"]
        results["median_difference"] = results["synthetic_median"] - results["real_median"]

        return results

    def compare_ordinal_distributions(self, real_data: pd.Series, synthetic_data: pd.Series,
                                    variable_name: str, ordered_categories: List[Any]) -> Dict[str, Any]:
        """
        Compare ordinal variable distributions (e.g., Likert scales).

        Args:
            real_data: Real survey responses
            synthetic_data: Synthetic responses
            variable_name: Name of the variable
            ordered_categories: List of categories in order

        Returns:
            Dictionary with comparison metrics
        """
        # Map categories to numeric values for ordinal comparison
        category_map = {cat: i for i, cat in enumerate(ordered_categories)}

        real_numeric = real_data.map(category_map).dropna()
        synthetic_numeric = synthetic_data.map(category_map).dropna()

        # Start with categorical comparison
        results = self.compare_categorical_distributions(real_data, synthetic_data, variable_name)
        results["type"] = "ordinal"
        results["ordered_categories"] = ordered_categories

        # Add ordinal-specific metrics
        if len(real_numeric) > 0 and len(synthetic_numeric) > 0:
            # Treat as continuous for some metrics
            continuous_results = self.compare_continuous_distributions(
                real_numeric, synthetic_numeric, f"{variable_name}_numeric"
            )

            # Add relevant continuous metrics
            results.update({
                "real_mean_ordinal": continuous_results["real_mean"],
                "synthetic_mean_ordinal": continuous_results["synthetic_mean"],
                "mean_difference_ordinal": continuous_results["mean_difference"],
                "ks_statistic": continuous_results["ks_statistic"],
                "ks_p_value": continuous_results["ks_p_value"],
                "wasserstein_distance": continuous_results["wasserstein_distance"]
            })

        return results

    def compare_dataset_distributions(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                    variable_types: Optional[Dict[str, str]] = None,
                                    ordinal_mappings: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Compare distributions for all variables in two datasets.

        Args:
            real_df: Real dataset
            synthetic_df: Synthetic dataset
            variable_types: Dict mapping variable names to types ("categorical", "continuous", "ordinal")
            ordinal_mappings: Dict mapping ordinal variable names to ordered categories

        Returns:
            Dictionary with all comparison results
        """
        variable_types = variable_types or {}
        ordinal_mappings = ordinal_mappings or {}

        results = {
            "overall_summary": {
                "real_n": len(real_df),
                "synthetic_n": len(synthetic_df),
                "common_variables": [],
                "real_only_variables": [],
                "synthetic_only_variables": []
            },
            "variable_comparisons": {}
        }

        # Identify common variables
        real_vars = set(real_df.columns)
        synthetic_vars = set(synthetic_df.columns)
        common_vars = real_vars & synthetic_vars

        results["overall_summary"]["common_variables"] = sorted(list(common_vars))
        results["overall_summary"]["real_only_variables"] = sorted(list(real_vars - synthetic_vars))
        results["overall_summary"]["synthetic_only_variables"] = sorted(list(synthetic_vars - real_vars))

        # Compare each common variable
        for var in common_vars:
            var_type = variable_types.get(var, "categorical")  # Default to categorical

            if var_type == "categorical":
                comparison = self.compare_categorical_distributions(
                    real_df[var], synthetic_df[var], var
                )
            elif var_type == "continuous":
                comparison = self.compare_continuous_distributions(
                    real_df[var], synthetic_df[var], var
                )
            elif var_type == "ordinal":
                ordered_cats = ordinal_mappings.get(var, sorted(real_df[var].unique()))
                comparison = self.compare_ordinal_distributions(
                    real_df[var], synthetic_df[var], var, ordered_cats
                )
            else:
                # Fallback to categorical
                comparison = self.compare_categorical_distributions(
                    real_df[var], synthetic_df[var], var
                )

            results["variable_comparisons"][var] = comparison

        # Overall quality metrics
        quality_metrics = self._calculate_overall_quality(results["variable_comparisons"])
        results["overall_quality"] = quality_metrics

        return results

    def _calculate_overall_quality(self, variable_comparisons: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall quality metrics across all variables."""
        js_divergences = []
        tv_distances = []
        ks_p_values = []

        for var, comparison in variable_comparisons.items():
            if "jensen_shannon_divergence" in comparison:
                js_divergences.append(comparison["jensen_shannon_divergence"])

            if "total_variation_distance" in comparison:
                tv_distances.append(comparison["total_variation_distance"])

            if "ks_p_value" in comparison and comparison["ks_p_value"] is not None:
                ks_p_values.append(comparison["ks_p_value"])

        quality_metrics = {}

        if js_divergences:
            quality_metrics["mean_jensen_shannon_divergence"] = float(np.mean(js_divergences))
            quality_metrics["max_jensen_shannon_divergence"] = float(np.max(js_divergences))

        if tv_distances:
            quality_metrics["mean_total_variation_distance"] = float(np.mean(tv_distances))
            quality_metrics["max_total_variation_distance"] = float(np.max(tv_distances))

        if ks_p_values:
            quality_metrics["mean_ks_p_value"] = float(np.mean(ks_p_values))
            quality_metrics["fraction_significant_ks_tests"] = float(np.mean([p < 0.05 for p in ks_p_values]))

        return quality_metrics