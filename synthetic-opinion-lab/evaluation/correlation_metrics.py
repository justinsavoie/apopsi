import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform
import warnings


class CorrelationComparator:
    """Compares correlation structures between real and synthetic data."""

    def __init__(self):
        self.results = {}

    def compare_correlation_matrices(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                   method: str = "pearson", variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare correlation matrices between real and synthetic data.

        Args:
            real_df: Real dataset
            synthetic_df: Synthetic dataset
            method: Correlation method ("pearson", "spearman", "kendall")
            variables: Specific variables to include (if None, use common variables)

        Returns:
            Dictionary with correlation comparison results
        """
        # Identify variables to analyze
        if variables is None:
            common_vars = list(set(real_df.columns) & set(synthetic_df.columns))
            # Remove non-numeric variables for correlation analysis
            numeric_vars = []
            for var in common_vars:
                try:
                    pd.to_numeric(real_df[var], errors='coerce')
                    pd.to_numeric(synthetic_df[var], errors='coerce')
                    numeric_vars.append(var)
                except:
                    continue
            variables = numeric_vars

        if len(variables) < 2:
            return {"error": "Need at least 2 numeric variables for correlation analysis"}

        # Extract numeric data
        real_numeric = real_df[variables].apply(pd.to_numeric, errors='coerce')
        synthetic_numeric = synthetic_df[variables].apply(pd.to_numeric, errors='coerce')

        # Calculate correlation matrices
        if method == "pearson":
            real_corr = real_numeric.corr(method="pearson")
            synthetic_corr = synthetic_numeric.corr(method="pearson")
        elif method == "spearman":
            real_corr = real_numeric.corr(method="spearman")
            synthetic_corr = synthetic_numeric.corr(method="spearman")
        else:
            raise ValueError(f"Unsupported correlation method: {method}")

        # Handle missing values
        real_corr = real_corr.fillna(0)
        synthetic_corr = synthetic_corr.fillna(0)

        results = {
            "method": method,
            "variables": variables,
            "real_correlation_matrix": real_corr.to_dict(),
            "synthetic_correlation_matrix": synthetic_corr.to_dict(),
        }

        # Compare correlation matrices
        correlation_comparison = self._compare_matrices(real_corr, synthetic_corr)
        results.update(correlation_comparison)

        return results

    def _compare_matrices(self, real_corr: pd.DataFrame, synthetic_corr: pd.DataFrame) -> Dict[str, Any]:
        """Compare two correlation matrices."""
        real_vals = real_corr.values
        synthetic_vals = synthetic_corr.values

        # Extract upper triangle (excluding diagonal) for comparison
        mask = np.triu(np.ones_like(real_vals, dtype=bool), k=1)
        real_upper = real_vals[mask]
        synthetic_upper = synthetic_vals[mask]

        # Remove any remaining NaN values
        valid_mask = ~(np.isnan(real_upper) | np.isnan(synthetic_upper))
        real_upper = real_upper[valid_mask]
        synthetic_upper = synthetic_upper[valid_mask]

        if len(real_upper) == 0:
            return {"error": "No valid correlation pairs to compare"}

        results = {}

        # Matrix difference metrics
        diff_matrix = synthetic_vals - real_vals
        results["mean_absolute_difference"] = float(np.nanmean(np.abs(diff_matrix)))
        results["max_absolute_difference"] = float(np.nanmax(np.abs(diff_matrix)))
        results["root_mean_square_error"] = float(np.sqrt(np.nanmean(diff_matrix**2)))

        # Correlation between correlation values
        if len(real_upper) > 1:
            corr_of_corrs, p_value = spearmanr(real_upper, synthetic_upper)
            results["correlation_of_correlations"] = float(corr_of_corrs)
            results["correlation_p_value"] = float(p_value)
        else:
            results["correlation_of_correlations"] = None
            results["correlation_p_value"] = None

        # Frobenius norm of difference
        results["frobenius_norm_difference"] = float(np.linalg.norm(diff_matrix, 'fro'))

        # Element-wise comparison statistics
        results["correlation_comparison_stats"] = {
            "real_mean": float(np.mean(real_upper)),
            "real_std": float(np.std(real_upper)),
            "synthetic_mean": float(np.mean(synthetic_upper)),
            "synthetic_std": float(np.std(synthetic_upper)),
            "difference_mean": float(np.mean(synthetic_upper - real_upper)),
            "difference_std": float(np.std(synthetic_upper - real_upper))
        }

        return results

    def analyze_correlation_preservation(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                       correlation_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Analyze how well strong correlations are preserved.

        Args:
            real_df: Real dataset
            synthetic_df: Synthetic dataset
            correlation_threshold: Threshold for considering a correlation "strong"

        Returns:
            Analysis of correlation preservation
        """
        pearson_comparison = self.compare_correlation_matrices(real_df, synthetic_df, "pearson")

        if "error" in pearson_comparison:
            return pearson_comparison

        real_corr = pd.DataFrame(pearson_comparison["real_correlation_matrix"])
        synthetic_corr = pd.DataFrame(pearson_comparison["synthetic_correlation_matrix"])

        # Find strong correlations in real data
        real_vals = real_corr.values
        synthetic_vals = synthetic_corr.values

        # Get upper triangle excluding diagonal
        mask = np.triu(np.ones_like(real_vals, dtype=bool), k=1)
        real_upper = real_vals[mask]
        synthetic_upper = synthetic_vals[mask]

        # Identify strong correlations
        strong_real_mask = np.abs(real_upper) >= correlation_threshold
        strong_correlations_real = real_upper[strong_real_mask]
        strong_correlations_synthetic = synthetic_upper[strong_real_mask]

        results = {
            "correlation_threshold": correlation_threshold,
            "total_correlation_pairs": len(real_upper),
            "strong_correlations_count": len(strong_correlations_real),
            "strong_correlations_fraction": len(strong_correlations_real) / len(real_upper) if len(real_upper) > 0 else 0
        }

        if len(strong_correlations_real) > 0:
            # How well are strong correlations preserved?
            correlation_preservation = spearmanr(strong_correlations_real, strong_correlations_synthetic)[0]
            results["strong_correlation_preservation"] = float(correlation_preservation)

            # Average difference for strong correlations
            strong_differences = np.abs(strong_correlations_synthetic - strong_correlations_real)
            results["strong_correlation_mean_difference"] = float(np.mean(strong_differences))
            results["strong_correlation_max_difference"] = float(np.max(strong_differences))

        else:
            results["strong_correlation_preservation"] = None
            results["strong_correlation_mean_difference"] = None
            results["strong_correlation_max_difference"] = None

        return results

    def compare_pairwise_relationships(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                     variable_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Compare specific pairwise relationships between variables.

        Args:
            real_df: Real dataset
            synthetic_df: Synthetic dataset
            variable_pairs: List of (var1, var2) tuples to analyze

        Returns:
            Detailed comparison of pairwise relationships
        """
        results = {"pairwise_comparisons": {}}

        for var1, var2 in variable_pairs:
            if var1 not in real_df.columns or var2 not in real_df.columns:
                continue
            if var1 not in synthetic_df.columns or var2 not in synthetic_df.columns:
                continue

            pair_key = f"{var1}_vs_{var2}"

            # Extract data
            real_x = pd.to_numeric(real_df[var1], errors='coerce')
            real_y = pd.to_numeric(real_df[var2], errors='coerce')
            synthetic_x = pd.to_numeric(synthetic_df[var1], errors='coerce')
            synthetic_y = pd.to_numeric(synthetic_df[var2], errors='coerce')

            # Remove missing values
            real_mask = ~(real_x.isna() | real_y.isna())
            synthetic_mask = ~(synthetic_x.isna() | synthetic_y.isna())

            real_x_clean = real_x[real_mask]
            real_y_clean = real_y[real_mask]
            synthetic_x_clean = synthetic_x[synthetic_mask]
            synthetic_y_clean = synthetic_y[synthetic_mask]

            if len(real_x_clean) < 2 or len(synthetic_x_clean) < 2:
                results["pairwise_comparisons"][pair_key] = {"error": "Insufficient data"}
                continue

            pair_results = {}

            # Pearson correlations
            try:
                real_pearson = real_x_clean.corr(real_y_clean)
                synthetic_pearson = synthetic_x_clean.corr(synthetic_y_clean)
                pair_results["pearson_real"] = float(real_pearson) if not np.isnan(real_pearson) else 0
                pair_results["pearson_synthetic"] = float(synthetic_pearson) if not np.isnan(synthetic_pearson) else 0
                pair_results["pearson_difference"] = pair_results["pearson_synthetic"] - pair_results["pearson_real"]
            except:
                pair_results["pearson_real"] = None
                pair_results["pearson_synthetic"] = None
                pair_results["pearson_difference"] = None

            # Spearman correlations
            try:
                real_spearman, _ = spearmanr(real_x_clean, real_y_clean)
                synthetic_spearman, _ = spearmanr(synthetic_x_clean, synthetic_y_clean)
                pair_results["spearman_real"] = float(real_spearman) if not np.isnan(real_spearman) else 0
                pair_results["spearman_synthetic"] = float(synthetic_spearman) if not np.isnan(synthetic_spearman) else 0
                pair_results["spearman_difference"] = pair_results["spearman_synthetic"] - pair_results["spearman_real"]
            except:
                pair_results["spearman_real"] = None
                pair_results["spearman_synthetic"] = None
                pair_results["spearman_difference"] = None

            results["pairwise_comparisons"][pair_key] = pair_results

        return results

    def create_correlation_summary(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create comprehensive correlation comparison summary.

        Args:
            real_df: Real dataset
            synthetic_df: Synthetic dataset

        Returns:
            Comprehensive correlation analysis summary
        """
        summary = {}

        # Basic correlation matrix comparison
        pearson_comparison = self.compare_correlation_matrices(real_df, synthetic_df, "pearson")
        summary["pearson_comparison"] = pearson_comparison

        if "error" not in pearson_comparison:
            spearman_comparison = self.compare_correlation_matrices(real_df, synthetic_df, "spearman")
            summary["spearman_comparison"] = spearman_comparison

            # Correlation preservation analysis
            preservation_analysis = self.analyze_correlation_preservation(real_df, synthetic_df)
            summary["correlation_preservation"] = preservation_analysis

            # Overall quality score (simple heuristic)
            quality_score = self._calculate_correlation_quality_score(pearson_comparison, preservation_analysis)
            summary["correlation_quality_score"] = quality_score

        return summary

    def _calculate_correlation_quality_score(self, pearson_comparison: Dict[str, Any],
                                           preservation_analysis: Dict[str, Any]) -> float:
        """Calculate a simple quality score for correlation preservation."""
        try:
            # Base score on correlation of correlations and RMSE
            corr_of_corrs = pearson_comparison.get("correlation_of_correlations", 0)
            rmse = pearson_comparison.get("root_mean_square_error", 1)

            # Normalize RMSE (assuming correlations range from -1 to 1)
            normalized_rmse = min(rmse / 2.0, 1.0)

            # Base quality: high correlation of correlations, low RMSE
            base_quality = max(0, corr_of_corrs) * (1 - normalized_rmse)

            # Bonus for preserving strong correlations
            strong_preservation = preservation_analysis.get("strong_correlation_preservation", 0)
            if strong_preservation is not None and strong_preservation > 0:
                preservation_bonus = 0.2 * strong_preservation
            else:
                preservation_bonus = 0

            quality_score = min(1.0, base_quality + preservation_bonus)
            return float(quality_score)

        except:
            return 0.0