import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os

from .distribution_metrics import DistributionComparator
from .correlation_metrics import CorrelationComparator
from .regression_tests import RegressionReplicator


class EvaluationPipeline:
    """Comprehensive evaluation pipeline for synthetic survey data quality."""

    def __init__(self):
        self.distribution_comparator = DistributionComparator()
        self.correlation_comparator = CorrelationComparator()
        self.regression_replicator = RegressionReplicator()
        self.results = {}

    def run_full_evaluation(self,
                          real_df: pd.DataFrame,
                          synthetic_df: pd.DataFrame,
                          variable_types: Optional[Dict[str, str]] = None,
                          ordinal_mappings: Optional[Dict[str, List[Any]]] = None,
                          regression_tests: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation comparing real and synthetic data.

        Args:
            real_df: Real survey data
            synthetic_df: Synthetic survey data
            variable_types: Dict mapping variables to types ('categorical', 'continuous', 'ordinal')
            ordinal_mappings: Dict mapping ordinal variables to ordered categories
            regression_tests: List of regression test specifications

        Returns:
            Comprehensive evaluation results
        """
        print("Running comprehensive evaluation...")

        evaluation_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "real_data_shape": real_df.shape,
                "synthetic_data_shape": synthetic_df.shape,
                "variable_types": variable_types or {},
                "ordinal_mappings": ordinal_mappings or {}
            }
        }

        # 1. Distribution Analysis
        print("  Analyzing distributions...")
        try:
            distribution_results = self.distribution_comparator.compare_dataset_distributions(
                real_df, synthetic_df, variable_types, ordinal_mappings
            )
            evaluation_results["distribution_analysis"] = distribution_results
        except Exception as e:
            evaluation_results["distribution_analysis"] = {"error": str(e)}

        # 2. Correlation Analysis
        print("  Analyzing correlations...")
        try:
            correlation_results = self.correlation_comparator.create_correlation_summary(
                real_df, synthetic_df
            )
            evaluation_results["correlation_analysis"] = correlation_results
        except Exception as e:
            evaluation_results["correlation_analysis"] = {"error": str(e)}

        # 3. Regression Analysis
        if regression_tests:
            print("  Running regression tests...")
            try:
                regression_results = self.regression_replicator.run_regression_test_suite(
                    real_df, synthetic_df, regression_tests
                )
                evaluation_results["regression_analysis"] = regression_results
            except Exception as e:
                evaluation_results["regression_analysis"] = {"error": str(e)}

        # 4. Overall Quality Assessment
        print("  Calculating overall quality metrics...")
        overall_quality = self._calculate_overall_quality(evaluation_results)
        evaluation_results["overall_quality"] = overall_quality

        self.results = evaluation_results
        return evaluation_results

    def _calculate_overall_quality(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics from all evaluation components."""
        quality_metrics = {}

        # Distribution quality
        if "distribution_analysis" in evaluation_results and "error" not in evaluation_results["distribution_analysis"]:
            dist_analysis = evaluation_results["distribution_analysis"]
            if "overall_quality" in dist_analysis:
                quality_metrics["distribution_quality"] = dist_analysis["overall_quality"]

        # Correlation quality
        if "correlation_analysis" in evaluation_results and "error" not in evaluation_results["correlation_analysis"]:
            corr_analysis = evaluation_results["correlation_analysis"]
            if "correlation_quality_score" in corr_analysis:
                quality_metrics["correlation_quality"] = corr_analysis["correlation_quality_score"]

        # Regression quality
        if "regression_analysis" in evaluation_results and "error" not in evaluation_results["regression_analysis"]:
            reg_analysis = evaluation_results["regression_analysis"]
            if "summary" in reg_analysis and "average_similarity_score" in reg_analysis["summary"]:
                quality_metrics["regression_quality"] = reg_analysis["summary"]["average_similarity_score"]

        # Overall composite score
        if quality_metrics:
            # Extract scalar values for overall scoring
            dist_score = self._extract_scalar_score(quality_metrics.get("distribution_quality", {}))
            corr_score = quality_metrics.get("correlation_quality", 0)
            reg_score = quality_metrics.get("regression_quality", 0)

            scores = [s for s in [dist_score, corr_score, reg_score] if s is not None and s > 0]
            if scores:
                quality_metrics["overall_score"] = float(np.mean(scores))
            else:
                quality_metrics["overall_score"] = 0.0

            # Individual component summaries
            quality_metrics["summary"] = {
                "distribution_score": dist_score,
                "correlation_score": corr_score,
                "regression_score": reg_score,
                "components_evaluated": len(scores),
                "overall_score": quality_metrics.get("overall_score", 0.0)
            }

        return quality_metrics

    def _extract_scalar_score(self, distribution_quality: Dict[str, Any]) -> Optional[float]:
        """Extract a scalar score from distribution quality metrics."""
        try:
            # Use Jensen-Shannon divergence as primary metric (lower is better)
            js_div = distribution_quality.get("mean_jensen_shannon_divergence")
            if js_div is not None:
                # Convert to quality score (higher is better)
                return max(0, 1 - js_div)

            # Fallback to total variation distance
            tv_dist = distribution_quality.get("mean_total_variation_distance")
            if tv_dist is not None:
                return max(0, 1 - tv_dist)

            return None
        except:
            return None

    def generate_visualizations(self, output_dir: str, show_plots: bool = False) -> List[str]:
        """
        Generate evaluation visualizations.

        Args:
            output_dir: Directory to save plots
            show_plots: Whether to display plots

        Returns:
            List of generated file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []

        if not self.results:
            print("No evaluation results available. Run evaluation first.")
            return generated_files

        try:
            # 1. Distribution comparison plots
            dist_plots = self._create_distribution_plots(output_dir, show_plots)
            generated_files.extend(dist_plots)

            # 2. Correlation comparison plots
            corr_plots = self._create_correlation_plots(output_dir, show_plots)
            generated_files.extend(corr_plots)

            # 3. Quality summary plot
            summary_plot = self._create_quality_summary_plot(output_dir, show_plots)
            if summary_plot:
                generated_files.append(summary_plot)

        except Exception as e:
            print(f"Error generating visualizations: {e}")

        return generated_files

    def _create_distribution_plots(self, output_dir: str, show_plots: bool) -> List[str]:
        """Create distribution comparison plots."""
        generated_files = []

        if "distribution_analysis" not in self.results:
            return generated_files

        dist_analysis = self.results["distribution_analysis"]
        if "variable_comparisons" not in dist_analysis:
            return generated_files

        # Plot distribution comparison metrics
        try:
            variables = list(dist_analysis["variable_comparisons"].keys())
            js_divergences = []
            tv_distances = []

            for var in variables:
                var_results = dist_analysis["variable_comparisons"][var]
                if "jensen_shannon_divergence" in var_results:
                    js_divergences.append(var_results["jensen_shannon_divergence"])
                else:
                    js_divergences.append(0)

                if "total_variation_distance" in var_results:
                    tv_distances.append(var_results["total_variation_distance"])
                else:
                    tv_distances.append(0)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Jensen-Shannon divergence plot
            ax1.bar(variables, js_divergences)
            ax1.set_title("Jensen-Shannon Divergence by Variable")
            ax1.set_ylabel("JS Divergence")
            ax1.tick_params(axis='x', rotation=45)

            # Total variation distance plot
            ax2.bar(variables, tv_distances)
            ax2.set_title("Total Variation Distance by Variable")
            ax2.set_ylabel("TV Distance")
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            filepath = os.path.join(output_dir, "distribution_comparison.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            generated_files.append(filepath)

            if show_plots:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"Error creating distribution plots: {e}")

        return generated_files

    def _create_correlation_plots(self, output_dir: str, show_plots: bool) -> List[str]:
        """Create correlation comparison plots."""
        generated_files = []

        if "correlation_analysis" not in self.results:
            return generated_files

        corr_analysis = self.results["correlation_analysis"]

        try:
            # Plot correlation matrices
            if "pearson_comparison" in corr_analysis and "error" not in corr_analysis["pearson_comparison"]:
                pearson_comp = corr_analysis["pearson_comparison"]

                real_corr = pd.DataFrame(pearson_comp["real_correlation_matrix"])
                synthetic_corr = pd.DataFrame(pearson_comp["synthetic_correlation_matrix"])

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

                # Real correlation matrix
                sns.heatmap(real_corr, annot=True, cmap='RdBu_r', center=0, ax=ax1)
                ax1.set_title("Real Data Correlations")

                # Synthetic correlation matrix
                sns.heatmap(synthetic_corr, annot=True, cmap='RdBu_r', center=0, ax=ax2)
                ax2.set_title("Synthetic Data Correlations")

                # Difference matrix
                diff_matrix = synthetic_corr - real_corr
                sns.heatmap(diff_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax3)
                ax3.set_title("Difference (Synthetic - Real)")

                plt.tight_layout()

                filepath = os.path.join(output_dir, "correlation_comparison.png")
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                generated_files.append(filepath)

                if show_plots:
                    plt.show()
                else:
                    plt.close()

        except Exception as e:
            print(f"Error creating correlation plots: {e}")

        return generated_files

    def _create_quality_summary_plot(self, output_dir: str, show_plots: bool) -> Optional[str]:
        """Create overall quality summary plot."""
        if "overall_quality" not in self.results:
            return None

        try:
            quality = self.results["overall_quality"]
            if "summary" not in quality:
                return None

            summary = quality["summary"]

            # Create radar/bar chart of quality scores
            metrics = []
            scores = []

            if summary.get("distribution_score") is not None:
                metrics.append("Distribution\nQuality")
                scores.append(summary["distribution_score"])

            if summary.get("correlation_score") is not None:
                metrics.append("Correlation\nQuality")
                scores.append(summary["correlation_score"])

            if summary.get("regression_score") is not None:
                metrics.append("Regression\nQuality")
                scores.append(summary["regression_score"])

            if not metrics:
                return None

            fig, ax = plt.subplots(figsize=(10, 6))

            bars = ax.bar(metrics, scores, color=['skyblue', 'lightgreen', 'lightcoral'][:len(scores)])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Quality Score")
            ax.set_title(f"Synthetic Data Quality Assessment\n(Overall Score: {summary.get('overall_score', 0):.3f})")

            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')

            plt.tight_layout()

            filepath = os.path.join(output_dir, "quality_summary.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')

            if show_plots:
                plt.show()
            else:
                plt.close()

            return filepath

        except Exception as e:
            print(f"Error creating quality summary plot: {e}")
            return None

    def export_results(self, filepath: str) -> None:
        """Export evaluation results to JSON file."""
        if not self.results:
            print("No evaluation results to export. Run evaluation first.")
            return

        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"Results exported to {filepath}")
        except Exception as e:
            print(f"Error exporting results: {e}")

    def generate_html_report(self, output_path: str, include_plots: bool = True) -> None:
        """
        Generate an HTML report of evaluation results.

        Args:
            output_path: Path for the HTML file
            include_plots: Whether to include visualizations
        """
        if not self.results:
            print("No evaluation results available. Run evaluation first.")
            return

        try:
            # Generate plots if requested
            plot_dir = os.path.dirname(output_path)
            plot_files = []
            if include_plots:
                plot_files = self.generate_visualizations(plot_dir, show_plots=False)

            # Create HTML content
            html_content = self._create_html_content(plot_files)

            # Write HTML file
            with open(output_path, 'w') as f:
                f.write(html_content)

            print(f"HTML report generated: {output_path}")

        except Exception as e:
            print(f"Error generating HTML report: {e}")

    def _create_html_content(self, plot_files: List[str]) -> str:
        """Create HTML content for the evaluation report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Synthetic Data Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .score { font-weight: bold; color: #2e7d32; }
                .error { color: #d32f2f; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .plot { text-align: center; margin: 20px 0; }
            </style>
        </head>
        <body>
        """

        html += f"""
        <h1>Synthetic Data Evaluation Report</h1>
        <p><strong>Generated:</strong> {self.results.get('metadata', {}).get('timestamp', 'Unknown')}</p>
        """

        # Overall Quality Summary
        if "overall_quality" in self.results:
            html += self._create_quality_section_html()

        # Distribution Analysis
        if "distribution_analysis" in self.results:
            html += self._create_distribution_section_html()

        # Correlation Analysis
        if "correlation_analysis" in self.results:
            html += self._create_correlation_section_html()

        # Regression Analysis
        if "regression_analysis" in self.results:
            html += self._create_regression_section_html()

        # Include plots
        for plot_file in plot_files:
            plot_name = os.path.basename(plot_file)
            html += f'<div class="plot"><img src="{plot_name}" alt="{plot_name}" style="max-width: 100%;"></div>'

        html += """
        </body>
        </html>
        """

        return html

    def _create_quality_section_html(self) -> str:
        """Create HTML section for overall quality."""
        quality = self.results["overall_quality"]
        html = "<h2>Overall Quality Assessment</h2>"

        if "summary" in quality:
            summary = quality["summary"]
            overall_score = summary.get("overall_score", 0)

            html += f'<div class="metric"><span class="score">Overall Quality Score: {overall_score:.3f}</span></div>'

            # Individual scores
            if summary.get("distribution_score") is not None:
                html += f'<div class="metric">Distribution Quality: {summary["distribution_score"]:.3f}</div>'
            if summary.get("correlation_score") is not None:
                html += f'<div class="metric">Correlation Quality: {summary["correlation_score"]:.3f}</div>'
            if summary.get("regression_score") is not None:
                html += f'<div class="metric">Regression Quality: {summary["regression_score"]:.3f}</div>'

        return html

    def _create_distribution_section_html(self) -> str:
        """Create HTML section for distribution analysis."""
        return "<h2>Distribution Analysis</h2><p>Distribution comparison completed. See visualization below.</p>"

    def _create_correlation_section_html(self) -> str:
        """Create HTML section for correlation analysis."""
        return "<h2>Correlation Analysis</h2><p>Correlation structure comparison completed. See visualization below.</p>"

    def _create_regression_section_html(self) -> str:
        """Create HTML section for regression analysis."""
        return "<h2>Regression Analysis</h2><p>Regression replication tests completed. See detailed results in exported JSON.</p>"