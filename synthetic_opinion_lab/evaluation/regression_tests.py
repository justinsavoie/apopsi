import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings


class RegressionReplicator:
    """Tests whether synthetic data replicates regression relationships from real data."""

    def __init__(self):
        self.results = {}

    def compare_linear_regressions(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                 dependent_var: str, independent_vars: List[str]) -> Dict[str, Any]:
        """
        Compare linear regression models between real and synthetic data.

        Args:
            real_df: Real dataset
            synthetic_df: Synthetic dataset
            dependent_var: Name of dependent variable
            independent_vars: List of independent variable names

        Returns:
            Dictionary with regression comparison results
        """
        # Validate variables exist
        all_vars = [dependent_var] + independent_vars
        missing_real = [var for var in all_vars if var not in real_df.columns]
        missing_synthetic = [var for var in all_vars if var not in synthetic_df.columns]

        if missing_real or missing_synthetic:
            return {
                "error": "Missing variables",
                "missing_real": missing_real,
                "missing_synthetic": missing_synthetic
            }

        # Prepare data
        real_data = real_df[all_vars].copy()
        synthetic_data = synthetic_df[all_vars].copy()

        # Convert to numeric
        for var in all_vars:
            real_data[var] = pd.to_numeric(real_data[var], errors='coerce')
            synthetic_data[var] = pd.to_numeric(synthetic_data[var], errors='coerce')

        # Remove rows with missing values
        real_data_clean = real_data.dropna()
        synthetic_data_clean = synthetic_data.dropna()

        if len(real_data_clean) < len(independent_vars) + 1:
            return {"error": "Insufficient real data for regression"}

        if len(synthetic_data_clean) < len(independent_vars) + 1:
            return {"error": "Insufficient synthetic data for regression"}

        # Fit models
        X_real = real_data_clean[independent_vars]
        y_real = real_data_clean[dependent_var]
        X_synthetic = synthetic_data_clean[independent_vars]
        y_synthetic = synthetic_data_clean[dependent_var]

        # Fit linear regression models
        real_model = LinearRegression()
        synthetic_model = LinearRegression()

        real_model.fit(X_real, y_real)
        synthetic_model.fit(X_synthetic, y_synthetic)

        # Model predictions
        real_pred = real_model.predict(X_real)
        synthetic_pred = synthetic_model.predict(X_synthetic)

        results = {
            "dependent_variable": dependent_var,
            "independent_variables": independent_vars,
            "real_n": len(real_data_clean),
            "synthetic_n": len(synthetic_data_clean)
        }

        # Coefficients comparison
        real_coeffs = real_model.coef_
        synthetic_coeffs = synthetic_model.coef_

        coefficients_comparison = {}
        for i, var in enumerate(independent_vars):
            coefficients_comparison[var] = {
                "real_coefficient": float(real_coeffs[i]),
                "synthetic_coefficient": float(synthetic_coeffs[i]),
                "difference": float(synthetic_coeffs[i] - real_coeffs[i]),
                "relative_difference": float((synthetic_coeffs[i] - real_coeffs[i]) / real_coeffs[i])
                                     if real_coeffs[i] != 0 else None
            }

        results["coefficients"] = coefficients_comparison

        # Intercept comparison
        results["intercept"] = {
            "real_intercept": float(real_model.intercept_),
            "synthetic_intercept": float(synthetic_model.intercept_),
            "difference": float(synthetic_model.intercept_ - real_model.intercept_)
        }

        # Model performance comparison
        real_r2 = r2_score(y_real, real_pred)
        synthetic_r2 = r2_score(y_synthetic, synthetic_pred)
        real_mse = mean_squared_error(y_real, real_pred)
        synthetic_mse = mean_squared_error(y_synthetic, synthetic_pred)

        results["model_performance"] = {
            "real_r2": float(real_r2),
            "synthetic_r2": float(synthetic_r2),
            "r2_difference": float(synthetic_r2 - real_r2),
            "real_mse": float(real_mse),
            "synthetic_mse": float(synthetic_mse),
            "mse_ratio": float(synthetic_mse / real_mse) if real_mse != 0 else None
        }

        # Cross-prediction test (how well does real model predict synthetic data and vice versa)
        cross_predictions = self._cross_prediction_test(
            real_model, synthetic_model, X_real, y_real, X_synthetic, y_synthetic
        )
        results["cross_predictions"] = cross_predictions

        # Overall similarity score
        similarity_score = self._calculate_regression_similarity_score(results)
        results["similarity_score"] = similarity_score

        return results

    def compare_logistic_regressions(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                   dependent_var: str, independent_vars: List[str]) -> Dict[str, Any]:
        """
        Compare logistic regression models between real and synthetic data.

        Args:
            real_df: Real dataset
            synthetic_df: Synthetic dataset
            dependent_var: Name of dependent variable (binary/categorical)
            independent_vars: List of independent variable names

        Returns:
            Dictionary with logistic regression comparison results
        """
        # Validate variables exist
        all_vars = [dependent_var] + independent_vars
        missing_real = [var for var in all_vars if var not in real_df.columns]
        missing_synthetic = [var for var in all_vars if var not in synthetic_df.columns]

        if missing_real or missing_synthetic:
            return {
                "error": "Missing variables",
                "missing_real": missing_real,
                "missing_synthetic": missing_synthetic
            }

        # Prepare data
        real_data = real_df[all_vars].copy()
        synthetic_data = synthetic_df[all_vars].copy()

        # Convert independent variables to numeric
        for var in independent_vars:
            real_data[var] = pd.to_numeric(real_data[var], errors='coerce')
            synthetic_data[var] = pd.to_numeric(synthetic_data[var], errors='coerce')

        # Encode dependent variable
        le = LabelEncoder()
        all_y_values = list(real_data[dependent_var].dropna()) + list(synthetic_data[dependent_var].dropna())
        le.fit(all_y_values)

        real_data[dependent_var + '_encoded'] = le.transform(real_data[dependent_var])
        synthetic_data[dependent_var + '_encoded'] = le.transform(synthetic_data[dependent_var])

        # Remove rows with missing values
        real_data_clean = real_data.dropna()
        synthetic_data_clean = synthetic_data.dropna()

        if len(real_data_clean) < 10:
            return {"error": "Insufficient real data for logistic regression"}
        if len(synthetic_data_clean) < 10:
            return {"error": "Insufficient synthetic data for logistic regression"}

        # Prepare features and targets
        X_real = real_data_clean[independent_vars]
        y_real = real_data_clean[dependent_var + '_encoded']
        X_synthetic = synthetic_data_clean[independent_vars]
        y_synthetic = synthetic_data_clean[dependent_var + '_encoded']

        # Check if there's variation in the dependent variable
        if len(y_real.unique()) < 2 or len(y_synthetic.unique()) < 2:
            return {"error": "Dependent variable lacks variation for logistic regression"}

        try:
            # Fit logistic regression models
            real_model = LogisticRegression(random_state=42, max_iter=1000)
            synthetic_model = LogisticRegression(random_state=42, max_iter=1000)

            real_model.fit(X_real, y_real)
            synthetic_model.fit(X_synthetic, y_synthetic)

            # Model predictions
            real_pred = real_model.predict(X_real)
            synthetic_pred = synthetic_model.predict(X_synthetic)

        except Exception as e:
            return {"error": f"Failed to fit models: {str(e)}"}

        results = {
            "dependent_variable": dependent_var,
            "independent_variables": independent_vars,
            "real_n": len(real_data_clean),
            "synthetic_n": len(synthetic_data_clean),
            "classes": le.classes_.tolist()
        }

        # Coefficients comparison
        real_coeffs = real_model.coef_[0] if real_model.coef_.ndim > 1 else real_model.coef_
        synthetic_coeffs = synthetic_model.coef_[0] if synthetic_model.coef_.ndim > 1 else synthetic_model.coef_

        coefficients_comparison = {}
        for i, var in enumerate(independent_vars):
            coefficients_comparison[var] = {
                "real_coefficient": float(real_coeffs[i]),
                "synthetic_coefficient": float(synthetic_coeffs[i]),
                "difference": float(synthetic_coeffs[i] - real_coeffs[i]),
                "relative_difference": float((synthetic_coeffs[i] - real_coeffs[i]) / real_coeffs[i])
                                     if real_coeffs[i] != 0 else None
            }

        results["coefficients"] = coefficients_comparison

        # Intercept comparison
        results["intercept"] = {
            "real_intercept": float(real_model.intercept_[0]),
            "synthetic_intercept": float(synthetic_model.intercept_[0]),
            "difference": float(synthetic_model.intercept_[0] - real_model.intercept_[0])
        }

        # Model performance comparison
        real_accuracy = accuracy_score(y_real, real_pred)
        synthetic_accuracy = accuracy_score(y_synthetic, synthetic_pred)

        results["model_performance"] = {
            "real_accuracy": float(real_accuracy),
            "synthetic_accuracy": float(synthetic_accuracy),
            "accuracy_difference": float(synthetic_accuracy - real_accuracy)
        }

        # Overall similarity score
        similarity_score = self._calculate_logistic_similarity_score(results)
        results["similarity_score"] = similarity_score

        return results

    def _cross_prediction_test(self, real_model, synthetic_model, X_real, y_real, X_synthetic, y_synthetic) -> Dict[str, Any]:
        """Test how well models trained on one dataset predict the other."""
        try:
            # Real model predicting synthetic data
            real_to_synthetic_pred = real_model.predict(X_synthetic)
            real_to_synthetic_r2 = r2_score(y_synthetic, real_to_synthetic_pred)

            # Synthetic model predicting real data
            synthetic_to_real_pred = synthetic_model.predict(X_real)
            synthetic_to_real_r2 = r2_score(y_real, synthetic_to_real_pred)

            return {
                "real_model_on_synthetic_r2": float(real_to_synthetic_r2),
                "synthetic_model_on_real_r2": float(synthetic_to_real_r2),
                "average_cross_r2": float((real_to_synthetic_r2 + synthetic_to_real_r2) / 2)
            }
        except:
            return {
                "real_model_on_synthetic_r2": None,
                "synthetic_model_on_real_r2": None,
                "average_cross_r2": None
            }

    def _calculate_regression_similarity_score(self, results: Dict[str, Any]) -> float:
        """Calculate a similarity score for regression comparison."""
        try:
            # Score based on coefficient similarity and R² preservation
            coeff_similarities = []
            for var_results in results["coefficients"].values():
                if var_results["real_coefficient"] != 0:
                    rel_diff = abs(var_results["relative_difference"])
                    similarity = max(0, 1 - rel_diff)
                    coeff_similarities.append(similarity)

            # Average coefficient similarity
            avg_coeff_similarity = np.mean(coeff_similarities) if coeff_similarities else 0

            # R² preservation
            real_r2 = results["model_performance"]["real_r2"]
            synthetic_r2 = results["model_performance"]["synthetic_r2"]
            r2_similarity = 1 - abs(synthetic_r2 - real_r2)

            # Cross-prediction performance
            cross_r2 = results["cross_predictions"].get("average_cross_r2", 0)
            cross_performance = max(0, cross_r2) if cross_r2 is not None else 0

            # Weighted average
            similarity_score = (0.4 * avg_coeff_similarity + 0.4 * r2_similarity + 0.2 * cross_performance)
            return float(max(0, min(1, similarity_score)))

        except:
            return 0.0

    def _calculate_logistic_similarity_score(self, results: Dict[str, Any]) -> float:
        """Calculate a similarity score for logistic regression comparison."""
        try:
            # Score based on coefficient similarity and accuracy preservation
            coeff_similarities = []
            for var_results in results["coefficients"].values():
                if var_results["real_coefficient"] != 0:
                    rel_diff = abs(var_results["relative_difference"])
                    similarity = max(0, 1 - rel_diff)
                    coeff_similarities.append(similarity)

            # Average coefficient similarity
            avg_coeff_similarity = np.mean(coeff_similarities) if coeff_similarities else 0

            # Accuracy preservation
            real_acc = results["model_performance"]["real_accuracy"]
            synthetic_acc = results["model_performance"]["synthetic_accuracy"]
            acc_similarity = 1 - abs(synthetic_acc - real_acc)

            # Weighted average
            similarity_score = 0.7 * avg_coeff_similarity + 0.3 * acc_similarity
            return float(max(0, min(1, similarity_score)))

        except:
            return 0.0

    def run_regression_test_suite(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                test_specifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a suite of regression tests.

        Args:
            real_df: Real dataset
            synthetic_df: Synthetic dataset
            test_specifications: List of test specs, each with keys:
                - 'type': 'linear' or 'logistic'
                - 'dependent_var': str
                - 'independent_vars': List[str]
                - 'name': str (optional name for the test)

        Returns:
            Dictionary with all test results
        """
        results = {"tests": {}, "summary": {}}

        for i, test_spec in enumerate(test_specifications):
            test_name = test_spec.get("name", f"test_{i+1}")

            try:
                if test_spec["type"] == "linear":
                    test_result = self.compare_linear_regressions(
                        real_df, synthetic_df,
                        test_spec["dependent_var"],
                        test_spec["independent_vars"]
                    )
                elif test_spec["type"] == "logistic":
                    test_result = self.compare_logistic_regressions(
                        real_df, synthetic_df,
                        test_spec["dependent_var"],
                        test_spec["independent_vars"]
                    )
                else:
                    test_result = {"error": f"Unknown test type: {test_spec['type']}"}

                results["tests"][test_name] = test_result

            except Exception as e:
                results["tests"][test_name] = {"error": f"Test failed: {str(e)}"}

        # Calculate summary statistics
        similarity_scores = []
        successful_tests = 0

        for test_name, test_result in results["tests"].items():
            if "error" not in test_result and "similarity_score" in test_result:
                similarity_scores.append(test_result["similarity_score"])
                successful_tests += 1

        results["summary"] = {
            "total_tests": len(test_specifications),
            "successful_tests": successful_tests,
            "average_similarity_score": float(np.mean(similarity_scores)) if similarity_scores else 0,
            "min_similarity_score": float(np.min(similarity_scores)) if similarity_scores else 0,
            "max_similarity_score": float(np.max(similarity_scores)) if similarity_scores else 0
        }

        return results