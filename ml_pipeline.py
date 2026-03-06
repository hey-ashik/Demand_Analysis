"""
ML Pipeline Module
Handles regression model training, evaluation, and feature engineering.
Supports Linear Regression, Random Forest, and Gradient Boosting regressors.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MLPipeline:
    """Machine Learning Pipeline for regression experiments."""

    REGRESSORS = {
        'Linear Regression': LinearRegression,
        'Random Forest Regressor': RandomForestRegressor,
        'Gradient Boosting Regressor': GradientBoostingRegressor,
    }

    def __init__(self):
        self.dataset = None
        self.features = None
        self.target = None
        self.model = None
        self.scaler = StandardScaler()
        self.predictions = None
        self.metrics = {}
        self.experiment_history = []

    def load_dataset(self, filepath):
        """Load CSV dataset with validation."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        if not filepath.lower().endswith('.csv'):
            raise ValueError("File must be a CSV file.")

        try:
            self.dataset = pd.read_csv(filepath)
            logger.info(f"Dataset loaded: {filepath} | Shape: {self.dataset.shape}")
            return self.dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def get_columns(self):
        """Return dataset column names."""
        if self.dataset is None:
            return []
        return list(self.dataset.columns)

    def get_numeric_columns(self):
        """Return only numeric column names."""
        if self.dataset is None:
            return []
        return list(self.dataset.select_dtypes(include=[np.number]).columns)

    def generate_features(self, target_column, history_size=3):
        """Generate features using history window (lag features)."""
        if self.dataset is None:
            raise ValueError("No dataset loaded.")

        if target_column not in self.dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        numeric_df = self.dataset.select_dtypes(include=[np.number])

        if target_column not in numeric_df.columns:
            raise ValueError(f"Target column '{target_column}' must be numeric.")

        target_series = numeric_df[target_column].values

        # Generate lag features
        feature_list = []
        labels = []

        for i in range(history_size, len(target_series)):
            feature_list.append(target_series[i - history_size:i])
            labels.append(target_series[i])

        self.features = np.array(feature_list)
        self.target = np.array(labels)

        logger.info(f"Features generated | History: {history_size} | "
                     f"Samples: {len(self.features)} | Features shape: {self.features.shape}")
        return self.features, self.target

    def train_and_evaluate(self, regressor_name, n_folds=5):
        """Train the selected regressor and evaluate with K-Fold CV."""
        if self.features is None or self.target is None:
            raise ValueError("Features not generated. Call generate_features() first.")

        if regressor_name not in self.REGRESSORS:
            raise ValueError(f"Unknown regressor: {regressor_name}. "
                             f"Choose from: {list(self.REGRESSORS.keys())}")

        # Scale features
        X_scaled = self.scaler.fit_transform(self.features)

        # Initialize model
        model_class = self.REGRESSORS[regressor_name]
        if regressor_name == 'Random Forest Regressor':
            self.model = model_class(n_estimators=100, random_state=42, n_jobs=-1)
        elif regressor_name == 'Gradient Boosting Regressor':
            self.model = model_class(n_estimators=100, random_state=42, max_depth=5)
        else:
            self.model = model_class()

        # K-Fold Cross Validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        cv_mae_scores = -cross_val_score(self.model, X_scaled, self.target,
                                          cv=kf, scoring='neg_mean_absolute_error')
        cv_mse_scores = -cross_val_score(self.model, X_scaled, self.target,
                                          cv=kf, scoring='neg_mean_squared_error')
        cv_r2_scores = cross_val_score(self.model, X_scaled, self.target,
                                        cv=kf, scoring='r2')

        # Train on full data for predictions
        self.model.fit(X_scaled, self.target)
        self.predictions = self.model.predict(X_scaled)

        # Compute metrics
        self.metrics = {
            'MAE': round(float(np.mean(cv_mae_scores)), 4),
            'MSE': round(float(np.mean(cv_mse_scores)), 4),
            'R2_Score': round(float(np.mean(cv_r2_scores)), 4),
            'MAE_std': round(float(np.std(cv_mae_scores)), 4),
            'MSE_std': round(float(np.std(cv_mse_scores)), 4),
            'R2_std': round(float(np.std(cv_r2_scores)), 4),
        }

        logger.info(f"Model trained: {regressor_name} | "
                     f"MAE: {self.metrics['MAE']} | "
                     f"MSE: {self.metrics['MSE']} | "
                     f"R²: {self.metrics['R2_Score']}")
        return self.metrics

    def get_predictions(self):
        """Return actual vs predicted values."""
        if self.predictions is None:
            return None, None
        return self.target.tolist(), self.predictions.tolist()

    def save_results(self, output_filepath, filename, history, regressor_name):
        """Append experiment results to CSV file."""
        os.makedirs(os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else '.', exist_ok=True)

        result_row = {
            'Filename': filename,
            'History': history,
            'Regressor': regressor_name,
            'MAE': self.metrics.get('MAE', 'N/A'),
            'MSE': self.metrics.get('MSE', 'N/A'),
            'R2_Score': self.metrics.get('R2_Score', 'N/A'),
        }

        self.experiment_history.append(result_row)

        # Append to CSV
        result_df = pd.DataFrame([result_row])
        if os.path.exists(output_filepath):
            result_df.to_csv(output_filepath, mode='a', header=False, index=False)
        else:
            result_df.to_csv(output_filepath, index=False)

        logger.info(f"Results saved to: {output_filepath}")
        return result_row

    def load_results(self, output_filepath):
        """Load existing experiment results from CSV."""
        if not os.path.exists(output_filepath):
            return pd.DataFrame(columns=['Filename', 'History', 'Regressor', 'MAE', 'MSE', 'R2_Score'])
        try:
            df = pd.read_csv(output_filepath)
            return df
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return pd.DataFrame(columns=['Filename', 'History', 'Regressor', 'MAE', 'MSE', 'R2_Score'])

    def run_experiment(self, filepath, target_column, history_size, regressor_name,
                       n_folds, output_filepath):
        """Run a complete experiment pipeline."""
        self.load_dataset(filepath)
        self.generate_features(target_column, history_size)
        metrics = self.train_and_evaluate(regressor_name, n_folds)
        filename = os.path.basename(filepath)
        self.save_results(output_filepath, filename, history_size, regressor_name)
        return metrics
