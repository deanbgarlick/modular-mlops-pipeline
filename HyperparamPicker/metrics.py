"""Custom Ax Metrics for Pipeline integration."""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path

from ax.core import Trial
from ax.core.metric import Metric
from ax.core.data import Data

import pandas as pd
import numpy as np


class PipelineMetric(Metric):
    """Custom Ax Metric that extracts results from Pipeline runs."""
    
    def __init__(self,
                 name: str,
                 results_dir: str,
                 lower_is_better: bool = False):
        """
        Initialize the Pipeline metric.
        
        Args:
            name: Name of the metric (e.g., 'accuracy', 'training_time')
            results_dir: Directory where trial results are stored
            lower_is_better: Whether lower values are better for this metric
        """
        super().__init__(name=name, lower_is_better=lower_is_better)
        self.results_dir = Path(results_dir)
        
    def fetch_trial_data(self, trial: Trial, **kwargs) -> Data:
        """
        Fetch metric data from a completed trial.
        
        Args:
            trial: Ax trial object
            **kwargs: Additional arguments
            
        Returns:
            Data object containing the metric data
        """
        try:
            # Check if trial has results
            results_file = self.results_dir / f"trial_{trial.index}_results.json"
            error_file = self.results_dir / f"trial_{trial.index}_error.json"
            
            if not results_file.exists():
                # Return empty data if no results yet
                return Data(df=pd.DataFrame())
            
            # Load results
            with open(results_file, 'r') as f:
                trial_data = json.load(f)
            
            # Extract the metric value
            results = trial_data.get("results", {})
            metric_value = self._extract_metric_value(results)
            
            if metric_value is None:
                return Data(df=pd.DataFrame())
            
            # Create DataFrame with the metric data
            df = pd.DataFrame({
                "arm_name": [trial.arm.name if trial.arm else f"trial_{trial.index}"],
                "metric_name": [self.name],
                "mean": [metric_value],
                "sem": [0.0],  # Standard error of the mean (0 for single observations)
                "trial_index": [trial.index],
                "n": [1]  # Number of observations
            })
            
            return Data(df=df)
            
        except Exception as e:
            print(f"Error fetching metric {self.name} for trial {trial.index}: {e}")
            return Data(df=pd.DataFrame())
    
    def _extract_metric_value(self, results: Dict[str, Any]) -> Optional[float]:
        """
        Extract the metric value from Pipeline results.
        
        Args:
            results: Dictionary of results from Pipeline
            
        Returns:
            Metric value or None if not found
        """
        if self.name in results:
            value = results[self.name]
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return None
        
        # Handle common metric mappings
        metric_mappings = {
            "accuracy": ["accuracy", "test_accuracy", "val_accuracy"],
            "f1_score": ["f1_score", "f1_macro", "test_f1_macro", "val_f1_macro"],
            "precision": ["precision", "test_precision", "val_precision"],
            "recall": ["recall", "test_recall", "val_recall"],
            "training_time": ["training_time", "train_time", "elapsed_time"],
            "prediction_time": ["prediction_time", "inference_time", "test_time"],
            "model_size": ["model_size", "n_parameters", "num_parameters"]
        }
        
        if self.name in metric_mappings:
            for key in metric_mappings[self.name]:
                if key in results:
                    value = results[key]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, str):
                        try:
                            return float(value)
                        except ValueError:
                            continue
        
        return None
    
    def is_available_while_running(self):
        """Return whether the metric is available while the trial is running."""
        return False


class AccuracyMetric(PipelineMetric):
    """Metric for model accuracy."""
    
    def __init__(self, results_dir: str):
        super().__init__(name="accuracy", results_dir=results_dir, lower_is_better=False)


class F1ScoreMetric(PipelineMetric):
    """Metric for F1 score."""
    
    def __init__(self, results_dir: str):
        super().__init__(name="f1_score", results_dir=results_dir, lower_is_better=False)


class TrainingTimeMetric(PipelineMetric):
    """Metric for training time."""
    
    def __init__(self, results_dir: str):
        super().__init__(name="training_time", results_dir=results_dir, lower_is_better=True)


class PredictionTimeMetric(PipelineMetric):
    """Metric for prediction/inference time."""
    
    def __init__(self, results_dir: str):
        super().__init__(name="prediction_time", results_dir=results_dir, lower_is_better=True)


class ModelSizeMetric(PipelineMetric):
    """Metric for model size (number of parameters)."""
    
    def __init__(self, results_dir: str):
        super().__init__(name="model_size", results_dir=results_dir, lower_is_better=True)


def create_metrics_for_objectives(objectives: List[str], results_dir: str) -> Dict[str, PipelineMetric]:
    """
    Create metric objects for a list of objectives.
    
    Args:
        objectives: List of objective names
        results_dir: Directory where results are stored
        
    Returns:
        Dictionary mapping objective names to metric objects
    """
    metrics = {}
    
    for objective in objectives:
        if objective == "accuracy":
            metrics[objective] = AccuracyMetric(results_dir)
        elif objective == "f1_score":
            metrics[objective] = F1ScoreMetric(results_dir)
        elif objective == "training_time":
            metrics[objective] = TrainingTimeMetric(results_dir)
        elif objective == "prediction_time":
            metrics[objective] = PredictionTimeMetric(results_dir)
        elif objective == "model_size":
            metrics[objective] = ModelSizeMetric(results_dir)
        else:
            # Generic metric
            metrics[objective] = PipelineMetric(
                name=objective,
                results_dir=results_dir,
                lower_is_better=False  # Default assumption
            )
    
    return metrics 