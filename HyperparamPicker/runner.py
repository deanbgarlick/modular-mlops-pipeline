"""Custom Ax Runner for Pipeline integration."""

import json
import time
import uuid
from typing import Dict, Any, Optional, Tuple, Iterable
from pathlib import Path
import tempfile
import os
import numpy as np

from ax.core import Trial
from ax.core.runner import Runner
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.utils import get_model_times

from DataLoader import DataSourceType
from FeatureExtractor import FeatureExtractorType
from Model import ModelType
from Pipeline import run_pipeline


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy arrays and custom objects."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle custom objects by converting to string representation
        if hasattr(obj, '__dict__'):
            return f"<{obj.__class__.__name__} object>"
        return super().default(obj)


def filter_serializable_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out non-serializable objects from results."""
    filtered_results = {}
    
    # Keep only basic numeric metrics and simple data structures
    for key, value in results.items():
        if key in ['accuracy', 'f1_macro', 'f1_weighted', 'training_time']:
            filtered_results[key] = value
        elif key == 'target_names' and isinstance(value, list):
            filtered_results[key] = value
        elif key == 'predictions' and isinstance(value, np.ndarray):
            # Convert to list for JSON serialization
            filtered_results[key] = value.tolist()
        # Skip fitted_extractor and trained_model as they're not serializable
    
    return filtered_results


class PipelineRunner(Runner):
    """Custom Ax Runner that integrates with existing Pipeline infrastructure."""
    
    def __init__(self, 
                 data_source_type: DataSourceType,
                 feature_extractor_type: FeatureExtractorType,
                 model_type: ModelType,
                 loader_kwargs: Dict[str, Any],
                 base_extractor_kwargs: Dict[str, Any],
                 base_model_kwargs: Dict[str, Any],
                 use_class_weights: bool = True,
                 results_dir: Optional[str] = None):
        """
        Initialize the Pipeline runner.
        
        Args:
            data_source_type: Type of data source
            feature_extractor_type: Type of feature extractor
            model_type: Type of model
            loader_kwargs: Arguments for data loader
            base_extractor_kwargs: Base arguments for feature extractor
            base_model_kwargs: Base arguments for model
            use_class_weights: Whether to use class weights
            results_dir: Directory to store results (will create temp dir if None)
        """
        self.data_source_type = data_source_type
        self.feature_extractor_type = feature_extractor_type
        self.model_type = model_type
        self.loader_kwargs = loader_kwargs
        self.base_extractor_kwargs = base_extractor_kwargs
        self.base_model_kwargs = base_model_kwargs
        self.use_class_weights = use_class_weights
        
        # Create results directory
        if results_dir is None:
            self.results_dir = Path(tempfile.mkdtemp(prefix="hyperparam_"))
        else:
            self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Track running trials
        self.running_trials: Dict[str, Dict[str, Any]] = {}
    
    def run(self, trial: Trial) -> Dict[str, Any]:
        """
        Run a single trial with the specified hyperparameters.
        
        Args:
            trial: Ax trial object containing hyperparameters
            
        Returns:
            Dictionary containing trial metadata
        """
        trial_id = str(uuid.uuid4())
        trial_start_time = time.time()
        
        print(f"Starting trial {trial.index} with ID {trial_id}")
        
        try:
            # Extract hyperparameters from trial
            if trial.arm is None:
                raise ValueError(f"Trial {trial.index} has no arm")
            
            hyperparams = trial.arm.parameters
            
            # Build Pipeline configuration
            extractor_kwargs = self._build_extractor_kwargs(hyperparams)
            model_kwargs = self._build_model_kwargs(hyperparams)
            
            print(f"Trial {trial.index}: Extractor kwargs: {extractor_kwargs}")
            print(f"Trial {trial.index}: Model kwargs: {model_kwargs}")
            
            # Run the pipeline
            results = run_pipeline(
                data_source_type=self.data_source_type,
                feature_extractor_type=self.feature_extractor_type,
                model_type=self.model_type,
                use_class_weights=self.use_class_weights,
                loader_kwargs=self.loader_kwargs,
                extractor_kwargs=extractor_kwargs,
                model_kwargs=model_kwargs
            )
            
            trial_end_time = time.time()
            training_time = trial_end_time - trial_start_time
            
            # Add training time to results
            results["training_time"] = training_time
            
            # Filter results to keep only serializable data
            filtered_results = filter_serializable_results(results)
            
            # Save results to file
            results_file = self.results_dir / f"trial_{trial.index}_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "trial_index": trial.index,
                    "trial_id": trial_id,
                    "hyperparameters": hyperparams,
                    "results": filtered_results,
                    "training_time": training_time,
                    "status": "completed"
                }, f, indent=2, cls=NumpyEncoder)
            
            print(f"Trial {trial.index} completed successfully in {training_time:.2f}s")
            
            return {
                "trial_id": trial_id,
                "trial_index": trial.index,
                "results_file": str(results_file),
                "status": "completed",
                "training_time": training_time
            }
            
        except Exception as e:
            trial_end_time = time.time()
            training_time = trial_end_time - trial_start_time
            
            print(f"Trial {trial.index} failed: {str(e)}")
            
            # Get hyperparameters for error logging
            hyperparams = trial.arm.parameters if trial.arm else {}
            
            # Save error information
            error_file = self.results_dir / f"trial_{trial.index}_error.json"
            with open(error_file, 'w') as f:
                json.dump({
                    "trial_index": trial.index,
                    "trial_id": trial_id,
                    "hyperparameters": hyperparams,
                    "error": str(e),
                    "training_time": training_time,
                    "status": "failed"
                }, f, indent=2, cls=NumpyEncoder)
            
            return {
                "trial_id": trial_id,
                "trial_index": trial.index,
                "error_file": str(error_file),
                "status": "failed",
                "training_time": training_time,
                "error": str(e)
            }

    def _build_extractor_kwargs(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Build feature extractor arguments from hyperparameters."""
        kwargs = self.base_extractor_kwargs.copy()
        
        # Add hyperparameters that start with 'extractor_'
        for key, value in hyperparams.items():
            if key.startswith("extractor_"):
                param_name = key[len("extractor_"):]
                kwargs[param_name] = value
        
        return kwargs

    def _build_model_kwargs(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Build model arguments from hyperparameters."""
        kwargs = self.base_model_kwargs.copy()
        
        # Add hyperparameters that start with 'model_'
        for key, value in hyperparams.items():
            if key.startswith("model_"):
                param_name = key[len("model_"):]
                kwargs[param_name] = value
        
        return kwargs

    def poll_trial_status(self, trials: Iterable[BaseTrial]) -> Dict[TrialStatus, set[int]]:
        """
        Poll the status of trials.
        
        Args:
            trials: Iterable of trials to poll
            
        Returns:
            Dictionary mapping TrialStatus to set of trial indices
        """
        status_to_trials = {
            TrialStatus.COMPLETED: set(),
            TrialStatus.FAILED: set(),
            TrialStatus.RUNNING: set()
        }
        
        for trial in trials:
            results_file = self.results_dir / f"trial_{trial.index}_results.json"
            error_file = self.results_dir / f"trial_{trial.index}_error.json"
            
            if results_file.exists():
                status_to_trials[TrialStatus.COMPLETED].add(trial.index)
            elif error_file.exists():
                status_to_trials[TrialStatus.FAILED].add(trial.index)
            else:
                status_to_trials[TrialStatus.RUNNING].add(trial.index)
        
        return status_to_trials
    
    def get_trial_results(self, trial_index: int) -> Dict[str, Any]:
        """
        Get results for a specific trial.
        
        Args:
            trial_index: Index of the trial
            
        Returns:
            Dictionary containing trial results
        """
        results_file = self.results_dir / f"trial_{trial_index}_results.json"
        error_file = self.results_dir / f"trial_{trial_index}_error.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        elif error_file.exists():
            with open(error_file, 'r') as f:
                return json.load(f)
        else:
            return {"status": "running"}
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        # Optionally clean up results directory
        # For now, we'll keep the results for analysis
        pass 