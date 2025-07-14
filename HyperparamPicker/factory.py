"""Factory function for creating HyperparamPicker instances."""

from typing import Dict, List, Any, Optional, Tuple

from DataLoader import DataSourceType
from FeatureExtractor import FeatureExtractorType
from Model import ModelType

from .config import (
    HyperparamSearchConfig,
    MultiObjectiveConfig,
    create_default_search_config,
    create_default_multi_objective_config
)
from .core import HyperparamPicker


def create_hyperparam_picker(
    search_config: Optional[HyperparamSearchConfig] = None,
    multi_objective_config: Optional[MultiObjectiveConfig] = None,
    experiment_name: str = "hyperparameter_optimization"
) -> HyperparamPicker:
    """
    Create a HyperparamPicker with sensible defaults.
    
    Args:
        search_config: Configuration for hyperparameter search space (uses default if None)
        multi_objective_config: Configuration for multi-objective optimization (uses default if None)
        experiment_name: Name for the experiment
        
    Returns:
        HyperparamPicker instance
    """
    if search_config is None:
        search_config = create_default_search_config()
    
    if multi_objective_config is None:
        multi_objective_config = create_default_multi_objective_config()
    
    return HyperparamPicker(
        search_config=search_config,
        multi_objective_config=multi_objective_config,
        experiment_name=experiment_name
    )


def run_hyperparameter_optimization(
    data_source_type: DataSourceType,
    feature_extractor_type: FeatureExtractorType,
    model_type: ModelType,
    loader_kwargs: Dict[str, Any],
    base_extractor_kwargs: Optional[Dict[str, Any]] = None,
    base_model_kwargs: Optional[Dict[str, Any]] = None,
    total_trials: int = 20,
    max_parallel_trials: int = 2,
    use_class_weights: bool = True,
    objectives: Optional[List[Tuple[str, bool]]] = None,
    experiment_name: str = "hyperparameter_optimization"
):
    """
    Run hyperparameter optimization with sensible defaults.
    
    Args:
        data_source_type: Type of data source
        feature_extractor_type: Type of feature extractor
        model_type: Type of model
        loader_kwargs: Arguments for data loader
        base_extractor_kwargs: Base arguments for feature extractor
        base_model_kwargs: Base arguments for model
        total_trials: Total number of trials to run
        max_parallel_trials: Maximum number of parallel trials
        use_class_weights: Whether to use class weights
        objectives: List of (objective_name, minimize) tuples
        experiment_name: Name for the experiment
        
    Returns:
        HyperparamResults object containing optimization results
    """
    # Set defaults
    if base_extractor_kwargs is None:
        base_extractor_kwargs = {}
    if base_model_kwargs is None:
        base_model_kwargs = {}
    
    # Create search config
    search_config = create_default_search_config()
    
    # Create multi-objective config
    if objectives is None:
        objectives = [("accuracy", False), ("training_time", True)]
    
    multi_objective_config = MultiObjectiveConfig(
        objectives=objectives,
        objective_thresholds={
            "accuracy": 0.80,
            "training_time": 300
        }
    )
    
    # Create picker
    picker = create_hyperparam_picker(
        search_config=search_config,
        multi_objective_config=multi_objective_config,
        experiment_name=experiment_name
    )
    
    # Run optimization
    results = picker.optimize(
        data_source_type=data_source_type,
        feature_extractor_type=feature_extractor_type,
        model_type=model_type,
        loader_kwargs=loader_kwargs,
        base_extractor_kwargs=base_extractor_kwargs,
        base_model_kwargs=base_model_kwargs,
        total_trials=total_trials,
        max_parallel_trials=max_parallel_trials,
        use_class_weights=use_class_weights
    )
    
    return results


# Example usage configurations
def create_tfidf_logistic_regression_config():
    """Create configuration for TF-IDF + Logistic Regression optimization."""
    return {
        "data_source_type": DataSourceType.CSV_FILE,
        "feature_extractor_type": FeatureExtractorType.TFIDF_VECTORIZER,
        "model_type": ModelType.LOGISTIC_REGRESSION,
        "loader_kwargs": {
            "file_path": "dataset.csv",
            "text_column": "customer_review",
            "target_column": "return",
            "sep": "\t"
        },
        "base_extractor_kwargs": {},
        "base_model_kwargs": {}
    }


def create_openai_embeddings_config():
    """Create configuration for OpenAI Embeddings + Logistic Regression optimization."""
    return {
        "data_source_type": DataSourceType.CSV_FILE,
        "feature_extractor_type": FeatureExtractorType.OPENAI_EMBEDDINGS,
        "model_type": ModelType.LOGISTIC_REGRESSION,
        "loader_kwargs": {
            "file_path": "dataset.csv",
            "text_column": "customer_review",
            "target_column": "return",
            "sep": "\t"
        },
        "base_extractor_kwargs": {},
        "base_model_kwargs": {}
    } 