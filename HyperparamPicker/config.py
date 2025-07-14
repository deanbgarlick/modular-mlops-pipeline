"""Configuration classes for hyperparameter optimization."""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ax.core import (
    ParameterType,
    RangeParameter,
    ChoiceParameter,
    FixedParameter,
    OptimizationConfig,
    MultiObjectiveOptimizationConfig,
    Objective,
    MultiObjective,
    ObjectiveThreshold
)
from ax.metrics.tensorboard import TensorboardMetric

from FeatureExtractor import FeatureExtractorType
from SupervisedModel import SupervisedModelType


class OptimizationObjective(Enum):
    """Supported optimization objectives."""
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    TRAINING_TIME = "training_time"
    PREDICTION_TIME = "prediction_time"
    MODEL_SIZE = "model_size"


@dataclass
class ParameterSpec:
    """Specification for a single hyperparameter."""
    name: str
    param_type: str  # 'range', 'choice', 'fixed'
    
    # For range parameters
    lower: Optional[float] = None
    upper: Optional[float] = None
    log_scale: bool = False
    parameter_type: ParameterType = ParameterType.FLOAT
    
    # For choice parameters
    values: Optional[List[Any]] = None
    is_ordered: bool = False
    
    # For fixed parameters
    value: Optional[Any] = None
    
    def to_ax_parameter(self):
        """Convert to Ax parameter object."""
        if self.param_type == 'range':
            if self.lower is None or self.upper is None:
                raise ValueError(f"Range parameter {self.name} must have lower and upper bounds")
            return RangeParameter(
                name=self.name,
                lower=self.lower,
                upper=self.upper,
                parameter_type=self.parameter_type,
                log_scale=self.log_scale
            )
        elif self.param_type == 'choice':
            if self.values is None:
                raise ValueError(f"Choice parameter {self.name} must have values")
            return ChoiceParameter(
                name=self.name,
                values=self.values,
                parameter_type=self.parameter_type,
                is_ordered=self.is_ordered
            )
        elif self.param_type == 'fixed':
            if self.value is None:
                raise ValueError(f"Fixed parameter {self.name} must have a value")
            return FixedParameter(
                name=self.name,
                value=self.value,
                parameter_type=self.parameter_type
            )
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")


@dataclass
class HyperparamSearchConfig:
    """Configuration for hyperparameter search space."""
    
    # Feature extractor hyperparameters
    feature_extractor_params: Dict[FeatureExtractorType, Dict[str, ParameterSpec]] = field(default_factory=dict)
    
    # Model hyperparameters
    model_params: Dict[SupervisedModelType, Dict[str, ParameterSpec]] = field(default_factory=dict)
    
    # Training hyperparameters (common across all models)
    training_params: Dict[str, ParameterSpec] = field(default_factory=dict)
    
    def get_parameters_for_config(self, 
                                 feature_extractor_type: FeatureExtractorType,
                                 model_type: SupervisedModelType) -> List[ParameterSpec]:
        """Get all parameters for a specific feature extractor and model combination."""
        parameters = []
        
        # Add feature extractor parameters
        if feature_extractor_type in self.feature_extractor_params:
            for param_name, param_spec in self.feature_extractor_params[feature_extractor_type].items():
                param_spec.name = f"extractor_{param_name}"
                parameters.append(param_spec)
        
        # Add model parameters
        if model_type in self.model_params:
            for param_name, param_spec in self.model_params[model_type].items():
                param_spec.name = f"model_{param_name}"
                parameters.append(param_spec)
        
        # Add training parameters
        for param_name, param_spec in self.training_params.items():
            param_spec.name = f"training_{param_name}"
            parameters.append(param_spec)
        
        return parameters


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    
    # List of objectives: (metric_name, minimize)
    objectives: List[Tuple[str, bool]] = field(default_factory=list)
    
    # Objective thresholds for constraint handling
    objective_thresholds: Optional[Dict[str, float]] = None
    
    def to_ax_optimization_config(self, metrics: Dict[str, TensorboardMetric]) -> OptimizationConfig:
        """Convert to Ax optimization configuration."""
        if len(self.objectives) == 1:
            # Single objective optimization
            metric_name, minimize = self.objectives[0]
            return OptimizationConfig(
                objective=Objective(
                    metric=metrics[metric_name],
                    minimize=minimize
                )
            )
        else:
            # Multi-objective optimization
            ax_objectives = []
            for metric_name, minimize in self.objectives:
                ax_objectives.append(
                    Objective(
                        metric=metrics[metric_name],
                        minimize=minimize
                    )
                )
            
            # Create objective thresholds if specified
            ax_thresholds = []
            if self.objective_thresholds:
                for metric_name, threshold in self.objective_thresholds.items():
                    if metric_name in metrics:
                        ax_thresholds.append(
                            ObjectiveThreshold(
                                metric=metrics[metric_name],
                                bound=threshold,
                                relative=False
                            )
                        )
            
            return MultiObjectiveOptimizationConfig(
                objective=MultiObjective(objectives=ax_objectives),
                objective_thresholds=ax_thresholds if ax_thresholds else None
            )


def create_default_search_config() -> HyperparamSearchConfig:
    """Create a default hyperparameter search configuration."""
    config = HyperparamSearchConfig()
    
    # TF-IDF Vectorizer parameters
    config.feature_extractor_params[FeatureExtractorType.TFIDF_VECTORIZER] = {
        "max_features": ParameterSpec(
            name="max_features",
            param_type="choice",
            values=[1000, 5000, 10000, 20000, 50000],
            parameter_type=ParameterType.INT,
            is_ordered=True
        ),
        "min_df": ParameterSpec(
            name="min_df",
            param_type="range",
            lower=1,
            upper=10,
            parameter_type=ParameterType.INT
        ),
        "max_df": ParameterSpec(
            name="max_df",
            param_type="range",
            lower=0.5,
            upper=0.95,
            parameter_type=ParameterType.FLOAT
        )
    }
    
    # Count Vectorizer parameters
    config.feature_extractor_params[FeatureExtractorType.COUNT_VECTORIZER] = {
        "max_features": ParameterSpec(
            name="max_features",
            param_type="choice",
            values=[1000, 5000, 10000, 20000, 50000],
            parameter_type=ParameterType.INT,
            is_ordered=True
        ),
        "min_df": ParameterSpec(
            name="min_df",
            param_type="range",
            lower=1,
            upper=10,
            parameter_type=ParameterType.INT
        ),
        "max_df": ParameterSpec(
            name="max_df",
            param_type="range",
            lower=0.5,
            upper=0.95,
            parameter_type=ParameterType.FLOAT
        )
    }
    
    # OpenAI Embeddings parameters
    config.feature_extractor_params[FeatureExtractorType.OPENAI_EMBEDDINGS] = {
        "model_name": ParameterSpec(
            name="model_name",
            param_type="choice",
            values=["text-embedding-3-small", "text-embedding-3-large"],
            parameter_type=ParameterType.STRING
        ),
        "batch_size": ParameterSpec(
            name="batch_size",
            param_type="choice",
            values=[50, 100, 200],
            parameter_type=ParameterType.INT,
            is_ordered=True
        )
    }
    
    # Logistic Regression parameters
    config.model_params[SupervisedModelType.LOGISTIC_REGRESSION] = {
        "C": ParameterSpec(
            name="C",
            param_type="range",
            lower=0.001,
            upper=10.0,
            parameter_type=ParameterType.FLOAT,
            log_scale=True
        ),
        "max_iter": ParameterSpec(
            name="max_iter",
            param_type="choice",
            values=[100, 200, 500, 1000],
            parameter_type=ParameterType.INT,
            is_ordered=True
        )
    }
    
    # KNN Classifier parameters
    config.model_params[SupervisedModelType.KNN_CLASSIFIER] = {
        "n_neighbors": ParameterSpec(
            name="n_neighbors",
            param_type="range",
            lower=3,
            upper=20,
            parameter_type=ParameterType.INT
        ),
        "weights": ParameterSpec(
            name="weights",
            param_type="choice",
            values=["uniform", "distance"],
            parameter_type=ParameterType.STRING
        )
    }
    
    return config


def create_default_multi_objective_config() -> MultiObjectiveConfig:
    """Create a default multi-objective configuration."""
    return MultiObjectiveConfig(
        objectives=[
            ("accuracy", False),  # Maximize accuracy
            ("training_time", True),  # Minimize training time
        ],
        objective_thresholds={
            "accuracy": 0.80,  # Require at least 80% accuracy
            "training_time": 300  # Prefer under 5 minutes training time
        }
    ) 