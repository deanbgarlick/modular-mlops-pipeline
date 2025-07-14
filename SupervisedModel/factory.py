"""Factory function for creating supervised models."""

import os
from typing import Optional
from .base import SupervisedModel, SupervisedModelType
from .persistence import (
    ModelPersistence,
    PickleGCPBucketPersistence,
    TorchGCPBucketPersistence
)
from .logistic_regression import LogisticRegression
from .pytorch_neural_network import PyTorchNeuralNetwork
from .pytorch_neural_network_simple import SimplePyTorchNeuralNetwork
from .knn_classifier import KNNClassifier


def create_model(model_type: SupervisedModelType, 
                persistence: Optional[ModelPersistence] = None,
                default_bucket_name: Optional[str] = None,
                **kwargs) -> SupervisedModel:
    """Create and return the appropriate supervised model.
    
    Args:
        model_type: The type of supervised model to create
        persistence: Model persistence handler. If None, creates appropriate GCP persistence 
                    based on model type (PyTorch models use TorchGCPBucketPersistence, 
                    others use PickleGCPBucketPersistence).
        default_bucket_name: Default GCP bucket name to use when persistence is None.
                            Falls back to SUPERVISED_MODEL_BUCKET env var or "default-model-bucket"
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        SupervisedModel: The created model instance
        
    Raises:
        ValueError: If model_type is not supported
        
    Examples:
        # Basic usage with automatic GCP persistence
        model = create_model(SupervisedModelType.LOGISTIC_REGRESSION)
        
        # PyTorch model with custom bucket name
        model = create_model(
            SupervisedModelType.PYTORCH_NEURAL_NETWORK,
            default_bucket_name="my-pytorch-models-bucket",
            hidden_size=128
        )
        
        # With explicit persistence
        from .persistence import TorchGCPBucketPersistence
        persistence = TorchGCPBucketPersistence("custom-bucket")
        model = create_model(
            SupervisedModelType.SIMPLE_PYTORCH_NEURAL_NETWORK,
            persistence=persistence
        )
    """
    # Create default GCP persistence if none provided
    if persistence is None:
        # Get bucket name from parameter, environment variable, or default
        bucket_name = (
            default_bucket_name or 
            os.getenv('SUPERVISED_MODEL_BUCKET', 'default-model-bucket')
        )
        
        # Choose appropriate persistence based on model type
        if model_type in [SupervisedModelType.PYTORCH_NEURAL_NETWORK, 
                         SupervisedModelType.SIMPLE_PYTORCH_NEURAL_NETWORK]:
            persistence = TorchGCPBucketPersistence(bucket_name)
        else:
            persistence = PickleGCPBucketPersistence(bucket_name)
    
    # Add persistence to kwargs
    kwargs['persistence'] = persistence
    
    if model_type == SupervisedModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(**kwargs)
    elif model_type == SupervisedModelType.PYTORCH_NEURAL_NETWORK:
        return PyTorchNeuralNetwork(**kwargs)
    elif model_type == SupervisedModelType.SIMPLE_PYTORCH_NEURAL_NETWORK:
        return SimplePyTorchNeuralNetwork(**kwargs)
    elif model_type == SupervisedModelType.KNN_CLASSIFIER:
        return KNNClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown supervised model type: {model_type}") 