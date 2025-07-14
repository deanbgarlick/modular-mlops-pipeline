"""Factory function for creating supervised models."""

from typing import Optional
from .base import SupervisedModel, SupervisedModelType
from .persistence import ModelPersistence
from .logistic_regression import LogisticRegression
from .pytorch_neural_network import PyTorchNeuralNetwork
from .pytorch_neural_network_simple import SimplePyTorchNeuralNetwork
from .knn_classifier import KNNClassifier


def create_model(model_type: SupervisedModelType, persistence: Optional[ModelPersistence] = None, **kwargs) -> SupervisedModel:
    """Create and return the appropriate supervised model.
    
    Args:
        model_type: The type of supervised model to create
        persistence: Model persistence handler. If None, uses default GCP bucket persistence.
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        SupervisedModel: The created model instance
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == SupervisedModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(persistence=persistence, **kwargs)
    elif model_type == SupervisedModelType.PYTORCH_NEURAL_NETWORK:
        return PyTorchNeuralNetwork(persistence=persistence, **kwargs)
    elif model_type == SupervisedModelType.SIMPLE_PYTORCH_NEURAL_NETWORK:
        return SimplePyTorchNeuralNetwork(persistence=persistence, **kwargs)
    elif model_type == SupervisedModelType.KNN_CLASSIFIER:
        return KNNClassifier(persistence=persistence, **kwargs)
    else:
        raise ValueError(f"Unknown supervised model type: {model_type}") 