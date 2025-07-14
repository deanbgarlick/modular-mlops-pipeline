"""Factory function for creating supervised models."""

from .base import SupervisedModel, SupervisedModelType
from .logistic_regression import LogisticRegression
from .pytorch_neural_network import PyTorchNeuralNetwork
from .pytorch_neural_network_simple import SimplePyTorchNeuralNetwork
from .knn_classifier import KNNClassifier


def create_model(model_type: SupervisedModelType, **kwargs) -> SupervisedModel:
    """Create and return the appropriate supervised model.
    
    Args:
        model_type: The type of supervised model to create
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        SupervisedModel: The created model instance
        
    Raises:
        ValueError: If model_type is not supported
    """
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