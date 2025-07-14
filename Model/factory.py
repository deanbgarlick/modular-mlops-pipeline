"""Factory function for creating models."""

from .base import Model, ModelType
from .logistic_regression import LogisticRegression
from .pytorch_neural_network import PyTorchNeuralNetwork
from .knn_classifier import KNNClassifier


def create_model(model_type: ModelType, **kwargs) -> Model:
    """Create and return the appropriate model.
    
    Args:
        model_type: The type of model to create
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Model: The created model instance
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(**kwargs)
    elif model_type == ModelType.PYTORCH_NEURAL_NETWORK:
        return PyTorchNeuralNetwork(**kwargs)
    elif model_type == ModelType.KNN_CLASSIFIER:
        return KNNClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 