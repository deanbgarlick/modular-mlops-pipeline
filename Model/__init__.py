"""Model Package for Text Classification.

This package provides a modular approach to model selection for text classification,
supporting multiple model types through a clean interface.
"""

from .base import Model, ModelType
from .logistic_regression import LogisticRegression
from .pytorch_neural_network import PyTorchNeuralNetwork
from .pytorch_neural_network_simple import SimplePyTorchNeuralNetwork
from .knn_classifier import KNNClassifier
from .factory import create_model

__all__ = [
    'Model',
    'ModelType',
    'LogisticRegression',
    'PyTorchNeuralNetwork',
    'SimplePyTorchNeuralNetwork',
    'KNNClassifier',
    'create_model'
] 