"""SupervisedModel Package for Text Classification.

This package provides a modular approach to supervised model selection for text classification,
supporting multiple model types through a clean interface. It includes pluggable persistence
for saving and loading models to different storage backends (GCP, AWS, local) with appropriate
serialization methods (pickle for sklearn models, torch for PyTorch models).
"""

from .base import SupervisedModel, SupervisedModelType
from .persistence import (
    ModelPersistence, 
    PickleGCPBucketPersistence, 
    PickleAWSBucketPersistence, 
    PickleLocalFilePersistence,
    TorchGCPBucketPersistence,
    TorchAWSBucketPersistence,
    TorchLocalFilePersistence,
    # Backward compatibility aliases
    GCPBucketPersistence,
    AWSBucketPersistence,
    LocalFilePersistence
)
from .logistic_regression import LogisticRegression
from .pytorch_neural_network import PyTorchNeuralNetwork
from .pytorch_neural_network_simple import SimplePyTorchNeuralNetwork
from .knn_classifier import KNNClassifier
from .factory import create_model

__all__ = [
    'SupervisedModel',
    'SupervisedModelType',
    'ModelPersistence',
    # Pickle persistence classes
    'PickleGCPBucketPersistence',
    'PickleAWSBucketPersistence',
    'PickleLocalFilePersistence',
    # Torch persistence classes
    'TorchGCPBucketPersistence',
    'TorchAWSBucketPersistence',
    'TorchLocalFilePersistence',
    # Backward compatibility aliases (deprecated)
    'GCPBucketPersistence',
    'AWSBucketPersistence',
    'LocalFilePersistence',
    # Model classes
    'LogisticRegression',
    'PyTorchNeuralNetwork',
    'SimplePyTorchNeuralNetwork',
    'KNNClassifier',
    'create_model'
]