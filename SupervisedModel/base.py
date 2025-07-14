"""Base classes and enums for supervised models."""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Tuple, Optional

from .persistence import ModelPersistence, PickleGCPBucketPersistence, TorchGCPBucketPersistence


class SupervisedModelType(Enum):
    """Enum for different supervised model types."""
    LOGISTIC_REGRESSION = "logistic_regression"
    PYTORCH_NEURAL_NETWORK = "pytorch_neural_network"
    SIMPLE_PYTORCH_NEURAL_NETWORK = "simple_pytorch_neural_network"
    KNN_CLASSIFIER = "knn_classifier"


class SupervisedModel(ABC):
    """Abstract base class for supervised models."""
    
    def __init__(self, persistence: Optional[ModelPersistence] = None):
        """
        Initialize the supervised model.
        
        Args:
            persistence: Model persistence handler. If None, uses appropriate default 
                        (PickleGCPBucketPersistence for most models, TorchGCPBucketPersistence for PyTorch models).
        """
        if persistence is None:
            # Choose appropriate persistence based on model type
            if self._is_torch_model():
                self.persistence = TorchGCPBucketPersistence("default-model-bucket")
            else:
                self.persistence = PickleGCPBucketPersistence("default-model-bucket")
        else:
            self.persistence = persistence
    
    def _is_torch_model(self) -> bool:
        """Check if this is a PyTorch model that should use torch persistence."""
        model_class_name = self.__class__.__name__
        return 'PyTorch' in model_class_name or 'Torch' in model_class_name
    
    @abstractmethod
    def fit(self, X_train: Any, y_train: Any, class_weights: Optional[dict] = None) -> None:
        """
        Train the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weights: Optional dictionary of class weights for handling imbalance
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Make probability predictions on input data.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Return information about the model."""
        pass
    
    def save(self, path: str) -> None:
        """
        Save the model using the configured persistence handler.
        
        Args:
            path: Path where to save the model
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'model_info': self.get_model_info(),
            'model_type': self.__class__.__name__
        }
        
        self.persistence.save(model_data, path)
    
    def load(self, path: str) -> None:
        """
        Load the model using the configured persistence handler.
        
        Args:
            path: Path to load the model from
        """
        model_data = self.persistence.load(path)
        
        if not isinstance(model_data, dict):
            raise ValueError("Invalid model data format")
        
        if 'model' not in model_data:
            raise ValueError("Model data missing 'model' key")
        
        self.model = model_data['model']
        
        # Set fitted flag if it exists
        if hasattr(self, 'is_fitted'):
            self.is_fitted = True
    
    @classmethod
    def load_from_path(cls, path: str, persistence: Optional[ModelPersistence] = None):
        """
        Class method to create a new instance and load a model from path.
        
        Args:
            path: Path to load the model from
            persistence: Model persistence handler. If None, uses appropriate default 
                        (PickleGCPBucketPersistence for most models, TorchGCPBucketPersistence for PyTorch models).
            
        Returns:
            SupervisedModel: New instance with loaded model
        """
        instance = cls(persistence=persistence)
        instance.load(path)
        return instance 