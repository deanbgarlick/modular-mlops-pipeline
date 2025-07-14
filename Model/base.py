"""Base classes and enums for models."""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Tuple, Optional


class ModelType(Enum):
    """Enum for different model types."""
    LOGISTIC_REGRESSION = "logistic_regression"
    PYTORCH_NEURAL_NETWORK = "pytorch_neural_network"


class Model(ABC):
    """Abstract base class for models."""
    
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