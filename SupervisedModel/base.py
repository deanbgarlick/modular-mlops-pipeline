"""Base classes and enums for supervised models."""

import numpy as np
import pandas as pd
import scipy.sparse
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Tuple, Optional, Union

from .persistence import ModelPersistence, PickleGCPBucketPersistence, TorchGCPBucketPersistence

# Type alias for feature matrices that can be used with supervised models
FeatureMatrix = Union[np.ndarray, scipy.sparse.csr_matrix, pd.DataFrame]
# Type alias for target labels
TargetLabels = Union[np.ndarray, pd.Series]


class SupervisedModelType(Enum):
    """Enum for different supervised model types."""
    LOGISTIC_REGRESSION = "logistic_regression"
    PYTORCH_NEURAL_NETWORK = "pytorch_neural_network"
    SIMPLE_PYTORCH_NEURAL_NETWORK = "simple_pytorch_neural_network"
    KNN_CLASSIFIER = "knn_classifier"
    XGB_CLASSIFIER = "xgb_classifier"
    XGB_REGRESSOR = "xgb_regressor"


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
    def fit(self, X_train: FeatureMatrix, y_train: TargetLabels, class_weights: Optional[dict] = None) -> None:
        """
        Train the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weights: Optional dictionary of class weights for handling imbalance
        """
        pass
    
    @abstractmethod
    def predict(self, X: FeatureMatrix) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: FeatureMatrix) -> np.ndarray:
        """
        Make probability predictions on input data.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        pass
    
    def fit_transform(self, X_train: FeatureMatrix, X_test: Optional[FeatureMatrix] = None, 
                     y_train: Optional[TargetLabels] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Fit the model and return predictions/probabilities for training and optionally test data.
        This method makes SupervisedModel compatible with the TransformationStep protocol.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            y_train: Training labels (optional, used if model needs to be fitted)
            
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                - If X_test is None: predictions/probabilities for X_train
                - If X_test provided: tuple of (X_train_predictions, X_test_predictions)
        """
        if y_train is not None:
            self.fit(X_train, y_train)
        
        train_predictions = self.predict_proba(X_train)
        
        if X_test is not None:
            test_predictions = self.predict_proba(X_test)
            return train_predictions, test_predictions
        else:
            return train_predictions
    
    def transform(self, X: FeatureMatrix) -> np.ndarray:
        """
        Transform input features to predictions/probabilities.
        This method makes SupervisedModel compatible with the TransformationStep protocol.
        
        Args:
            X: Input features to transform
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        return self.predict_proba(X)
    
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
        
        # Create a copy of the model without persistence object (which contains unpickleable clients)
        model_copy = self.__class__.__new__(self.__class__)
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name != 'persistence':
                attr_value = getattr(self, attr_name)
                if not callable(attr_value):
                    setattr(model_copy, attr_name, attr_value)
        
        model_data = {
            'model': self.model,
            'model_copy': model_copy,
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
        
        # Load model copy attributes if available
        if 'model_copy' in model_data:
            model_copy = model_data['model_copy']
            for attr_name in dir(model_copy):
                if not attr_name.startswith('_') and hasattr(model_copy, attr_name):
                    attr_value = getattr(model_copy, attr_name)
                    if not callable(attr_value):
                        setattr(self, attr_name, attr_value)
        
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