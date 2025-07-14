"""K-Nearest Neighbors classifier implementation."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
from typing import Optional

from .base import Model


class KNNClassifier(Model):
    """K-Nearest Neighbors classifier implementation."""
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', 
                 algorithm: str = 'auto', metric: str = 'minkowski', p: int = 2):
        """
        Initialize KNN classifier.
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function used in prediction ('uniform', 'distance')
            algorithm: Algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
            metric: Distance metric to use ('minkowski', 'euclidean', 'manhattan', etc.)
            p: Power parameter for the Minkowski metric (1=manhattan, 2=euclidean)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric,
            p=p
        )
        self.is_fitted = False
    
    def fit(self, X_train, y_train, class_weights: Optional[dict] = None) -> None:
        """
        Train the KNN classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weights: Optional dictionary of class weights for handling imbalance
        """
        # Note: KNeighborsClassifier doesn't support sample weights directly
        # For KNN, class weights are typically handled by using weighted voting
        # We'll use the 'distance' weights mode when class_weights are specified
        if class_weights is not None:
            print(f"Class weights specified for KNN - using 'distance' weighting in voting")
            # Create a new model with distance-based weighting
            self.model = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights='distance',  # Use distance-based weighting instead of uniform
                algorithm=self.algorithm,
                metric=self.metric,
                p=self.p
            )
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the KNN classifier.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities using the KNN classifier.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_model_info(self) -> dict:
        """
        Get information about the model.
        
        Returns:
            dict: Model configuration and status
        """
        return {
            'model_type': 'K-Nearest Neighbors',
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'metric': self.metric,
            'p': self.p,
            'is_fitted': self.is_fitted,
            'n_features_in': getattr(self.model, 'n_features_in_', 'Not fitted'),
            'classes': self.model.classes_.tolist() if hasattr(self.model, 'classes_') and self.is_fitted else 'Not fitted'
        } 