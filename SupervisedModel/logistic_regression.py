"""Logistic regression model implementation using sklearn."""

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from typing import Any, Optional

from .base import SupervisedModel
from .persistence import ModelPersistence


class LogisticRegression(SupervisedModel):
    """Logistic regression model using sklearn."""
    
    def __init__(self, random_state: int = 42, max_iter: int = 1000, 
                 persistence: Optional[ModelPersistence] = None, **kwargs):
        super().__init__(persistence=persistence)
        self.random_state = random_state
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
    
    def fit(self, X_train: Any, y_train: Any, class_weights: Optional[dict] = None) -> None:
        """Train the logistic regression model with optional class weights."""
        print("Training logistic regression model...")
        
        # Set up class weights
        if class_weights:
            print(f"Using class weights: {class_weights}")
            # Convert to sklearn format
            class_weight = class_weights
        else:
            class_weight = None
            
        self.model = SklearnLogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter,
            class_weight=class_weight,
            **self.kwargs
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print("Model training completed!")
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """Make probability predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_model_info(self) -> dict:
        """Return information about the logistic regression model."""
        if not self.is_fitted:
            return {"error": "Model not fitted yet"}
        
        return {
            "model_type": "logistic_regression",
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "n_features": self.model.n_features_in_,
            "n_classes": len(self.model.classes_),
            "classes": self.model.classes_.tolist()
        } 