"""XGBoost classifier model implementation using xgboost."""

import numpy as np
from typing import Any, Optional

from .base import SupervisedModel
from .persistence import ModelPersistence


class XGBClassifier(SupervisedModel):
    """XGBoost classifier model using xgboost."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, random_state: int = 42,
                 persistence: Optional[ModelPersistence] = None, **kwargs):
        super().__init__(persistence=persistence)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs
        self.model: Optional[Any] = None
        self.is_fitted = False
    
    def _check_xgboost_available(self):
        """Check if XGBoost is available."""
        try:
            import xgboost as xgb
            return xgb
        except ImportError:
            raise ImportError("XGBoost is required for XGBClassifier. Install with: pip install xgboost")
    
    def fit(self, X_train: Any, y_train: Any, class_weights: Optional[dict] = None) -> None:
        """Train the XGBoost classifier model with optional class weights."""
        xgb = self._check_xgboost_available()
        
        print("Training XGBoost classifier model...")
        
        # Set up class weights
        sample_weight = None
        if class_weights:
            print(f"Using class weights: {class_weights}")
            # Convert class weights to sample weights
            sample_weight = np.array([class_weights.get(label, 1.0) for label in y_train])
            
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            **self.kwargs
        )
        
        # Fit with sample weights if provided
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
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
        """Return information about the XGBoost classifier model."""
        if not self.is_fitted:
            return {"error": "Model not fitted yet"}
        
        return {
            "model_type": "xgb_classifier",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "n_features": self.model.n_features_in_,
            "n_classes": len(self.model.classes_),
            "classes": self.model.classes_.tolist()
        } 