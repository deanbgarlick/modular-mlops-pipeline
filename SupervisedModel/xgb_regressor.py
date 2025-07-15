"""XGBoost regressor model implementation using xgboost."""

import numpy as np
from typing import Any, Optional

from .base import SupervisedModel
from .persistence import ModelPersistence


class XGBRegressor(SupervisedModel):
    """XGBoost regressor model using xgboost."""
    
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
            raise ImportError("XGBoost is required for XGBRegressor. Install with: pip install xgboost")
    
    def fit(self, X_train: Any, y_train: Any, class_weights: Optional[dict] = None) -> None:
        """Train the XGBoost regressor model with optional sample weights."""
        xgb = self._check_xgboost_available()
        
        print("Training XGBoost regressor model...")
        
        # Convert class_weights to sample_weights for regression
        # Note: class_weights doesn't make sense for regression, but we support it for interface compatibility
        sample_weight = None
        if class_weights:
            print("Warning: class_weights parameter is ignored for regression. Use sample_weight instead.")
            
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            **self.kwargs
        )
        
        # Fit the model
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
            
        self.is_fitted = True
        print("Model training completed!")
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        """
        Not applicable for regression models.
        Returns None to indicate this is a regression model.
        """
        return None
    
    def get_model_info(self) -> dict:
        """Return information about the XGBoost regressor model."""
        if not self.is_fitted:
            return {"error": "Model not fitted yet"}
        
        return {
            "model_type": "xgb_regressor",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "n_features": self.model.n_features_in_,
            "is_regression": True
        } 