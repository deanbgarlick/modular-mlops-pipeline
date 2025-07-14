"""Count Vectorizer feature extractor implementation."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple, Any

from .base import FeatureExtractor


class CountVectorizerExtractor(FeatureExtractor):
    """Feature extractor using sklearn's CountVectorizer."""
    
    def __init__(self, max_features: int = 10000):
        self.max_features = max_features
        self.vectorizer = None
    
    def fit_transform(self, X_train: pd.Series, X_test: pd.Series) -> Tuple[Any, Any]:
        """Fit count vectorizer on training data and transform both sets."""
        print(f"Creating count vectorizer features (max_features={self.max_features})...")
        self.vectorizer = CountVectorizer(max_features=self.max_features, stop_words='english')
        
        X_train_transformed = self.vectorizer.fit_transform(X_train)
        X_test_transformed = self.vectorizer.transform(X_test)
        
        return X_train_transformed, X_test_transformed
    
    def transform(self, X: list) -> Any:
        """Transform new text data using fitted vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted yet. Call fit_transform first.")
        return self.vectorizer.transform(X)
    
    def get_feature_info(self) -> dict:
        """Return information about count vectorizer features."""
        if self.vectorizer is None:
            return {"error": "Vectorizer not fitted yet"}
        
        return {
            "feature_type": "count_vectorizer",
            "vocab_size": len(self.vectorizer.vocabulary_),
            "max_features": self.max_features,
            "feature_shape": f"(n_samples, {len(self.vectorizer.vocabulary_)})"
        } 