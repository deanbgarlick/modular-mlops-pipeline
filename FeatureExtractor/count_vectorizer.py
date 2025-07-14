"""Count Vectorizer feature extractor implementation."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple, List, Optional, cast
import scipy.sparse

from .base import FeatureExtractor, FeatureMatrix
from .persistence import FeatureExtractorPersistence


class CountVectorizerExtractor(FeatureExtractor):
    """Feature extractor using sklearn's CountVectorizer."""
    
    def __init__(self, max_features: int = 10000, persistence: Optional[FeatureExtractorPersistence] = None):
        """
        Initialize CountVectorizer extractor.
        
        Args:
            max_features: Maximum number of features to extract
            persistence: Feature extractor persistence handler
        """
        super().__init__(persistence=persistence)
        self.max_features = max_features
        self.vectorizer = None
        self.is_fitted = False
    
    def fit_transform(self, X_train: pd.Series, X_test: pd.Series) -> Tuple[FeatureMatrix, FeatureMatrix]:
        """Fit count vectorizer on training data and transform both sets."""
        print(f"Creating count vectorizer features (max_features={self.max_features})...")
        self.vectorizer = CountVectorizer(max_features=self.max_features, stop_words='english')
        
        X_train_transformed = cast(scipy.sparse.csr_matrix, self.vectorizer.fit_transform(X_train))
        X_test_transformed = cast(scipy.sparse.csr_matrix, self.vectorizer.transform(X_test))
        
        self.is_fitted = True
        return X_train_transformed, X_test_transformed
    
    def transform(self, X: List[str]) -> FeatureMatrix:
        """Transform new text data using fitted vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted yet. Call fit_transform first.")
        return cast(scipy.sparse.csr_matrix, self.vectorizer.transform(X))
    
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
    
    def save(self, path: str) -> None:
        """
        Save the count vectorizer extractor.
        
        Args:
            path: Path where to save the extractor
        """
        if not self.is_fitted or self.vectorizer is None:
            raise ValueError("CountVectorizer extractor must be fitted before saving")
        
        extractor_data = {
            'vectorizer': self.vectorizer,
            'max_features': self.max_features,
            'is_fitted': self.is_fitted,
            'feature_info': self.get_feature_info(),
            'extractor_type': self.__class__.__name__
        }
        
        self.persistence.save(extractor_data, path)
    
    def load(self, path: str) -> None:
        """
        Load the count vectorizer extractor.
        
        Args:
            path: Path to load the extractor from
        """
        extractor_data = self.persistence.load(path)
        
        if isinstance(extractor_data, dict):
            # New format with structured data
            self.vectorizer = extractor_data.get('vectorizer')
            self.max_features = extractor_data.get('max_features', 10000)
            self.is_fitted = extractor_data.get('is_fitted', True)
        else:
            # Backward compatibility - assume it's a direct vectorizer object
            self.vectorizer = extractor_data
            self.is_fitted = True
        
        if self.vectorizer is None:
            raise ValueError("Failed to load vectorizer from saved data") 