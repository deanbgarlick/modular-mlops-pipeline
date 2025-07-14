"""Base classes and enums for feature extractors."""

import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Any


class FeatureExtractorType(Enum):
    """Enum for different feature extraction methods."""
    COUNT_VECTORIZER = "count_vectorizer"
    TFIDF_VECTORIZER = "tfidf_vectorizer"
    HUGGINGFACE_TRANSFORMER = "huggingface_transformer"


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    @abstractmethod
    def fit_transform(self, X_train: pd.Series, X_test: pd.Series) -> Tuple[Any, Any]:
        """
        Fit on training data and transform both train and test sets.
        
        Args:
            X_train: Training text data
            X_test: Test text data
            
        Returns:
            Tuple of (X_train_transformed, X_test_transformed)
        """
        pass
    
    @abstractmethod
    def transform(self, X: list) -> Any:
        """
        Transform new text data using fitted extractor.
        
        Args:
            X: List of text strings to transform
            
        Returns:
            Transformed features
        """
        pass
    
    @abstractmethod
    def get_feature_info(self) -> dict:
        """Return information about the features created."""
        pass 