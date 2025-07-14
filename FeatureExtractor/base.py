"""Base classes and enums for feature extractors."""

import pandas as pd
import numpy as np
import scipy.sparse
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional, Union, List

from .persistence import FeatureExtractorPersistence

# Type alias for feature matrices that can be returned by extractors
FeatureMatrix = Union[np.ndarray, scipy.sparse.csr_matrix, pd.DataFrame]


class FeatureExtractorType(Enum):
    """Enum for different feature extraction methods."""
    COUNT_VECTORIZER = "count_vectorizer"
    TFIDF_VECTORIZER = "tfidf_vectorizer"
    HUGGINGFACE_TRANSFORMER = "huggingface_transformer"
    WORD2VEC = "word2vec"
    OPENAI_EMBEDDINGS = "openai_embeddings"


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, persistence: Optional[FeatureExtractorPersistence] = None):
        """
        Initialize the feature extractor.
        
        Args:
            persistence: Feature extractor persistence handler. If None, uses appropriate default 
                        (PickleGCPExtractorPersistence for most extractors, HuggingFaceExtractorPersistence for transformers).
        """
        if persistence is None:
            raise ValueError("Persistence is required for feature extractors")

        self.persistence = persistence
        # Choose appropriate persistence based on extractor type
        #     if self._is_huggingface_extractor():
        #         self.persistence = HuggingFaceExtractorPersistence(bucket_name="default-extractor-bucket")
        #     else:
        #         self.persistence = PickleGCPExtractorPersistence("default-extractor-bucket")
        # else:
        #     self.persistence = persistence
    
    def _is_huggingface_extractor(self) -> bool:
        """Check if this is a HuggingFace transformer that should use specialized persistence."""
        extractor_class_name = self.__class__.__name__
        return 'HuggingFace' in extractor_class_name or 'Transformer' in extractor_class_name
    
    @abstractmethod
    def fit_transform(self, X_train: pd.Series, X_test: pd.Series) -> Tuple[FeatureMatrix, FeatureMatrix]:
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
    def transform(self, X: List[str]) -> FeatureMatrix:
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
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the feature extractor using the configured persistence handler.
        
        Args:
            path: Path where to save the extractor
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the feature extractor using the configured persistence handler.
        
        Args:
            path: Path to load the extractor from
        """
        pass
    
    @classmethod
    def load_from_path(cls, path: str, persistence: Optional[FeatureExtractorPersistence] = None):
        """
        Class method to create a new instance and load a feature extractor from path.
        
        Args:
            path: Path to load the extractor from
            persistence: Feature extractor persistence handler. If None, uses appropriate default.
            
        Returns:
            FeatureExtractor: New instance with loaded extractor
        """
        instance = cls(persistence=persistence)
        instance.load(path)
        return instance 