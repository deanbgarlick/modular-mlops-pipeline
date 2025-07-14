"""Base classes and enums for feature extractors."""

import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Any, Optional

from .persistence import FeatureExtractorPersistence, PickleGCPExtractorPersistence, HuggingFaceExtractorPersistence


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
            # Choose appropriate persistence based on extractor type
            if self._is_huggingface_extractor():
                self.persistence = HuggingFaceExtractorPersistence(bucket_name="default-extractor-bucket")
            else:
                self.persistence = PickleGCPExtractorPersistence("default-extractor-bucket")
        else:
            self.persistence = persistence
    
    def _is_huggingface_extractor(self) -> bool:
        """Check if this is a HuggingFace transformer that should use specialized persistence."""
        extractor_class_name = self.__class__.__name__
        return 'HuggingFace' in extractor_class_name or 'Transformer' in extractor_class_name
    
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
    
    def save(self, path: str) -> None:
        """
        Save the feature extractor using the configured persistence handler.
        
        Args:
            path: Path where to save the extractor
        """
        if not hasattr(self, 'vectorizer') and not hasattr(self, 'model') and not hasattr(self, 'is_fitted'):
            raise ValueError("Feature extractor must be fitted before saving")
        
        # Create a copy of the extractor without persistence object (which contains unpickleable clients)
        extractor_copy = self.__class__.__new__(self.__class__)
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name != 'persistence':
                attr_value = getattr(self, attr_name)
                if not callable(attr_value):
                    setattr(extractor_copy, attr_name, attr_value)
        
        extractor_data = {
            'extractor': extractor_copy,
            'feature_info': self.get_feature_info(),
            'extractor_type': self.__class__.__name__
        }
        
        # For extractors with internal models/vectorizers, save them separately for better compatibility
        if hasattr(self, 'vectorizer'):
            extractor_data['vectorizer'] = self.vectorizer
        if hasattr(self, 'model'):
            extractor_data['model'] = self.model
        if hasattr(self, 'tokenizer'):
            extractor_data['tokenizer'] = self.tokenizer
        
        self.persistence.save(extractor_data, path)
    
    def load(self, path: str) -> None:
        """
        Load the feature extractor using the configured persistence handler.
        
        Args:
            path: Path to load the extractor from
        """
        extractor_data = self.persistence.load(path)
        
        if isinstance(extractor_data, dict):
            # New format with structured data
            if 'extractor' in extractor_data:
                loaded_extractor = extractor_data['extractor']
                # Copy attributes from loaded extractor
                for attr_name in dir(loaded_extractor):
                    if not attr_name.startswith('_') and hasattr(loaded_extractor, attr_name):
                        attr_value = getattr(loaded_extractor, attr_name)
                        if not callable(attr_value):
                            setattr(self, attr_name, attr_value)
            
            # Load specific components
            if 'vectorizer' in extractor_data:
                self.vectorizer = extractor_data['vectorizer']
            if 'model' in extractor_data:
                self.model = extractor_data['model']
            if 'tokenizer' in extractor_data:
                self.tokenizer = extractor_data['tokenizer']
        else:
            # Backward compatibility - direct extractor object
            loaded_extractor = extractor_data
            for attr_name in dir(loaded_extractor):
                if not attr_name.startswith('_') and hasattr(loaded_extractor, attr_name):
                    attr_value = getattr(loaded_extractor, attr_name)
                    if not callable(attr_value):
                        setattr(self, attr_name, attr_value)
        
        # Set fitted flag
        self.is_fitted = True
    
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