"""HuggingFace transformer feature extractor implementation."""

import pandas as pd
from typing import Tuple, Any, Optional

from .base import FeatureExtractor
from .persistence import FeatureExtractorPersistence


class HuggingFaceExtractor(FeatureExtractor):
    """Feature extractor using HuggingFace transformers for sentence embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persistence: Optional[FeatureExtractorPersistence] = None):
        """
        Initialize HuggingFace transformer extractor.
        
        Args:
            model_name: Name of the HuggingFace model to use
            persistence: Feature extractor persistence handler
        """
        super().__init__(persistence=persistence)
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self.is_fitted = False
    
    def fit_transform(self, X_train: pd.Series, X_test: pd.Series) -> Tuple[Any, Any]:
        """Create sentence embeddings using HuggingFace transformer."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers package is required. Install with: pip install sentence-transformers")
        
        print(f"Loading HuggingFace transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        print("Creating sentence embeddings...")
        X_train_transformed = self.model.encode(X_train.tolist(), show_progress_bar=True)
        X_test_transformed = self.model.encode(X_test.tolist(), show_progress_bar=True)
        
        self.embedding_dim = X_train_transformed.shape[1]
        self.is_fitted = True
        
        return X_train_transformed, X_test_transformed
    
    def transform(self, X: list) -> Any:
        """Transform new text data using fitted transformer."""
        if self.model is None:
            raise ValueError("Model not loaded yet. Call fit_transform first.")
        return self.model.encode(X)
    
    def get_feature_info(self) -> dict:
        """Return information about transformer embeddings."""
        if self.model is None:
            return {"error": "Model not loaded yet"}
        
        return {
            "feature_type": "huggingface_transformer",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "feature_shape": f"(n_samples, {self.embedding_dim})"
        }
    
    def save(self, path: str) -> None:
        """
        Save the HuggingFace transformer extractor.
        
        Args:
            path: Path where to save the extractor
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("HuggingFace extractor must be fitted before saving")
        
        extractor_data = {
            'model': self.model,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'is_fitted': self.is_fitted,
            'feature_info': self.get_feature_info(),
            'extractor_type': self.__class__.__name__
        }
        
        self.persistence.save(extractor_data, path)
    
    def load(self, path: str) -> None:
        """
        Load the HuggingFace transformer extractor.
        
        Args:
            path: Path to load the extractor from
        """
        extractor_data = self.persistence.load(path)
        
        if isinstance(extractor_data, dict):
            # New format with structured data
            self.model = extractor_data.get('model')
            self.model_name = extractor_data.get('model_name', "sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_dim = extractor_data.get('embedding_dim')
            self.is_fitted = extractor_data.get('is_fitted', True)
        else:
            # Backward compatibility - assume it's a direct model object
            self.model = extractor_data
            self.is_fitted = True
            # Try to get embedding dimension from model if available
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        if self.model is None:
            raise ValueError("Failed to load model from saved data") 