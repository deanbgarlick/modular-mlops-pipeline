"""HuggingFace transformer feature extractor implementation."""

import pandas as pd
from typing import Tuple, Any

from .base import FeatureExtractor


class HuggingFaceExtractor(FeatureExtractor):
    """Feature extractor using HuggingFace transformers for sentence embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
    
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