"""OpenAI embeddings feature extractor implementation."""

import pandas as pd
from typing import Tuple, List, Optional
import numpy as np
import os
import time

from .base import FeatureExtractor, FeatureMatrix
from .persistence import FeatureExtractorPersistence


class OpenAIEmbeddingsExtractor(FeatureExtractor):
    """Feature extractor using OpenAI's text-embedding models."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None, 
                 batch_size: int = 100, persistence: Optional[FeatureExtractorPersistence] = None):
        """
        Initialize OpenAI embeddings extractor.
        
        Args:
            model_name: OpenAI embedding model name (default: text-embedding-3-small)
            api_key: OpenAI API key. If None, will look for OPENAI_API_KEY environment variable
            batch_size: Number of texts to process in each batch to avoid rate limits
            persistence: Feature extractor persistence handler
        """
        super().__init__(persistence=persistence)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.batch_size = batch_size
        self.client = None
        self.embedding_dim = None
        self.is_fitted = False
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package is required. Install with: pip install openai")
            
            self.client = OpenAI(api_key=self.api_key)
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        if self.client is None:
            raise ValueError("Client not initialized")
            
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            if response and response.data:
                return [embedding.embedding for embedding in response.data]
            else:
                raise ValueError("Empty response from OpenAI API")
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            # Return zero embeddings as fallback
            if self.embedding_dim is None:
                # Try to get embedding dimension from a single text
                try:
                    test_response = self.client.embeddings.create(
                        model=self.model_name,
                        input=["test"]
                    )
                    if test_response and test_response.data:
                        self.embedding_dim = len(test_response.data[0].embedding)
                    else:
                        self.embedding_dim = 1536  # Default for text-embedding-3-small
                except:
                    self.embedding_dim = 1536  # Default for text-embedding-3-small
            return [[0.0] * self.embedding_dim for _ in texts]
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts, processing in batches."""
        self._initialize_client()
        
        all_embeddings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            batch_embeddings = self._get_embeddings_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Set embedding dimension from first successful batch
            if self.embedding_dim is None and batch_embeddings:
                self.embedding_dim = len(batch_embeddings[0])
            
            # Small delay to respect rate limits
            if i + self.batch_size < len(texts):
                time.sleep(0.1)
        
        return np.array(all_embeddings)
    
    def fit_transform(self, X_train: pd.Series, X_test: pd.Series) -> Tuple[FeatureMatrix, FeatureMatrix]:
        """Create embeddings using OpenAI API for both train and test sets."""
        print(f"Creating OpenAI embeddings using model: {self.model_name}")
        
        # Convert to lists and handle NaN values
        train_texts = X_train.fillna("").tolist()
        test_texts = X_test.fillna("").tolist()
        
        print(f"Getting embeddings for {len(train_texts)} training samples...")
        X_train_transformed = self._get_embeddings(train_texts)
        
        print(f"Getting embeddings for {len(test_texts)} test samples...")
        X_test_transformed = self._get_embeddings(test_texts)
        
        print(f"Embeddings created with dimension: {self.embedding_dim}")
        
        self.is_fitted = True
        return X_train_transformed, X_test_transformed
    
    def transform(self, X: List[str]) -> FeatureMatrix:
        """Transform new text data using OpenAI embeddings."""
        if self.client is None:
            raise ValueError("Client not initialized. Call fit_transform first.")
        
        # Handle empty or None values
        texts = [str(text) if text is not None else "" for text in X]
        
        return self._get_embeddings(texts)
    
    def get_feature_info(self) -> dict:
        """Return information about OpenAI embeddings."""
        if self.embedding_dim is None:
            return {"error": "Embeddings not created yet"}
        
        return {
            "feature_type": "openai_embeddings",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "feature_shape": f"(n_samples, {self.embedding_dim})",
            "batch_size": self.batch_size
        }
    
    def save(self, path: str) -> None:
        """
        Save the OpenAI embeddings extractor configuration.
        
        Args:
            path: Path where to save the extractor
        """
        if not self.is_fitted:
            raise ValueError("OpenAI embeddings extractor must be fitted before saving")
        
        # Note: We don't save the API key for security reasons
        extractor_data = {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'embedding_dim': self.embedding_dim,
            'is_fitted': self.is_fitted,
            'feature_info': self.get_feature_info(),
            'extractor_type': self.__class__.__name__
        }
        
        self.persistence.save(extractor_data, path)
    
    def load(self, path: str) -> None:
        """
        Load the OpenAI embeddings extractor configuration.
        
        Args:
            path: Path to load the extractor from
        """
        extractor_data = self.persistence.load(path)
        
        if isinstance(extractor_data, dict):
            # New format with structured data
            self.model_name = extractor_data.get('model_name', "text-embedding-3-small")
            self.batch_size = extractor_data.get('batch_size', 100)
            self.embedding_dim = extractor_data.get('embedding_dim')
            self.is_fitted = extractor_data.get('is_fitted', True)
        else:
            # Backward compatibility - assume it's configuration data
            self.is_fitted = True
        
        # Re-initialize API key from environment (for security, we don't persist it)
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Reset client to be re-initialized on next use
        self.client = None 