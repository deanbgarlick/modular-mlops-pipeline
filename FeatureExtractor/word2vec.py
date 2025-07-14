"""Word2Vec feature extractor implementation."""

import pandas as pd
import numpy as np
from typing import Tuple, Any, List, Optional

from .base import FeatureExtractor, FeatureMatrix
from .persistence import FeatureExtractorPersistence


class Word2VecExtractor(FeatureExtractor):
    """Feature extractor using Word2Vec for sentence embeddings."""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, 
                 workers: int = 4, sg: int = 0, epochs: int = 10, alpha: float = 0.025, 
                 min_alpha: float = 0.0001, persistence: Optional[FeatureExtractorPersistence] = None):
        """
        Initialize Word2Vec extractor.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum frequency count of words to consider
            workers: Number of worker threads to train the model
            sg: Training algorithm (0 for CBOW, 1 for skip-gram)
            epochs: Number of iterations over the corpus
            alpha: Initial learning rate
            min_alpha: Final learning rate
            persistence: Feature extractor persistence handler
        """
        super().__init__(persistence=persistence)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.model = None
        self.vocab_size = None
        self.is_fitted = False
    
    def _preprocess_text(self, text_series: pd.Series) -> List[List[str]]:
        """Preprocess text data for Word2Vec training."""
        # Simple tokenization - split by whitespace and convert to lowercase
        return [text.lower().split() for text in text_series.fillna('')]
    
    def _sentence_to_vector(self, sentence: List[str]) -> np.ndarray:
        """Convert a sentence (list of words) to a vector by averaging word vectors."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit_transform first.")
        
        vectors = []
        for word in sentence:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
        
        if len(vectors) == 0:
            # If no words in vocabulary, return zero vector
            return np.zeros(self.vector_size)
        
        # Average the word vectors
        return np.mean(vectors, axis=0)
    
    def fit_transform(self, X_train: pd.Series, X_test: pd.Series) -> Tuple[FeatureMatrix, FeatureMatrix]:
        """Train Word2Vec model on training data and transform both sets."""
        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError("gensim package is required. Install with: pip install gensim")
        
        print(f"Training Word2Vec model (vector_size={self.vector_size}, window={self.window}, min_count={self.min_count})...")
        
        # Preprocess training data
        sentences_train = self._preprocess_text(X_train)
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=sentences_train,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
            alpha=self.alpha,
            min_alpha=self.min_alpha
        )
        
        self.vocab_size = len(self.model.wv.key_to_index)
        print(f"Word2Vec model trained with vocabulary size: {self.vocab_size}")
        
        # Transform training data
        print("Transforming training data...")
        X_train_transformed = np.array([
            self._sentence_to_vector(sentence) for sentence in sentences_train
        ])
        
        # Transform test data
        print("Transforming test data...")
        sentences_test = self._preprocess_text(X_test)
        X_test_transformed = np.array([
            self._sentence_to_vector(sentence) for sentence in sentences_test
        ])
        
        self.is_fitted = True
        return X_train_transformed, X_test_transformed
    
    def transform(self, X: List[str]) -> FeatureMatrix:
        """Transform new text data using trained Word2Vec model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit_transform first.")
        
        # Convert to pandas Series for consistency with preprocessing
        X_series = pd.Series(X)
        sentences = self._preprocess_text(X_series)
        
        return np.array([
            self._sentence_to_vector(sentence) for sentence in sentences
        ])
    
    def get_feature_info(self) -> dict:
        """Return information about Word2Vec features."""
        if self.model is None:
            return {"error": "Model not trained yet"}
        
        return {
            "feature_type": "word2vec",
            "vector_size": self.vector_size,
            "vocab_size": self.vocab_size,
            "window": self.window,
            "min_count": self.min_count,
            "sg": "skip-gram" if self.sg == 1 else "cbow",
            "epochs": self.epochs,
            "feature_shape": f"(n_samples, {self.vector_size})"
        }
    
    def save(self, path: str) -> None:
        """
        Save the Word2Vec extractor.
        
        Args:
            path: Path where to save the extractor
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Word2Vec extractor must be fitted before saving")
        
        extractor_data = {
            'model': self.model,
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'workers': self.workers,
            'sg': self.sg,
            'epochs': self.epochs,
            'alpha': self.alpha,
            'min_alpha': self.min_alpha,
            'vocab_size': self.vocab_size,
            'is_fitted': self.is_fitted,
            'feature_info': self.get_feature_info(),
            'extractor_type': self.__class__.__name__
        }
        
        self.persistence.save(extractor_data, path)
    
    def load(self, path: str) -> None:
        """
        Load the Word2Vec extractor.
        
        Args:
            path: Path to load the extractor from
        """
        extractor_data = self.persistence.load(path)
        
        if isinstance(extractor_data, dict):
            # New format with structured data
            self.model = extractor_data.get('model')
            self.vector_size = extractor_data.get('vector_size', 100)
            self.window = extractor_data.get('window', 5)
            self.min_count = extractor_data.get('min_count', 1)
            self.workers = extractor_data.get('workers', 4)
            self.sg = extractor_data.get('sg', 0)
            self.epochs = extractor_data.get('epochs', 10)
            self.alpha = extractor_data.get('alpha', 0.025)
            self.min_alpha = extractor_data.get('min_alpha', 0.0001)
            self.vocab_size = extractor_data.get('vocab_size')
            self.is_fitted = extractor_data.get('is_fitted', True)
        else:
            # Backward compatibility - assume it's a direct model object
            self.model = extractor_data
            self.is_fitted = True
            # Try to get vocab_size from model if available
            if hasattr(self.model, 'wv') and hasattr(self.model.wv, 'key_to_index'):
                self.vocab_size = len(self.model.wv.key_to_index)
        
        if self.model is None:
            raise ValueError("Failed to load model from saved data") 