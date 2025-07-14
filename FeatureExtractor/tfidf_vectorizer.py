"""TF-IDF Vectorizer feature extractor implementation."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List, Optional, cast
import scipy.sparse

from .base import FeatureExtractor, FeatureMatrix
from .persistence import FeatureExtractorPersistence


class TfidfVectorizerExtractor(FeatureExtractor):
    """Feature extractor using sklearn's TfidfVectorizer."""
    
    def __init__(self, max_features: int = 10000, min_df: int = 1, max_df: float = 1.0, 
                 use_idf: bool = True, smooth_idf: bool = True, sublinear_tf: bool = False,
                 persistence: Optional[FeatureExtractorPersistence] = None):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency for a term to be included
            max_df: Maximum document frequency for a term to be included (fraction or absolute count)
            use_idf: Enable inverse-document-frequency reweighting
            smooth_idf: Smooth idf weights by adding 1 to document frequencies
            sublinear_tf: Apply sublinear tf scaling (replace tf with 1 + log(tf))
            persistence: Feature extractor persistence handler
        """
        super().__init__(persistence=persistence)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.vectorizer = None
        self.is_fitted = False
    
    def fit_transform(self, X_train: pd.Series, X_test: pd.Series) -> Tuple[FeatureMatrix, FeatureMatrix]:
        """Fit TF-IDF vectorizer on training data and transform both sets."""
        print(f"Creating TF-IDF vectorizer features (max_features={self.max_features}, min_df={self.min_df}, max_df={self.max_df})...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
            stop_words='english'
        )
        
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
        """Return information about TF-IDF vectorizer features."""
        if self.vectorizer is None:
            return {"error": "Vectorizer not fitted yet"}
        
        return {
            "feature_type": "tfidf_vectorizer",
            "vocab_size": len(self.vectorizer.vocabulary_),
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "use_idf": self.use_idf,
            "smooth_idf": self.smooth_idf,
            "sublinear_tf": self.sublinear_tf,
            "feature_shape": f"(n_samples, {len(self.vectorizer.vocabulary_)})"
        }
    
    def save(self, path: str) -> None:
        """
        Save the TF-IDF vectorizer extractor.
        
        Args:
            path: Path where to save the extractor
        """
        if not self.is_fitted or self.vectorizer is None:
            raise ValueError("TfidfVectorizer extractor must be fitted before saving")
        
        extractor_data = {
            'vectorizer': self.vectorizer,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'use_idf': self.use_idf,
            'smooth_idf': self.smooth_idf,
            'sublinear_tf': self.sublinear_tf,
            'is_fitted': self.is_fitted,
            'feature_info': self.get_feature_info(),
            'extractor_type': self.__class__.__name__
        }
        
        self.persistence.save(extractor_data, path)
    
    def load(self, path: str) -> None:
        """
        Load the TF-IDF vectorizer extractor.
        
        Args:
            path: Path to load the extractor from
        """
        extractor_data = self.persistence.load(path)
        
        if isinstance(extractor_data, dict):
            # New format with structured data
            self.vectorizer = extractor_data.get('vectorizer')
            self.max_features = extractor_data.get('max_features', 10000)
            self.min_df = extractor_data.get('min_df', 1)
            self.max_df = extractor_data.get('max_df', 1.0)
            self.use_idf = extractor_data.get('use_idf', True)
            self.smooth_idf = extractor_data.get('smooth_idf', True)
            self.sublinear_tf = extractor_data.get('sublinear_tf', False)
            self.is_fitted = extractor_data.get('is_fitted', True)
        else:
            # Backward compatibility - assume it's a direct vectorizer object
            self.vectorizer = extractor_data
            self.is_fitted = True
        
        if self.vectorizer is None:
            raise ValueError("Failed to load vectorizer from saved data") 