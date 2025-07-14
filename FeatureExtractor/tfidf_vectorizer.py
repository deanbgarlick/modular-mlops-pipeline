"""TF-IDF Vectorizer feature extractor implementation."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Any, Optional

from .base import FeatureExtractor
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
    
    def fit_transform(self, X_train: pd.Series, X_test: pd.Series) -> Tuple[Any, Any]:
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
        
        X_train_transformed = self.vectorizer.fit_transform(X_train)
        X_test_transformed = self.vectorizer.transform(X_test)
        
        self.is_fitted = True
        return X_train_transformed, X_test_transformed
    
    def transform(self, X: list) -> Any:
        """Transform new text data using fitted vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted yet. Call fit_transform first.")
        return self.vectorizer.transform(X)
    
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