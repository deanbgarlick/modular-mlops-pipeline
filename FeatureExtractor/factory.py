"""Factory function for creating feature extractors."""

import os
from typing import Optional

from .base import FeatureExtractor, FeatureExtractorType
from .count_vectorizer import CountVectorizerExtractor
from .tfidf_vectorizer import TfidfVectorizerExtractor
from .huggingface import HuggingFaceExtractor
from .word2vec import Word2VecExtractor
from .openai_embeddings import OpenAIEmbeddingsExtractor
from .persistence import (
    FeatureExtractorPersistence, 
    PickleGCPExtractorPersistence, 
    HuggingFaceExtractorPersistence
)


def create_feature_extractor(extractor_type: FeatureExtractorType, 
                           persistence: Optional[FeatureExtractorPersistence] = None,
                           default_bucket_name: Optional[str] = None,
                           **kwargs) -> FeatureExtractor:
    """Create and return the appropriate feature extractor.
    
    Args:
        extractor_type: The type of feature extractor to create
        persistence: Optional persistence handler for saving/loading extractors.
                    If None, creates appropriate GCP persistence based on extractor type.
        default_bucket_name: Default GCP bucket name to use when persistence is None.
                           Falls back to FEATURE_EXTRACTOR_BUCKET env var or "default-extractor-bucket"
        **kwargs: Additional arguments to pass to the extractor constructor
        
    Returns:
        FeatureExtractor: The created feature extractor instance
        
    Raises:
        ValueError: If extractor_type is not supported
        
    Examples:
        # Basic usage with automatic GCP persistence
        extractor = create_feature_extractor(FeatureExtractorType.TFIDF_VECTORIZER)
        
        # With custom bucket name
        extractor = create_feature_extractor(
            FeatureExtractorType.TFIDF_VECTORIZER,
            default_bucket_name="my-custom-bucket",
            max_features=5000,
            min_df=2
        )
        
        # With explicit persistence
        from .persistence import PickleGCPExtractorPersistence
        persistence = PickleGCPExtractorPersistence("my-bucket")
        extractor = create_feature_extractor(
            FeatureExtractorType.COUNT_VECTORIZER,
            persistence=persistence,
            max_features=10000
        )
    """
    # Create default GCP persistence if none provided
    if persistence is None:
        # Get bucket name from parameter, environment variable, or default
        bucket_name = (
            default_bucket_name or 
            os.getenv('FEATURE_EXTRACTOR_BUCKET', 'default-extractor-bucket')
        )
        
        # Choose appropriate persistence based on extractor type
        if extractor_type == FeatureExtractorType.HUGGINGFACE_TRANSFORMER:
            persistence = HuggingFaceExtractorPersistence(
                bucket_name=bucket_name, 
                use_gcp=True
            )
        else:
            persistence = PickleGCPExtractorPersistence(bucket_name)
    
    # Add persistence to kwargs
    kwargs['persistence'] = persistence
    
    if extractor_type == FeatureExtractorType.COUNT_VECTORIZER:
        return CountVectorizerExtractor(**kwargs)
    elif extractor_type == FeatureExtractorType.TFIDF_VECTORIZER:
        return TfidfVectorizerExtractor(**kwargs)
    elif extractor_type == FeatureExtractorType.HUGGINGFACE_TRANSFORMER:
        return HuggingFaceExtractor(**kwargs)
    elif extractor_type == FeatureExtractorType.WORD2VEC:
        return Word2VecExtractor(**kwargs)
    elif extractor_type == FeatureExtractorType.OPENAI_EMBEDDINGS:
        return OpenAIEmbeddingsExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}") 