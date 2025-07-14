"""Factory function for creating feature extractors."""

from .base import FeatureExtractor, FeatureExtractorType
from .count_vectorizer import CountVectorizerExtractor
from .huggingface import HuggingFaceExtractor


def create_feature_extractor(extractor_type: FeatureExtractorType, **kwargs) -> FeatureExtractor:
    """Create and return the appropriate feature extractor.
    
    Args:
        extractor_type: The type of feature extractor to create
        **kwargs: Additional arguments to pass to the extractor constructor
        
    Returns:
        FeatureExtractor: The created feature extractor instance
        
    Raises:
        ValueError: If extractor_type is not supported
    """
    if extractor_type == FeatureExtractorType.COUNT_VECTORIZER:
        return CountVectorizerExtractor(**kwargs)
    elif extractor_type == FeatureExtractorType.HUGGINGFACE_TRANSFORMER:
        return HuggingFaceExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}") 