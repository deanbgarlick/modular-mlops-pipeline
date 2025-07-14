"""Factory function for creating feature extractors."""

from typing import Optional

from .base import FeatureExtractor, FeatureExtractorType
from .count_vectorizer import CountVectorizerExtractor
from .tfidf_vectorizer import TfidfVectorizerExtractor
from .huggingface import HuggingFaceExtractor
from .word2vec import Word2VecExtractor
from .openai_embeddings import OpenAIEmbeddingsExtractor
from .persistence import FeatureExtractorPersistence


def create_feature_extractor(extractor_type: FeatureExtractorType, 
                           persistence: Optional[FeatureExtractorPersistence] = None,
                           **kwargs) -> FeatureExtractor:
    """Create and return the appropriate feature extractor.
    
    Args:
        extractor_type: The type of feature extractor to create
        persistence: Optional persistence handler for saving/loading extractors
        **kwargs: Additional arguments to pass to the extractor constructor
        
    Returns:
        FeatureExtractor: The created feature extractor instance
        
    Raises:
        ValueError: If extractor_type is not supported
        
    Examples:
        # Basic usage
        extractor = create_feature_extractor(FeatureExtractorType.TFIDF_VECTORIZER)
        
        # With custom parameters
        extractor = create_feature_extractor(
            FeatureExtractorType.TFIDF_VECTORIZER,
            max_features=5000,
            min_df=2
        )
        
        # With persistence
        from .persistence import PickleGCPExtractorPersistence
        persistence = PickleGCPExtractorPersistence("my-bucket")
        extractor = create_feature_extractor(
            FeatureExtractorType.COUNT_VECTORIZER,
            persistence=persistence,
            max_features=10000
        )
    """
    # Add persistence to kwargs if provided
    if persistence is not None:
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