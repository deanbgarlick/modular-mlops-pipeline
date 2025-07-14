"""Feature Extractor Package for Text Classification.

This package provides a modular approach to feature extraction from text data,
supporting multiple extraction methods through a clean interface.
"""

from .base import FeatureExtractor, FeatureExtractorType
from .count_vectorizer import CountVectorizerExtractor
from .huggingface import HuggingFaceExtractor
from .factory import create_feature_extractor

__all__ = [
    'FeatureExtractor',
    'FeatureExtractorType', 
    'CountVectorizerExtractor',
    'HuggingFaceExtractor',
    'create_feature_extractor'
] 