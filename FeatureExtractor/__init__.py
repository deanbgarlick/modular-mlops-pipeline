"""Feature Extractor Package for Text Classification.

This package provides a modular approach to feature extraction from text data,
supporting multiple extraction methods through a clean interface with comprehensive
persistence capabilities for saving and loading fitted extractors.
"""

from .base import FeatureExtractor, FeatureExtractorType
from .count_vectorizer import CountVectorizerExtractor
from .tfidf_vectorizer import TfidfVectorizerExtractor
from .huggingface import HuggingFaceExtractor
from .word2vec import Word2VecExtractor
from .openai_embeddings import OpenAIEmbeddingsExtractor
from .persistence import (
    FeatureExtractorPersistence,
    PickleGCPExtractorPersistence,
    PickleAWSExtractorPersistence,
    PickleLocalExtractorPersistence,
    HuggingFaceExtractorPersistence,
    # Backward compatibility aliases
    GCPExtractorPersistence,
    AWSExtractorPersistence,
    LocalExtractorPersistence
)
from .factory import create_feature_extractor

__all__ = [
    'FeatureExtractor',
    'FeatureExtractorType',
    'CountVectorizerExtractor',
    'TfidfVectorizerExtractor',
    'HuggingFaceExtractor',
    'Word2VecExtractor',
    'OpenAIEmbeddingsExtractor',
    'FeatureExtractorPersistence',
    'PickleGCPExtractorPersistence',
    'PickleAWSExtractorPersistence', 
    'PickleLocalExtractorPersistence',
    'HuggingFaceExtractorPersistence',
    'GCPExtractorPersistence',
    'AWSExtractorPersistence',
    'LocalExtractorPersistence',
    'create_feature_extractor'
] 