"""PipelineStep Package for Feature Extractor Adapters.

This package provides adapter classes that allow FeatureExtractors to work 
seamlessly with DataFrame-based data processing pipelines, along with comprehensive
persistence capabilities for saving and loading pipeline steps.
"""

from .pipeline_step import PipelineStep
from .persistence import (
    PipelineStepPersistence,
    GCPPipelineStepPersistence,
    AWSPipelineStepPersistence,
    LocalPipelineStepPersistence
)

__all__ = [
    'PipelineStep',
    'PipelineStepPersistence',
    'GCPPipelineStepPersistence',
    'AWSPipelineStepPersistence',
    'LocalPipelineStepPersistence',
] 