"""Pipeline Package for Text Classification.

This package provides the core machine learning pipeline functionality
for text classification, including data preparation, feature creation,
model training, and evaluation.
"""

from .run_pipeline import (
    prepare_data,
    # create_features, 
    # train_model,
    evaluate_model,
    run_pipeline
)
from .pipeline import Pipeline
from .persistence import (
    PipelinePersistence,
    LocalPipelinePersistence,
    GCPPipelinePersistence
)

__all__ = [
    'prepare_data',
    # 'create_features',
    # 'train_model', 
    'evaluate_model',
    'run_pipeline',
    'Pipeline',
    'PipelinePersistence',
    'LocalPipelinePersistence',
    'GCPPipelinePersistence'
] 