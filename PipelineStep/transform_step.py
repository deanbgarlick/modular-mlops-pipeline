"""Protocol for transformation steps in machine learning pipelines."""

import pandas as pd
import numpy as np
import scipy.sparse
from typing import Protocol, Union, Tuple, List, Any, Optional
from typing_extensions import runtime_checkable


# Unified type aliases for transformation steps
FeatureMatrix = Union[np.ndarray, scipy.sparse.csr_matrix, pd.DataFrame]
InputData = Union[pd.Series, List[str], FeatureMatrix]
OutputData = Union[FeatureMatrix, Tuple[FeatureMatrix, FeatureMatrix]]


@runtime_checkable
class TransformationStep(Protocol):
    """
    Protocol for transformation steps that can handle both feature extraction and model operations.
    
    This protocol defines a common interface that can accommodate both:
    - FeatureExtractor objects (with fit_transform returning tuple of matrices)
    - SupervisedModel objects (when used for feature transformation via predictions)
    """
    
    def fit_transform(self, X_train: InputData, X_test: Optional[InputData] = None) -> OutputData:
        """
        Fit on training data and transform both train and test sets.
        
        Args:
            X_train: Training data (pd.Series for text, FeatureMatrix for numerical features)
            X_test: Test data (optional, same type as X_train)
            
        Returns:
            OutputData: Transformed features. Can be:
                - Single FeatureMatrix for X_train only
                - Tuple[FeatureMatrix, FeatureMatrix] for (X_train_transformed, X_test_transformed) when X_test provided
        """
        ...
    
    def transform(self, X: InputData) -> FeatureMatrix:
        """
        Transform new data using fitted transformer.
        
        Args:
            X: Input data to transform (List[str] for text, pd.Series, or FeatureMatrix for numerical)
            
        Returns:
            FeatureMatrix: Transformed features as matrix/array/dataframe
        """
        ...
