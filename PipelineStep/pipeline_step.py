"""Pipeline step wrapper for feature extractors."""

import pandas as pd
import numpy as np
import scipy.sparse
from typing import Tuple, Optional, cast, List

from FeatureExtractor.base import FeatureExtractor, FeatureMatrix
from .persistence import PipelineStepPersistence


class PipelineStep:
    """
    Pipeline step wrapper for FeatureExtractor that handles DataFrame input.
    
    This class acts as an adapter that allows FeatureExtractors to work seamlessly 
    in DataFrame-based pipelines by automatically extracting the specified feature columns
    and converting outputs to DataFrames.
    """
    
    def __init__(self, feature_extractor: FeatureExtractor, included_features: List[str] = ["*"],
                 persistence: Optional[PipelineStepPersistence] = None):
        """
        Initialize the pipeline step.
        
        Args:
            feature_extractor: The FeatureExtractor instance to wrap
            included_features: List of column names to include as features from input DataFrames
            persistence: PipelineStep persistence handler for saving/loading
        """
        self.feature_extractor = feature_extractor
        self.included_features = included_features
        self.persistence = persistence
    
    def _convert_to_dataframe(self, features: FeatureMatrix, prefix: str = "feature") -> pd.DataFrame:
        """
        Convert feature matrix to DataFrame.
        
        Args:
            features: Feature matrix from extractor
            prefix: Prefix for generated column names
            
        Returns:
            DataFrame with features
            
        Raises:
            ValueError: If features are sparse (not supported)
        """
        # Check if sparse matrix and reject
        if scipy.sparse.issparse(features):
            raise ValueError(
                "PipelineStep does not support sparse feature matrices. "
                "Consider using a different feature extractor or preprocessing step to densify the features."
            )
        
        # If already a DataFrame, return as-is
        if isinstance(features, pd.DataFrame):
            return features
        
        # Convert numpy array to DataFrame with generated column names
        if isinstance(features, np.ndarray):
            n_features = features.shape[1] if len(features.shape) > 1 else 1
            column_names = pd.Index([f"{prefix}_{i}" for i in range(n_features)])
            
            if len(features.shape) == 1:
                # Handle 1D arrays
                return pd.DataFrame({f"{prefix}_0": features})
            else:
                # Handle 2D arrays
                return pd.DataFrame(data=features, columns=column_names)
        
        # Fallback for other types
        raise ValueError(f"Unsupported feature type: {type(features)}")
    
    def _extract_feature_data(self, df: pd.DataFrame) -> List[str]:
        """
        Extract and combine feature data from DataFrame columns.
        
        Args:
            df: DataFrame containing feature columns
            
        Returns:
            List of combined text strings from the specified columns
        """
        missing_columns = [col for col in self.included_features if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in DataFrame")
        
        # If single column, return as list of strings
        if len(self.included_features) == 1:
            if self.included_features[0] == "*":
                return df.astype(str).tolist()
            else:
                return df[self.included_features[0]].astype(str).tolist()
        
        # If multiple columns, concatenate them with space separator
        combined_series = df[self.included_features].astype(str).apply(
            lambda row: ' '.join(row.values), axis=1
        )
        return combined_series.tolist()

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit on training data and transform both train and test sets.
        
        Args:
            X_train: Training DataFrame containing feature data
            X_test: Test DataFrame containing feature data
            
        Returns:
            Tuple of (X_train_with_features, X_test_with_features) as DataFrames containing 
            original columns plus generated feature columns
        """
        # Extract feature data from included columns
        train_text = self._extract_feature_data(X_train)
        test_text = self._extract_feature_data(X_test)
        
        # Convert to Series for compatibility with FeatureExtractor
        train_series = pd.Series(train_text)
        test_series = pd.Series(test_text)
        
        # Get features from extractor
        train_features, test_features = self.feature_extractor.fit_transform(train_series, test_series)
        
        # Convert to DataFrames with meaningful prefix
        feature_prefix = "_".join(self.included_features) if len(self.included_features) <= 3 else "features"
        train_features_df = self._convert_to_dataframe(train_features, feature_prefix)
        test_features_df = self._convert_to_dataframe(test_features, feature_prefix) 
        
        # Combine original DataFrames with feature DataFrames
        train_combined = pd.concat([X_train, train_features_df], axis=1)
        test_combined = pd.concat([X_test, test_features_df], axis=1)
        
        return train_combined, test_combined
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new DataFrame data using fitted extractor.
        
        Args:
            X: DataFrame containing feature data to transform
            
        Returns:
            DataFrame with original columns plus transformed feature columns
        """
        # Extract feature data from included columns
        text_data = self._extract_feature_data(X)
        features = self.feature_extractor.transform(text_data)
        
        # Convert to DataFrame with meaningful prefix
        feature_prefix = "_".join(self.included_features) if len(self.included_features) <= 3 else "features"
        features_df = self._convert_to_dataframe(features, feature_prefix)
        
        # Combine original DataFrame with feature DataFrame
        return pd.concat([X, features_df], axis=1)
    
    def get_feature_info(self) -> dict:
        """Return information about the features created by the wrapped extractor."""
        feature_info = self.feature_extractor.get_feature_info()
        # Add pipeline step information
        if isinstance(feature_info, dict) and "error" not in feature_info:
            feature_info["included_features"] = self.included_features
            feature_info["pipeline_step"] = True
        return feature_info
    
    def save(self, path: str) -> None:
        """
        Save the PipelineStep using the configured persistence handler.
        
        Args:
            path: Path where to save the pipeline step
        """
        if self.persistence is None:
            raise ValueError("No persistence handler configured for PipelineStep")
        
        self.persistence.save(self, path)
    
    def load(self, path: str) -> None:
        """
        Load the PipelineStep using the configured persistence handler.
        
        Args:
            path: Path to load the pipeline step from
        """
        if self.persistence is None:
            raise ValueError("No persistence handler configured for PipelineStep")
        
        loaded_pipeline_step = self.persistence.load(path)
        
        # Update current instance with loaded data
        self.feature_extractor = loaded_pipeline_step.feature_extractor
        self.included_features = loaded_pipeline_step.included_features
    
    @classmethod
    def load_from_path(cls, path: str, persistence: PipelineStepPersistence):
        """
        Class method to create a new PipelineStep instance and load from path.
        
        Args:
            path: Path to load the pipeline step from
            persistence: PipelineStep persistence handler
            
        Returns:
            PipelineStep: New instance with loaded pipeline step
        """
        if persistence is None:
            raise ValueError("Persistence handler is required for loading PipelineStep")
        
        return persistence.load(path) 