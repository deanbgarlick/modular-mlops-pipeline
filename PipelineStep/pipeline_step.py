"""Pipeline step wrapper for feature extractors and supervised models."""

import pandas as pd
import numpy as np
import scipy.sparse
from typing import Tuple, Optional, cast, List, Union, Any

from FeatureExtractor.base import FeatureExtractor, FeatureMatrix
from SupervisedModel.base import SupervisedModel
from .persistence import PipelineStepPersistence


class PipelineStep:
    """
    Pipeline step wrapper that handles DataFrame input for both FeatureExtractor and SupervisedModel.
    
    This class acts as an adapter that allows both FeatureExtractors and SupervisedModels to work seamlessly 
    in DataFrame-based pipelines by automatically extracting the specified columns and converting outputs 
    to DataFrames. Supports flexible column selection including wildcard patterns.
    
    For FeatureExtractor: Extracts text from specified columns, applies feature extraction, adds feature columns.
    For SupervisedModel: Extracts features from specified columns, applies model prediction, adds prediction columns.
    """
    
    def __init__(self, component: Union[FeatureExtractor, SupervisedModel], 
                 included_features: List[str] = ["*"],
                 excluded_features: Optional[List[str]] = None,
                 target_column: str = "target",
                 prediction_column: str = "prediction",
                 probability_columns: bool = False,
                 persistence: Optional[PipelineStepPersistence] = None):
        """
        Initialize the pipeline step.
        
        Args:
            component: The FeatureExtractor or SupervisedModel instance to wrap
            included_features: List of column names/patterns to include as features. 
                             Use ["*"] for all columns. Supports wildcard patterns like "text*" 
            excluded_features: List of column names/patterns to exclude from features
            target_column: Column name containing target labels (used for SupervisedModel training)
            prediction_column: Column name for model predictions (SupervisedModel only)
            probability_columns: Whether to include probability columns (SupervisedModel only)
            persistence: PipelineStep persistence handler for saving/loading
        
        Examples:
            # FeatureExtractor usage
            PipelineStep(tfidf_extractor, included_features=["text"])
            
            # SupervisedModel usage  
            PipelineStep(logistic_model, included_features=["feature_*"], 
                        target_column="target", prediction_column="prediction")
        """
        self.component = component
        self.included_features = included_features
        self.excluded_features = excluded_features or []
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.probability_columns = probability_columns
        self.persistence = persistence
        
        # Detect component type
        self.is_feature_extractor = isinstance(component, FeatureExtractor)
        self.is_supervised_model = isinstance(component, SupervisedModel)
        
        if not (self.is_feature_extractor or self.is_supervised_model):
            raise ValueError(f"Component must be either FeatureExtractor or SupervisedModel, got {type(component)}")
    
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
    
    def _get_columns_to_use(self, df: pd.DataFrame) -> List[str]:
        """
        Determine which columns to use based on included_features and excluded_features.
        Supports wildcard patterns like "foo*" to match columns starting with "foo".
        
        Args:
            df: DataFrame to determine columns for
            
        Returns:
            List of column names to use
        """
        # Expand included patterns to actual column names
        columns_to_use = self._expand_column_patterns(df, self.included_features)
        
        # Remove any excluded columns (with wildcard support)
        columns_to_exclude = self._expand_column_patterns(df, self.excluded_features)
        columns_to_use = [col for col in columns_to_use if col not in columns_to_exclude]
        
        # Check that we have columns to work with
        if not columns_to_use:
            raise ValueError("No columns left after applying included_features and excluded_features filters")
        
        # Validate that included patterns matched something
        original_included_count = len(self._expand_column_patterns(df, self.included_features))
        if original_included_count == 0 and self.included_features != ["*"]:
            raise ValueError(f"No columns found matching included patterns: {self.included_features}")
        
        return columns_to_use
    
    def _expand_column_patterns(self, df: pd.DataFrame, patterns: List[str]) -> List[str]:
        """
        Expand a list of column patterns (including wildcards) to actual column names.
        
        Args:
            df: DataFrame to match patterns against
            patterns: List of column names/patterns (e.g., ["text*", "id", "meta*"])
            
        Returns:
            List of actual column names that match the patterns
        """
        expanded_columns = []
        for pattern in patterns:
            if pattern.endswith("*") and pattern != "*":
                # Wildcard pattern - match columns starting with prefix
                prefix = pattern[:-1]  # Remove the trailing *
                matched_columns = [col for col in df.columns if col.startswith(prefix)]
                expanded_columns.extend(matched_columns)
            elif pattern == "*":
                # Special case: include all columns
                expanded_columns.extend(df.columns.tolist())
            else:
                # Exact column name
                if pattern in df.columns:
                    expanded_columns.append(pattern)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(expanded_columns))
    
    def _extract_text_data(self, df: pd.DataFrame) -> List[str]:
        """
        Extract and combine text data from DataFrame columns (for FeatureExtractor).
        
        Args:
            df: DataFrame containing text columns
            
        Returns:
            List of combined text strings from the specified columns
        """
        columns_to_use = self._get_columns_to_use(df)
        
        # If single column, return as list of strings
        if len(columns_to_use) == 1:
            return df[columns_to_use[0]].astype(str).tolist()
        
        # If multiple columns, concatenate them with space separator
        combined_series = df[columns_to_use].astype(str).apply(
            lambda row: ' '.join(row.values), axis=1
        )
        return combined_series.tolist()
    
    def _extract_feature_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature data from DataFrame columns (for SupervisedModel).
        
        Args:
            df: DataFrame containing feature columns
            
        Returns:
            Feature matrix for model input
        """
        columns_to_use = self._get_columns_to_use(df)
        return df[columns_to_use].values  # type: ignore[return-value]
    
    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                     y_train: Optional[pd.Series] = None, y_test: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit on training data and transform both train and test sets.
        
        Args:
            X_train: Training DataFrame
            X_test: Test DataFrame
            y_train: Training target labels (optional, for SupervisedModel)
            y_test: Test target labels (optional, for SupervisedModel)
            
        Returns:
            Tuple of (X_train_transformed, X_test_transformed) as DataFrames
        """
        if self.is_feature_extractor:
            return self._fit_transform_feature_extractor(X_train, X_test)
        elif self.is_supervised_model:
            return self._fit_transform_supervised_model(X_train, X_test, y_train, y_test)
        else:
            raise ValueError("Unknown component type")
    
    def _fit_transform_feature_extractor(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle fit_transform for FeatureExtractor."""
        # Extract text data from included columns
        train_text = self._extract_text_data(X_train)
        test_text = self._extract_text_data(X_test)
        
        # Convert to Series for compatibility with FeatureExtractor
        train_series = pd.Series(train_text)
        test_series = pd.Series(test_text)
        
        # Get features from extractor
        train_features, test_features = self.component.fit_transform(train_series, test_series)  # type: ignore[attr-defined]
        
        # Convert to DataFrames with meaningful prefix
        columns_to_use = self._get_columns_to_use(X_train)
        feature_prefix = "_".join(columns_to_use) if len(columns_to_use) <= 3 else "features"
        train_features_df = self._convert_to_dataframe(train_features, feature_prefix)
        test_features_df = self._convert_to_dataframe(test_features, feature_prefix) 
        
        # Combine original DataFrames with feature DataFrames
        train_combined = pd.concat([X_train, train_features_df], axis=1)
        test_combined = pd.concat([X_test, test_features_df], axis=1)
        
        return train_combined, test_combined
    
    def _fit_transform_supervised_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                        y_train: Optional[pd.Series] = None, y_test: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle fit_transform for SupervisedModel."""
        # Extract feature data and target
        train_features = self._extract_feature_data(X_train)
        test_features = self._extract_feature_data(X_test)
        
        # Use provided y_train or extract from DataFrame
        if y_train is not None:
            train_target = y_train.values
        else:
            train_target = X_train[self.target_column].values  # type: ignore[index]
        
        # Fit the model
        self.component.fit(train_features, train_target)  # type: ignore[attr-defined,arg-type]
        
        # Make predictions
        train_predictions = self.component.predict(train_features)  # type: ignore[attr-defined]
        test_predictions = self.component.predict(test_features)  # type: ignore[attr-defined]
        
        # Create prediction DataFrames
        train_pred_df = pd.DataFrame({self.prediction_column: train_predictions})
        test_pred_df = pd.DataFrame({self.prediction_column: test_predictions})
        
        # Add probability columns if requested
        if self.probability_columns:
            train_proba = self.component.predict_proba(train_features)  # type: ignore[attr-defined]
            test_proba = self.component.predict_proba(test_features)  # type: ignore[attr-defined]
            
            # Add probability columns (assuming class indices match column order)
            for i in range(train_proba.shape[1]):
                prob_col = f"{self.prediction_column}_proba_{i}"
                train_pred_df[prob_col] = train_proba[:, i]
                test_pred_df[prob_col] = test_proba[:, i]
        
        # Combine original DataFrames with prediction DataFrames
        train_combined = pd.concat([X_train, train_pred_df], axis=1)
        test_combined = pd.concat([X_test, test_pred_df], axis=1)
        
        return train_combined, test_combined
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new DataFrame data using fitted component.
        
        Args:
            X: DataFrame containing data to transform
            
        Returns:
            DataFrame with original columns plus transformed/prediction columns
        """
        if self.is_feature_extractor:
            return self._transform_feature_extractor(X)
        elif self.is_supervised_model:
            return self._transform_supervised_model(X)
        else:
            raise ValueError("Unknown component type")
    
    def _transform_feature_extractor(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle transform for FeatureExtractor."""
        # Extract text data from included columns
        text_data = self._extract_text_data(X)
        features = self.component.transform(text_data)
        
        # Convert to DataFrame with meaningful prefix
        columns_to_use = self._get_columns_to_use(X)
        feature_prefix = "_".join(columns_to_use) if len(columns_to_use) <= 3 else "features"
        features_df = self._convert_to_dataframe(features, feature_prefix)
        
        # Combine original DataFrame with feature DataFrame
        return pd.concat([X, features_df], axis=1)
    
    def _transform_supervised_model(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle transform for SupervisedModel."""
        # Extract feature data
        features = self._extract_feature_data(X)
        
        # Make predictions
        predictions = self.component.predict(features)
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({self.prediction_column: predictions})
        
        # Add probability columns if requested
        if self.probability_columns:
            proba = self.component.predict_proba(features)
            
            # Add probability columns
            for i in range(proba.shape[1]):
                prob_col = f"{self.prediction_column}_proba_{i}"
                pred_df[prob_col] = proba[:, i]
        
        # Combine original DataFrame with prediction DataFrame
        return pd.concat([X, pred_df], axis=1)
    
    def get_feature_info(self) -> dict:
        """Return information about the features created by the wrapped extractor."""
        feature_info = self.component.get_feature_info()
        # Add pipeline step information
        if isinstance(feature_info, dict) and "error" not in feature_info:
            feature_info["included_features"] = self.included_features
            feature_info["excluded_features"] = self.excluded_features
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
        self.component = loaded_pipeline_step.component
        self.included_features = loaded_pipeline_step.included_features
        self.excluded_features = loaded_pipeline_step.excluded_features
        self.target_column = loaded_pipeline_step.target_column
        self.prediction_column = loaded_pipeline_step.prediction_column
        self.probability_columns = loaded_pipeline_step.probability_columns
    
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