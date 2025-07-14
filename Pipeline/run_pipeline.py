"""Core machine learning pipeline functions for text classification."""

import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from typing import Optional, Dict, Any

from FeatureExtractor import (
    FeatureExtractor, FeatureExtractorType, create_feature_extractor,
    PickleGCPExtractorPersistence, PickleLocalExtractorPersistence
)
from SupervisedModel import (
    SupervisedModel, SupervisedModelType, create_model,
    PickleGCPBucketPersistence, TorchGCPBucketPersistence, PickleLocalFilePersistence
)
from DataLoader import DataSourceType, create_data_loader
from PipelineStep.pipeline_step import PipelineStep
from PipelineStep.persistence import GCPPipelineStepPersistence, LocalPipelineStepPersistence
from Pipeline.pipeline import Pipeline, PipelinePersistence


def prepare_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets and return as DataFrames."""
    print(f"\nSplitting data into train/test sets (test_size={test_size})...")
    
    # Create train/test split while maintaining DataFrame structure
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['target']
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df


def evaluate_model(y_true, y_pred, target_names):
    """Evaluate predictions."""
    print(f"\nEvaluating model on test set...")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Print detailed results
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return accuracy, f1_macro, f1_weighted


def generate_artifact_name(feature_extractor_type: FeatureExtractorType, 
                          model_type: SupervisedModelType,
                          extractor_kwargs: Dict[str, Any],
                          model_kwargs: Dict[str, Any],
                          prefix: str = "",
                          stable_naming: bool = True) -> str:
    """Generate a name for saved artifacts based on configuration."""
    # Create a hash-like string from configuration
    config_str = f"{feature_extractor_type.value}_{model_type.value}"
    
    # Add key parameters to the name
    if extractor_kwargs:
        for key, value in sorted(extractor_kwargs.items()):
            if key != 'persistence':  # Skip persistence objects
                config_str += f"_{key}{value}"
    
    if model_kwargs:
        for key, value in sorted(model_kwargs.items()):
            if key != 'persistence':  # Skip persistence objects
                config_str += f"_{key}{value}"
    
    # Add timestamp only if not using stable naming
    if not stable_naming:
        timestamp = int(time.time())
        config_str += f"_{timestamp}"
    
    return f"{prefix}{config_str}" if prefix else config_str


def setup_persistence(use_gcp_persistence: bool = True,
                     gcp_bucket: Optional[str] = None):
    """Setup persistence handlers for pipeline components."""
    
    # Get bucket name from environment or parameter
    if gcp_bucket is None:
        gcp_bucket = os.getenv("GCP_ML_BUCKET", "default-ml-artifacts")
    
    if use_gcp_persistence:
        print(f"Setting up GCP persistence with bucket: {gcp_bucket}")
        
        # Feature extractor persistence
        extractor_persistence = PickleGCPExtractorPersistence(
            bucket_name=gcp_bucket,
            prefix="feature_extractors/"
        )
        
        # Model persistence
        model_persistence = PickleGCPBucketPersistence(bucket_name=gcp_bucket)
        
        # Pipeline step persistence
        pipeline_step_persistence = GCPPipelineStepPersistence(
            bucket_name=gcp_bucket,
            prefix="pipeline_steps/"
        )
        
        # Pipeline persistence
        pipeline_persistence = PipelinePersistence(base_path="pipelines")  # Still local for metadata
        
    else:
        print("Setting up local persistence")
        
        # Local persistence
        extractor_persistence = PickleLocalExtractorPersistence(base_path="saved_extractors")
        model_persistence = PickleLocalFilePersistence(base_path="saved_models")
        pipeline_step_persistence = LocalPipelineStepPersistence(base_path="saved_pipeline_steps")
        pipeline_persistence = PipelinePersistence(base_path="pipelines")
    
    return extractor_persistence, model_persistence, pipeline_step_persistence, pipeline_persistence


def create_pipeline_components(feature_extractor_type: FeatureExtractorType,
                              model_type: SupervisedModelType,
                              extractor_kwargs: Dict[str, Any],
                              model_kwargs: Dict[str, Any],
                              extractor_persistence,
                              model_persistence,
                              pipeline_step_persistence) -> Pipeline:
    """Create a pipeline with feature extraction and model components."""
    
    # Create feature extractor
    feature_extractor = create_feature_extractor(
        feature_extractor_type, 
        persistence=extractor_persistence,
        **extractor_kwargs
    )
    
    # Create model
    model = create_model(
        model_type,
        persistence=model_persistence,
        **model_kwargs
    )
    
    # Create pipeline step wrapper for feature extractor
    feature_pipeline_step = PipelineStep(
        component=feature_extractor,
        included_features=["text"],  # Use the 'text' column from our dataframes
        persistence=pipeline_step_persistence
    )
    
    # Create pipeline step wrapper for model  
    model_pipeline_step = PipelineStep(
        component=model,
        included_features=["*"],  # Use all feature columns
        excluded_features=["text", "target"],  # Exclude original text and target
        target_column="target",
        prediction_column="prediction",
        persistence=pipeline_step_persistence
    )
    
    # Create pipeline
    pipeline = Pipeline([feature_pipeline_step, model_pipeline_step])  # type: ignore[arg-type]
    
    return pipeline


def run_pipeline(data_source_type: DataSourceType = DataSourceType.NEWSGROUPS,
                 feature_extractor_type: FeatureExtractorType = FeatureExtractorType.COUNT_VECTORIZER,
                 model_type: SupervisedModelType = SupervisedModelType.LOGISTIC_REGRESSION,
                 use_class_weights: bool = False,
                 loader_kwargs: Optional[Dict[str, Any]] = None,
                 extractor_kwargs: Optional[Dict[str, Any]] = None, 
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 # Persistence options
                 use_gcp_persistence: bool = True,
                 gcp_bucket: Optional[str] = None,
                 save_artifacts: bool = True,
                 force_retrain: bool = False,
                 artifact_prefix: str = ""):
    """Run the complete machine learning pipeline using the Pipeline class.
    
    Args:
        data_source_type: Type of data source to use
        feature_extractor_type: Type of feature extractor to use
        model_type: Type of model to use
        use_class_weights: Whether to use class weights for imbalanced data
        loader_kwargs: Arguments for data loader
        extractor_kwargs: Arguments for feature extractor
        model_kwargs: Arguments for model
        use_gcp_persistence: Whether to use GCP storage (True) or local storage (False)
        gcp_bucket: GCP bucket name (uses env var GCP_ML_BUCKET if None)
        save_artifacts: Whether to save the entire pipeline
        force_retrain: Whether to force retraining even if saved artifacts exist
        artifact_prefix: Prefix for artifact names (useful for experiments)
    
    Returns:
        Dict containing results and pipeline information
    """
    if loader_kwargs is None:
        loader_kwargs = {}
    if extractor_kwargs is None:
        extractor_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
        
    print(f"Using data source: {data_source_type.value}")
    print(f"Using feature extractor: {feature_extractor_type.value}")
    print(f"Using model: {model_type.value}")
    print(f"GCP Persistence: {use_gcp_persistence}")
    print(f"Save artifacts: {save_artifacts}")
    print(f"Force retrain: {force_retrain}")
    
    # Setup persistence
    extractor_persistence, model_persistence, pipeline_step_persistence, pipeline_persistence = setup_persistence(
        use_gcp_persistence=use_gcp_persistence,
        gcp_bucket=gcp_bucket
    )
    
    # Generate artifact name for this configuration
    artifact_name = generate_artifact_name(
        feature_extractor_type, model_type, 
        extractor_kwargs, model_kwargs, 
        prefix=artifact_prefix,
        stable_naming=not force_retrain
    )
    print(f"Pipeline artifact name: {artifact_name}")
    
    # Load data
    data_loader = create_data_loader(data_source_type, **loader_kwargs)
    df, target_names = data_loader.load_data()
    
    # Print data loader info
    data_info = data_loader.get_data_info()
    print("\nData loader info:")
    for key, value in data_info.items():
        if key != 'class_distribution':  # Skip detailed distribution for cleaner output
            print(f"  {key}: {value}")
    
    # Prepare train/test splits
    train_df, test_df = prepare_data(df)
    
    # Try to load existing pipeline or create new one
    pipeline_loaded = False
    if not force_retrain and save_artifacts:
        try:
            print(f"\nAttempting to load existing pipeline: {artifact_name}")
            pipeline = Pipeline.load_from_path(artifact_name, persistence=pipeline_persistence)
            pipeline_loaded = True
            print("✓ Successfully loaded existing pipeline")
        except Exception as e:
            print(f"Failed to load existing pipeline: {e}")
            print("Creating new pipeline...")
    
    if not pipeline_loaded:
        # Create new pipeline
        pipeline = create_pipeline_components(
            feature_extractor_type, model_type,
            extractor_kwargs, model_kwargs,
            extractor_persistence, model_persistence, pipeline_step_persistence
        )
    
    # Train/fit the pipeline if needed
    if not pipeline_loaded or force_retrain:
        print(f"\nTraining pipeline...")
        
        # Add class weights to model if requested
        if use_class_weights:
             from sklearn.utils.class_weight import compute_class_weight
             classes = np.unique(train_df['target'])  # type: ignore[index]
             weights = compute_class_weight('balanced', classes=classes, y=train_df['target'])  # type: ignore[index]
             class_weights = dict(zip(classes, weights))
             print(f"Calculated class weights: {class_weights}")
             # Note: Pipeline doesn't currently support class weights in fit_transform
        
        # Fit and transform training data, transform test data with explicit targets
        train_transformed, test_transformed = pipeline.fit_transform(
            train_df, test_df, 
            y_train=train_df['target'], 
            y_test=test_df['target']
        )  # type: ignore[arg-type,index]
        print("Pipeline training completed!")
    else:
        print(f"\nUsing pre-trained pipeline...")
        # Just transform the data with the loaded pipeline
        train_transformed = pipeline.transform(train_df)  # type: ignore[arg-type]
        test_transformed = pipeline.transform(test_df)  # type: ignore[arg-type]
    
    # Make predictions on test set
    # The final transformed data should be predictions from the model
    # Convert to numpy array for consistent handling
    if hasattr(test_transformed, 'toarray'):
        # Handle sparse matrix
        y_pred = test_transformed.toarray().flatten()  # type: ignore[attr-defined]
    elif hasattr(test_transformed, 'flatten'):
        y_pred = test_transformed.flatten()  # type: ignore[attr-defined]
    else:
        y_pred = np.array(test_transformed).flatten()
    
    # Convert predictions to class indices if they're probabilities
    if y_pred.ndim > 1 or (y_pred.dtype == float and len(np.unique(y_pred)) > len(target_names)):
        # If we have probabilities, get the class with highest probability
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            # Single column of probabilities, threshold at 0.5 for binary classification
            y_pred = (y_pred > 0.5).astype(int)
    
    # Evaluate model
    accuracy, f1_macro, f1_weighted = evaluate_model(
        test_df['target'].values, y_pred, target_names
    )
    
    # Save pipeline if requested
    saved_paths = {}
    if save_artifacts and not pipeline_loaded:
        try:
            print(f"\nSaving pipeline...")
            pipeline.save(artifact_name)
            
            saved_paths = {
                "pipeline_path": artifact_name,
                "bucket": gcp_bucket if use_gcp_persistence else "local",
                "artifact_name": artifact_name
            }
            print(f"✓ Pipeline saved successfully")
            
        except Exception as e:
            print(f"⚠ Failed to save pipeline: {e}")
    
    # Example prediction on new text
    print(f"\nExample prediction:")
    sample_df = pd.DataFrame({
        'text': ["God is great and religion brings peace to the world"],
        'target': [0]  # Dummy target, not used for prediction
    })
    
    try:
        sample_prediction = pipeline.transform(sample_df)
        pred_class = sample_prediction.flatten()[0] if hasattr(sample_prediction, 'flatten') else sample_prediction[0]
        
        # Handle probability vs class prediction
        if isinstance(pred_class, (float, np.floating)) and 0 <= pred_class <= 1:
            pred_class = int(pred_class > 0.5) if len(target_names) == 2 else int(np.round(pred_class))
        
        print(f"Sample text: {sample_df['text'].iloc[0]}")
        print(f"Predicted class: {target_names[int(pred_class)]}")
    except Exception as e:
        print(f"Error making sample prediction: {e}")
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "predictions": y_pred,
        "target_names": target_names,
        "pipeline": pipeline,
        "saved_paths": saved_paths,
        "artifact_name": artifact_name,
        "training_time": time.time()
    } 