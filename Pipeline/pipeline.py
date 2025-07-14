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


def prepare_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    print(f"\nSplitting data into train/test sets (test_size={test_size})...")
    X = df['text']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def create_features(X_train, X_test, feature_extractor: FeatureExtractor):
    """Create features using the specified feature extractor."""
    print(f"\nExtracting features...")
    
    # Fit on train data and transform both train and test
    X_train_features, X_test_features = feature_extractor.fit_transform(X_train, X_test)
    
    # Print feature extractor info
    feature_info = feature_extractor.get_feature_info()
    for key, value in feature_info.items():
        print(f"{key}: {value}")
    
    return X_train_features, X_test_features


def train_model(X_train, y_train, model: SupervisedModel, use_class_weights: bool = False):
    """Train the model."""
    print(f"\nTraining model...")
    
    # Calculate class weights if requested
    class_weights = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print(f"Calculated class weights: {class_weights}")
    
    # Train the model
    model.fit(X_train, y_train, class_weights=class_weights)
    print("Model training completed!")
    
    # Print model info
    model_info = model.get_model_info()
    print("Model info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate the trained model."""
    print(f"\nEvaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Print detailed results
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy, f1_macro, f1_weighted, y_pred, y_pred_proba


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
                     gcp_bucket: Optional[str] = None,
                     extractor_prefix: str = "feature_extractors/",
                     model_prefix: str = "models/"):
    """Setup persistence handlers for feature extractors and models."""
    
    # Get bucket name from environment or parameter
    if gcp_bucket is None:
        gcp_bucket = os.getenv("GCP_ML_BUCKET", "default-ml-artifacts")
    
    if use_gcp_persistence:
        print(f"Setting up GCP persistence with bucket: {gcp_bucket}")
        
        # Feature extractor persistence
        extractor_persistence = PickleGCPExtractorPersistence(
            bucket_name=gcp_bucket,
            prefix=extractor_prefix
        )
        
        # Model persistence (will auto-select Torch vs Pickle based on model type)
        model_persistence = PickleGCPBucketPersistence(
            bucket_name=gcp_bucket
        )
        
    else:
        print("Setting up local persistence")
        
        # Local persistence
        extractor_persistence = PickleLocalExtractorPersistence(
            base_path="saved_extractors"
        )
        model_persistence = PickleLocalFilePersistence(
            base_path="saved_models"
        )
    
    return extractor_persistence, model_persistence


def load_or_create_feature_extractor(feature_extractor_type: FeatureExtractorType,
                                    extractor_kwargs: Dict[str, Any],
                                    extractor_persistence,
                                    artifact_name: str,
                                    force_retrain: bool = False) -> FeatureExtractor:
    """Load existing feature extractor or create new one."""
    
    extractor_path = f"{artifact_name}_extractor"
    
    if not force_retrain:
        try:
            print(f"Attempting to load existing feature extractor: {extractor_path}")
            feature_extractor = create_feature_extractor(
                feature_extractor_type, 
                persistence=extractor_persistence
            )
            feature_extractor.load(extractor_path)
            print("✓ Successfully loaded existing feature extractor")
            return feature_extractor
        except Exception as e:
            print(f"Failed to load existing extractor: {e}")
            print("Creating new feature extractor...")
    
    # Create new extractor
    feature_extractor = create_feature_extractor(
        feature_extractor_type, 
        persistence=extractor_persistence,
        **extractor_kwargs
    )
    
    return feature_extractor


def load_or_create_model(model_type: SupervisedModelType,
                        model_kwargs: Dict[str, Any],
                        model_persistence,
                        artifact_name: str,
                        force_retrain: bool = False) -> SupervisedModel:
    """Load existing model or create new one."""
    
    model_path = f"{artifact_name}_model"
    
    if not force_retrain:
        try:
            print(f"Attempting to load existing model: {model_path}")
            model = create_model(
                model_type,
                persistence=model_persistence
            )
            model.load(model_path)
            print("✓ Successfully loaded existing model")
            return model
        except Exception as e:
            print(f"Failed to load existing model: {e}")
            print("Creating new model...")
    
    # Create new model
    model = create_model(
        model_type,
        persistence=model_persistence,
        **model_kwargs
    )
    
    return model


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
    """Run the complete machine learning pipeline with persistence support.
    
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
        save_artifacts: Whether to save fitted extractors and trained models
        force_retrain: Whether to force retraining even if saved artifacts exist
        artifact_prefix: Prefix for artifact names (useful for experiments)
    
    Returns:
        Dict containing results and artifact information
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
    extractor_persistence, model_persistence = setup_persistence(
        use_gcp_persistence=use_gcp_persistence,
        gcp_bucket=gcp_bucket
    )
    
    # Generate artifact name for this configuration
    # Use stable naming when not forcing retrain to enable artifact reuse
    artifact_name = generate_artifact_name(
        feature_extractor_type, model_type, 
        extractor_kwargs, model_kwargs, 
        prefix=artifact_prefix,
        stable_naming=not force_retrain
    )
    print(f"Artifact name: {artifact_name}")
    if not force_retrain:
        print("Using stable naming for artifact reuse")
    
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
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Load or create feature extractor
    feature_extractor = load_or_create_feature_extractor(
        feature_extractor_type, extractor_kwargs, extractor_persistence,
        artifact_name, force_retrain
    )
    
    # Create features (fit if new extractor, just transform if loaded)
    if hasattr(feature_extractor, 'is_fitted') and feature_extractor.is_fitted:
        print("\nUsing pre-fitted feature extractor...")
        X_train_transformed = feature_extractor.transform(list(X_train))
        X_test_transformed = feature_extractor.transform(list(X_test))
        
        # Print feature extractor info
        feature_info = feature_extractor.get_feature_info()
        for key, value in feature_info.items():
            print(f"{key}: {value}")
    else:
        X_train_transformed, X_test_transformed = create_features(
            X_train, X_test, feature_extractor
        )
    
    # Load or create model
    model = load_or_create_model(
        model_type, model_kwargs, model_persistence,
        artifact_name, force_retrain
    )
    
    # Train model if not already trained
    if not (hasattr(model, 'is_fitted') and getattr(model, 'is_fitted', False)) or force_retrain:
        trained_model = train_model(X_train_transformed, y_train, model, use_class_weights)
    else:
        print("\nUsing pre-trained model...")
        trained_model = model
        
        # Print model info
        model_info = model.get_model_info()
        print("Model info:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
    
    # Evaluate model
    accuracy, f1_macro, f1_weighted, y_pred, y_pred_proba = evaluate_model(
        trained_model, X_test_transformed, y_test, target_names
    )
    
    # Save artifacts if requested
    saved_paths = {}
    if save_artifacts:
        try:
            extractor_path = f"{artifact_name}_extractor"
            model_path = f"{artifact_name}_model"
            
            print(f"\nSaving artifacts...")
            feature_extractor.save(extractor_path)
            trained_model.save(model_path)
            
            saved_paths = {
                "extractor_path": extractor_path,
                "model_path": model_path,
                "bucket": gcp_bucket if use_gcp_persistence else "local",
                "artifact_name": artifact_name
            }
            print(f"✓ Artifacts saved successfully")
            
        except Exception as e:
            print(f"⚠ Failed to save artifacts: {e}")
    
    # Example prediction on new text
    print(f"\nExample prediction:")
    sample_text = ["God is great and religion brings peace to the world"]
    
    # Transform sample text using the fitted feature extractor
    try:
        sample_transformed = feature_extractor.transform(sample_text)
        prediction = trained_model.predict(sample_transformed)[0]
        probability = trained_model.predict_proba(sample_transformed)[0]
        
        print(f"Sample text: {sample_text[0]}")
        print(f"Predicted class: {target_names[prediction]}")
        print(f"Prediction probabilities: {dict(zip(target_names, probability))}")
    except Exception as e:
        print(f"Error transforming sample text: {e}")
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "predictions": y_pred,
        "target_names": target_names,
        "fitted_extractor": feature_extractor,
        "trained_model": trained_model,
        "saved_paths": saved_paths,
        "artifact_name": artifact_name,
        "training_time": time.time()  # Could be used for performance tracking
    } 