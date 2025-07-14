"""Core machine learning pipeline functions for text classification."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from typing import Optional, Dict, Any

from FeatureExtractor import FeatureExtractor, FeatureExtractorType, create_feature_extractor
from Model import Model, ModelType, create_model
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
    X_train_transformed, X_test_transformed = feature_extractor.fit_transform(X_train, X_test)
    
    # Print feature information
    feature_info = feature_extractor.get_feature_info()
    print(f"Feature matrix shape: {X_train_transformed.shape}")
    for key, value in feature_info.items():
        print(f"{key}: {value}")
    
    return X_train_transformed, X_test_transformed, feature_extractor


def train_model(X_train, y_train, model: Model, use_class_weights: bool = False):
    """Train the specified model with optional class weighting."""
    print(f"\nTraining model...")
    
    class_weights = None
    if use_class_weights:
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print(f"Calculated class weights: {class_weights}")
    
    model.fit(X_train, y_train, class_weights=class_weights)
    
    # Print model info
    model_info = model.get_model_info()
    print("Model info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate model performance on test set."""
    print(f"\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return y_pred, accuracy


def run_pipeline(data_source_type: DataSourceType = DataSourceType.NEWSGROUPS,
                 feature_extractor_type: FeatureExtractorType = FeatureExtractorType.COUNT_VECTORIZER,
                 model_type: ModelType = ModelType.LOGISTIC_REGRESSION,
                 use_class_weights: bool = False,
                 loader_kwargs: Optional[Dict[str, Any]] = None,
                 extractor_kwargs: Optional[Dict[str, Any]] = None, 
                 model_kwargs: Optional[Dict[str, Any]] = None):
    """Run the complete machine learning pipeline."""
    if loader_kwargs is None:
        loader_kwargs = {}
    if extractor_kwargs is None:
        extractor_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
        
    print(f"Using data source: {data_source_type.value}")
    print(f"Using feature extractor: {feature_extractor_type.value}")
    print(f"Using model: {model_type.value}")
    
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
    
    # Create feature extractor using factory function
    feature_extractor = create_feature_extractor(
        feature_extractor_type, **extractor_kwargs
    )
    
    # Create features using the chosen extractor
    X_train_transformed, X_test_transformed, fitted_extractor = create_features(
        X_train, X_test, feature_extractor
    )
    
    # Create model using factory function
    model = create_model(model_type, **model_kwargs)
    
    # Train model
    trained_model = train_model(X_train_transformed, y_train, model, use_class_weights)
    
    # Evaluate model
    y_pred, accuracy = evaluate_model(trained_model, X_test_transformed, y_test, target_names)
    
    # Example prediction on new text
    print(f"\nExample prediction:")
    sample_text = ["God is great and religion brings peace to the world"]
    
    # Transform sample text using the fitted feature extractor
    try:
        sample_transformed = fitted_extractor.transform(sample_text)
        prediction = trained_model.predict(sample_transformed)[0]
        probability = trained_model.predict_proba(sample_transformed)[0]
        
        print(f"Sample text: {sample_text[0]}")
        print(f"Predicted class: {target_names[prediction]}")
        print(f"Prediction probabilities: {dict(zip(target_names, probability))}")
    except Exception as e:
        print(f"Error transforming sample text: {e}")
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        "predictions": y_pred,
        "target_names": target_names,
        "fitted_extractor": fitted_extractor,
        "trained_model": trained_model
    } 