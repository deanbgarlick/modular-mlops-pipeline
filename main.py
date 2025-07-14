import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from typing import Optional, Dict, Any

from FeatureExtractor import (
    FeatureExtractor,
    FeatureExtractorType,
    create_feature_extractor
)
from Model import (
    Model,
    ModelType,
    create_model
)

def load_data():
    """Load and return the binary text classification dataset."""
    print("Loading binary text classification dataset...")
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    
    df = pd.DataFrame({
        'text': newsgroups_data.data,
        'target': newsgroups_data.target
    })
    
    print(f"Dataset loaded: {df.shape[0]} samples, {len(categories)} classes")
    print(f"Target classes: {newsgroups_data.target_names}")
    print(f"Target distribution:")
    print(df['target'].value_counts())
    
    return df, newsgroups_data.target_names

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

def main(feature_extractor_type: FeatureExtractorType = FeatureExtractorType.COUNT_VECTORIZER,
         model_type: ModelType = ModelType.LOGISTIC_REGRESSION,
         use_class_weights: bool = False,
         extractor_kwargs: Optional[Dict[str, Any]] = None, 
         model_kwargs: Optional[Dict[str, Any]] = None):
    """Main function to run the complete pipeline."""
    if extractor_kwargs is None:
        extractor_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
        
    print(f"Using feature extractor: {feature_extractor_type.value}")
    print(f"Using model: {model_type.value}")
    
    # Load data
    df, target_names = load_data()
    
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



if __name__ == "__main__":
    # Configuration: Choose which feature extractor, model, and options to use
    FEATURE_EXTRACTOR = FeatureExtractorType.COUNT_VECTORIZER  # or FeatureExtractorType.TFIDF_VECTORIZER or FeatureExtractorType.HUGGINGFACE_TRANSFORMER
    MODEL = ModelType.LOGISTIC_REGRESSION  # or ModelType.PYTORCH_NEURAL_NETWORK or ModelType.KNN_CLASSIFIER
    USE_CLASS_WEIGHTS = False  # Enable class weights to handle imbalanced data
    
    # Configure extractor and model arguments based on selection
    extractor_kwargs = {}
    model_kwargs = {}
    
    if FEATURE_EXTRACTOR == FeatureExtractorType.COUNT_VECTORIZER:
        extractor_kwargs = {"max_features": 10000}
    elif FEATURE_EXTRACTOR == FeatureExtractorType.TFIDF_VECTORIZER:
        extractor_kwargs = {"max_features": 10000, "min_df": 2, "max_df": 0.8}
    elif FEATURE_EXTRACTOR == FeatureExtractorType.HUGGINGFACE_TRANSFORMER:
        extractor_kwargs = {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
    
    if MODEL == ModelType.PYTORCH_NEURAL_NETWORK:
        model_kwargs = {"hidden_size": 128, "epochs": 50}
    elif MODEL == ModelType.KNN_CLASSIFIER:
        model_kwargs = {"n_neighbors": 5, "weights": "uniform"}
    
    # Run the main pipeline
    main(
        feature_extractor_type=FEATURE_EXTRACTOR,
        model_type=MODEL,
        use_class_weights=USE_CLASS_WEIGHTS,
        extractor_kwargs=extractor_kwargs,
        model_kwargs=model_kwargs
    )
