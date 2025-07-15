"""Core machine learning pipeline functions for text classification."""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from typing import Optional, Dict, Any

from FeatureExtractor import (
    FeatureExtractorType, create_feature_extractor
)
from SupervisedModel import (
    SupervisedModelType, create_model
)
from DataLoader import DataSourceType, create_data_loader
from PipelineStep.pipeline_step import PipelineStep
from PipelineStep.persistence import GCPPipelineStepPersistence, LocalPipelineStepPersistence
from Pipeline.pipeline import Pipeline
from Pipeline.persistence import GCPPipelinePersistence, LocalPipelinePersistence


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


def create_pipeline_components(feature_extractor_type: FeatureExtractorType,
                              model_type: SupervisedModelType,
                              extractor_kwargs: Dict[str, Any],
                              model_kwargs: Dict[str, Any],
                              pipeline_persistence: Optional[Any] = None,
                              gcp_bucket_name: Optional[str] = None) -> Pipeline:
    """Create a pipeline with feature extraction and model components."""
    
    # Import local persistence classes
    from FeatureExtractor.persistence import PickleLocalExtractorPersistence
    from SupervisedModel.persistence import PickleLocalFilePersistence, TorchLocalFilePersistence
    from SupervisedModel import SupervisedModelType
    
    # Create local persistence for feature extractor
    extractor_persistence = PickleLocalExtractorPersistence(base_path="artifacts/feature_extractors")
    
    # Create local persistence for model (choose based on model type)
    if model_type == SupervisedModelType.PYTORCH_NEURAL_NETWORK:
        model_persistence = TorchLocalFilePersistence(base_path="artifacts/models")
    else:
        model_persistence = PickleLocalFilePersistence(base_path="artifacts/models")
    
    # Create feature extractor with local persistence
    feature_extractor = create_feature_extractor(
        feature_extractor_type,
        persistence=extractor_persistence,
        **extractor_kwargs
    )
    
    # Create model with local persistence
    model = create_model(
        model_type,
        persistence=model_persistence,
        **model_kwargs
    )
    
    # Set up persistence for PipelineSteps
    feature_extractor_persistence = None
    model_persistence = None
    
    # if gcp_bucket_name:
    if False:
        feature_extractor_persistence = GCPPipelineStepPersistence(
            bucket_name=gcp_bucket_name,
            prefix="pipeline_steps/feature_extractors/"
        )
        model_persistence = GCPPipelineStepPersistence(
            bucket_name=gcp_bucket_name,
            prefix="pipeline_steps/models/"
        )
        print(f"PipelineSteps will use GCP persistence in bucket: {gcp_bucket_name}")
    else:
        feature_extractor_persistence = LocalPipelineStepPersistence(
            base_path="pipeline_steps/feature_extractors"
        )
        model_persistence = LocalPipelineStepPersistence(
            base_path="pipeline_steps/models"
        )
        print("PipelineSteps will use local persistence")
    
    # Create pipeline step wrapper for feature extractor
    feature_pipeline_step = PipelineStep(
        component=feature_extractor,
        included_features=["text"],  # Use the 'text' column from our dataframes
        persistence=feature_extractor_persistence
    )
    
    # Create pipeline step wrapper for model  
    model_pipeline_step = PipelineStep(
        component=model,
        included_features=["*"],  # Use all feature columns
        excluded_features=["text", "target"],  # Exclude original text and target
        target_column="target",
        prediction_column="prediction",
        persistence=model_persistence
    )
    
    # Create pipeline with optional custom persistence
    pipeline = Pipeline([feature_pipeline_step, model_pipeline_step], persistence=pipeline_persistence)  # type: ignore[arg-type]
    
    return pipeline


def run_pipeline(data_source_type: DataSourceType = DataSourceType.NEWSGROUPS,
                 feature_extractor_type: FeatureExtractorType = FeatureExtractorType.COUNT_VECTORIZER,
                 model_type: SupervisedModelType = SupervisedModelType.LOGISTIC_REGRESSION,
                 use_class_weights: bool = False,
                 loader_kwargs: Optional[Dict[str, Any]] = None,
                 extractor_kwargs: Optional[Dict[str, Any]] = None, 
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 save_pipeline: bool = False,
                 gcp_bucket_name: Optional[str] = None,
                 pipeline_save_path: Optional[str] = None):
    """Run the complete machine learning pipeline using the Pipeline class.
    
    Args:
        data_source_type: Type of data source to use
        feature_extractor_type: Type of feature extractor to use
        model_type: Type of model to use
        use_class_weights: Whether to use class weights for imbalanced data
        loader_kwargs: Arguments for data loader
        extractor_kwargs: Arguments for feature extractor
        model_kwargs: Arguments for model
        save_pipeline: Whether to save the trained pipeline
        gcp_bucket_name: GCP bucket name for pipeline persistence (if None, uses local storage)
        pipeline_save_path: Path/name for saving the pipeline (if None, auto-generates)
    
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
    
    # Set up pipeline persistence
    pipeline_persistence = None
    # if save_pipeline and gcp_bucket_name:
    if save_pipeline and False:
        pipeline_persistence = GCPPipelinePersistence(gcp_bucket_name)
        print(f"Pipeline will be saved to GCP bucket: {gcp_bucket_name}")
    elif save_pipeline:
        pipeline_persistence = LocalPipelinePersistence(
            base_path="pipelines"
        )
        print("Pipeline will be saved locally")
    
    # Create pipeline
    pipeline = create_pipeline_components(
        feature_extractor_type, model_type,
        extractor_kwargs, model_kwargs,
        pipeline_persistence,
        gcp_bucket_name
    )
    
    # Train the pipeline
    print(f"\nTraining pipeline...")
    
    # Add class weights to model if requested
    # if use_class_weights:
    #      from sklearn.utils.class_weight import compute_class_weight
    #      classes = np.unique(train_df['target'].values)  # type: ignore[index]
    #      weights = compute_class_weight('balanced', classes=classes, y=train_df['target'].values)  # type: ignore[index]
    #      class_weights = dict(zip(classes, weights))
    #      print(f"Calculated class weights: {class_weights}")
    #     # Note: Pipeline doesn't currently support class weights in fit_transform

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Fit and transform training data, transform test data with explicit targets
    train_transformed, test_transformed = pipeline.fit_transform(
        train_df, test_df, 
        y_train=train_df['target'],  # type: ignore[index]
        y_test=test_df['target']  # type: ignore[index]
    )

    # Make predictions on test set
    # The final transformed data should be predictions from the model
    # Convert to numpy array for consistent handling
    if hasattr(test_transformed.prediction, 'toarray'):
        # Handle sparse matrix
        y_pred = test_transformed.prediction.toarray().flatten()  # type: ignore[attr-defined]
    elif hasattr(test_transformed.prediction, 'flatten'):
        y_pred = test_transformed.prediction.flatten()  # type: ignore[attr-defined]
    else:
        y_pred = np.array(test_transformed.prediction).flatten()
    
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
    
    # Example prediction on new text
    print(f"\nExample prediction:")
    sample_df = pd.DataFrame({
        'text': ["God is great and religion brings peace to the world"],
        'target': [0]  # Dummy target, not used for prediction
    })
    
    try:
        sample_prediction = pipeline.transform(sample_df)
        if hasattr(sample_prediction, 'toarray'):
            pred_class = sample_prediction.toarray().flatten()[0]  # type: ignore[attr-defined]
        elif hasattr(sample_prediction, 'flatten'):
            pred_class = sample_prediction.flatten()[0]  # type: ignore[attr-defined]
        else:
            pred_class = sample_prediction[0]
        
        # Handle probability vs class prediction
        if isinstance(pred_class, (float, np.floating)) and 0 <= pred_class <= 1:
            pred_class = int(pred_class > 0.5) if len(target_names) == 2 else int(np.round(pred_class))
        
        print(f"Sample text: {sample_df['text'].iloc[0]}")
        print(f"Predicted class: {target_names[int(pred_class)]}")
    except Exception as e:
        print(f"Error making sample prediction: {e}")
    
    # Save pipeline if requested
    if save_pipeline:
        if pipeline_save_path is None:
            # Auto-generate pipeline save path
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pipeline_save_path = f"pipeline_{feature_extractor_type.value}_{model_type.value}_{timestamp}"
        
        try:
            print(f"\nSaving pipeline to: {pipeline_save_path}")
            pipeline.save(pipeline_save_path)
            print("Pipeline saved successfully!")
        except Exception as e:
            print(f"Error saving pipeline: {e}")
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "predictions": y_pred,
        "target_names": target_names,
        "pipeline": pipeline,
        "training_time": time.time(),
        "pipeline_saved": save_pipeline,
        "pipeline_save_path": pipeline_save_path if save_pipeline else None
    } 