"""Experiment orchestration functionality for text classification."""

import json
import time
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from typing import List, Dict, Any

from DataLoader import DataSourceType, create_data_loader
from FeatureExtractor import FeatureExtractorType, create_feature_extractor
from SupervisedModel import SupervisedModelType, create_model
from Pipeline.run_pipeline import prepare_data, evaluate_model, create_pipeline_components


def run_experiments(experiment_configs: List[Dict[str, Any]], output_file: str = "experiment_results.json"):
    """
    Run multiple experiments with different vectorizer-model combinations.
    
    Args:
        experiment_configs: List of configuration dictionaries, each containing:
            - data_source_type: DataSourceType
            - feature_extractor_type: FeatureExtractorType 
            - model_type: ModelType
            - use_class_weights: bool (optional, default False)
            - loader_kwargs: dict (optional)
            - extractor_kwargs: dict (optional)
            - model_kwargs: dict (optional)
            - description: str (optional)
        output_file: Path to save JSON results
    """
    results = {
        "experiment_timestamp": datetime.now().isoformat(),
        "total_experiments": len(experiment_configs),
        "experiments": []
    }
    
    print(f"Running {len(experiment_configs)} experiments...")
    print("="*60)
    
    for i, config in enumerate(experiment_configs, 1):
        print(f"\nExperiment {i}/{len(experiment_configs)}")
        print(f"Description: {config.get('description', 'No description')}")
        
        # Set defaults
        data_source_type = config['data_source_type']
        feature_extractor_type = config['feature_extractor_type']
        model_type = config['model_type']
        use_class_weights = config.get('use_class_weights', False)
        loader_kwargs = config.get('loader_kwargs', {})
        extractor_kwargs = config.get('extractor_kwargs', {})
        model_kwargs = config.get('model_kwargs', {})
        
        # Start timing
        start_time = time.time()
        
        try:
            # Load data
            data_loader = create_data_loader(data_source_type, **loader_kwargs)
            df, target_names = data_loader.load_data()
            
            # Prepare train/test splits (returns DataFrames)
            train_df, test_df = prepare_data(df)
            
            # Clean data (ensure we have DataFrames)
            if hasattr(train_df, 'dropna'):
                train_df = train_df.dropna()
            if hasattr(test_df, 'dropna'):
                test_df = test_df.dropna()
            
            # Create pipeline with feature extraction and model components
            pipeline = create_pipeline_components(
                feature_extractor_type, model_type,
                extractor_kwargs, model_kwargs
            )
            
            # Train the pipeline and get transformed data
            print(f"Training pipeline...")
            try:
                train_transformed, test_transformed = pipeline.fit_transform(
                    train_df, test_df, 
                    y_train=train_df.get('target') if hasattr(train_df, 'get') else None,
                    y_test=test_df.get('target') if hasattr(test_df, 'get') else None
                )
            except TypeError:
                # Fallback if target parameters not supported
                train_transformed, test_transformed = pipeline.fit_transform(train_df, test_df)
            
            # Extract predictions from the transformed test data
            y_pred = None
            try:
                # Try to access prediction attribute if it exists
                if hasattr(test_transformed, 'prediction'):
                    y_pred = test_transformed.prediction  # type: ignore
                else:
                    # Fallback: assume the transformed data contains predictions
                    y_pred = test_transformed
                
                # Convert predictions to numpy array and handle different formats
                if hasattr(y_pred, 'toarray'):
                    # Handle sparse matrix
                    y_pred = y_pred.toarray().flatten()  # type: ignore
                elif hasattr(y_pred, 'flatten'):
                    y_pred = y_pred.flatten()  # type: ignore
                else:
                    y_pred = np.array(y_pred).flatten()
                
                # Convert predictions to class indices if they're probabilities
                if y_pred.ndim > 1 or (y_pred.dtype == float and len(np.unique(y_pred)) > len(target_names)):
                    # If we have probabilities, get the class with highest probability
                    if y_pred.ndim > 1:
                        y_pred = np.argmax(y_pred, axis=1)
                    else:
                        # Single column of probabilities, threshold at 0.5 for binary classification
                        y_pred = (y_pred > 0.5).astype(int)
                        
            except Exception as pred_error:
                raise ValueError(f"Failed to extract predictions from pipeline output: {pred_error}")
            
            # Get true labels
            try:
                if hasattr(test_df, 'get'):
                    y_test = test_df['target'].values  # type: ignore
                else:
                    # Fallback for list-like objects
                    y_test = np.array([row.get('target', 0) for row in test_df])  # type: ignore
            except Exception as label_error:
                raise ValueError(f"Failed to extract true labels: {label_error}")
            
            # Evaluate model (this function expects y_true, y_pred, target_names)
            accuracy, f1_macro, f1_weighted = evaluate_model(y_test, y_pred, target_names)
            
            # End timing
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Get component info for the experiment results
            feature_extractor = create_feature_extractor(feature_extractor_type, **extractor_kwargs)
            model = create_model(model_type, **model_kwargs)
            
            # Collect results
            experiment_result = {
                "experiment_id": i,
                "description": config.get('description', 'No description'),
                "configuration": {
                    "data_source": data_source_type.value,
                    "feature_extractor": feature_extractor_type.value,
                    "model": model_type.value,
                    "use_class_weights": use_class_weights,
                    "loader_kwargs": loader_kwargs,
                    "extractor_kwargs": extractor_kwargs,
                    "model_kwargs": model_kwargs
                },
                "data_info": data_loader.get_data_info(),
                "feature_info": feature_extractor.get_feature_info() if hasattr(feature_extractor, 'get_feature_info') else {},
                "model_info": model.get_model_info() if hasattr(model, 'get_model_info') else {},
                "performance": {
                    "accuracy": float(accuracy),
                    "f1_macro": float(f1_macro),
                    "f1_weighted": float(f1_weighted),
                    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                    "classification_report": classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                },
                "execution_time_seconds": execution_time,
                "status": "success"
            }
            
            print(f"✓ Completed successfully - Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
            
        except Exception as e:
            # Handle errors gracefully
            execution_time = time.time() - start_time
            experiment_result = {
                "experiment_id": i,
                "description": config.get('description', 'No description'),
                "configuration": {
                    "data_source": data_source_type.value,
                    "feature_extractor": feature_extractor_type.value,
                    "model": model_type.value,
                    "use_class_weights": use_class_weights,
                    "loader_kwargs": loader_kwargs,
                    "extractor_kwargs": extractor_kwargs,
                    "model_kwargs": model_kwargs
                },
                "error": str(e),
                "execution_time_seconds": execution_time,
                "status": "failed"
            }
            
            print(f"✗ Failed - Error: {e}")
        
        results["experiments"].append(experiment_result)
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiments completed! Results saved to: {output_file}")
    
    # Print summary
    successful_experiments = [exp for exp in results["experiments"] if exp["status"] == "success"]
    if successful_experiments:
        print(f"\nSummary of {len(successful_experiments)} successful experiments:")
        print("-" * 60)
        for exp in successful_experiments:
            desc = exp["description"]
            acc = exp["performance"]["accuracy"]
            f1 = exp["performance"]["f1_macro"]
            print(f"{desc}: Accuracy={acc:.4f}, F1-Macro={f1:.4f}")
    
    failed_experiments = [exp for exp in results["experiments"] if exp["status"] == "failed"]
    if failed_experiments:
        print(f"\nFailed experiments: {len(failed_experiments)}")
        for exp in failed_experiments:
            print(f"- {exp['description']}: {exp['error']}")
    
    return results 