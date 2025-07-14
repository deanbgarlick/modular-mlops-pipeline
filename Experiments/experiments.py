# """Experiment orchestration functionality for text classification."""

# import json
# import time
# from datetime import datetime
# from sklearn.metrics import f1_score, confusion_matrix, classification_report
# from typing import List, Dict, Any

# from DataLoader import DataSourceType, create_data_loader
# from FeatureExtractor import FeatureExtractorType, create_feature_extractor
# from SupervisedModel import SupervisedModelType, create_model
# from Pipeline import prepare_data, create_features, train_model, evaluate_model


# def run_experiments(experiment_configs: List[Dict[str, Any]], output_file: str = "experiment_results.json"):
#     """
#     Run multiple experiments with different vectorizer-model combinations.
    
#     Args:
#         experiment_configs: List of configuration dictionaries, each containing:
#             - data_source_type: DataSourceType
#             - feature_extractor_type: FeatureExtractorType 
#             - model_type: ModelType
#             - use_class_weights: bool (optional, default False)
#             - loader_kwargs: dict (optional)
#             - extractor_kwargs: dict (optional)
#             - model_kwargs: dict (optional)
#             - description: str (optional)
#         output_file: Path to save JSON results
#     """
#     results = {
#         "experiment_timestamp": datetime.now().isoformat(),
#         "total_experiments": len(experiment_configs),
#         "experiments": []
#     }
    
#     print(f"Running {len(experiment_configs)} experiments...")
#     print("="*60)
    
#     for i, config in enumerate(experiment_configs, 1):
#         print(f"\nExperiment {i}/{len(experiment_configs)}")
#         print(f"Description: {config.get('description', 'No description')}")
        
#         # Set defaults
#         data_source_type = config['data_source_type']
#         feature_extractor_type = config['feature_extractor_type']
#         model_type = config['model_type']
#         use_class_weights = config.get('use_class_weights', False)
#         loader_kwargs = config.get('loader_kwargs', {})
#         extractor_kwargs = config.get('extractor_kwargs', {})
#         model_kwargs = config.get('model_kwargs', {})
        
#         # Start timing
#         start_time = time.time()
        
#         try:
#             # Load data
#             data_loader = create_data_loader(data_source_type, **loader_kwargs)
#             df, target_names = data_loader.load_data()
            
#             # Prepare train/test splits
#             X_train, X_test, y_train, y_test = prepare_data(df)
            
#             # Create features
#             feature_extractor = create_feature_extractor(feature_extractor_type, **extractor_kwargs)
#             X_train_transformed, X_test_transformed = create_features(
#                 X_train, X_test, feature_extractor
#             )
            
#             # Create and train model
#             model = create_model(model_type, **model_kwargs)
#             trained_model = train_model(X_train_transformed, y_train, model, use_class_weights)
            
#             # Evaluate model
#             accuracy, f1_macro, f1_weighted, y_pred, y_pred_proba = evaluate_model(trained_model, X_test_transformed, y_test, target_names)
            
#             # Calculate additional metrics (no longer needed as they're returned from evaluate_model)
#             # f1_macro = f1_score(y_test, y_pred, average='macro')
#             # f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
#             # End timing
#             end_time = time.time()
#             execution_time = end_time - start_time
            
#             # Collect results
#             experiment_result = {
#                 "experiment_id": i,
#                 "description": config.get('description', 'No description'),
#                 "configuration": {
#                     "data_source": data_source_type.value,
#                     "feature_extractor": feature_extractor_type.value,
#                     "model": model_type.value,
#                     "use_class_weights": use_class_weights,
#                     "loader_kwargs": loader_kwargs,
#                     "extractor_kwargs": extractor_kwargs,
#                     "model_kwargs": model_kwargs
#                 },
#                 "data_info": data_loader.get_data_info(),
#                 "feature_info": feature_extractor.get_feature_info(), # Assuming feature_extractor has this method
#                 "model_info": trained_model.get_model_info(),
#                 "performance": {
#                     "accuracy": float(accuracy),
#                     "f1_macro": float(f1_macro),
#                     "f1_weighted": float(f1_weighted),
#                     "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
#                     "classification_report": classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
#                 },
#                 "execution_time_seconds": execution_time,
#                 "status": "success"
#             }
            
#             print(f"✓ Completed successfully - Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
            
#         except Exception as e:
#             # Handle errors gracefully
#             execution_time = time.time() - start_time
#             experiment_result = {
#                 "experiment_id": i,
#                 "description": config.get('description', 'No description'),
#                 "configuration": {
#                     "data_source": data_source_type.value,
#                     "feature_extractor": feature_extractor_type.value,
#                     "model": model_type.value,
#                     "use_class_weights": use_class_weights,
#                     "loader_kwargs": loader_kwargs,
#                     "extractor_kwargs": extractor_kwargs,
#                     "model_kwargs": model_kwargs
#                 },
#                 "error": str(e),
#                 "execution_time_seconds": execution_time,
#                 "status": "failed"
#             }
            
#             print(f"✗ Failed - Error: {e}")
        
#         results["experiments"].append(experiment_result)
    
#     # Save results to JSON file
#     with open(output_file, 'w') as f:
#         json.dump(results, f, indent=2)
    
#     print(f"\n{'='*60}")
#     print(f"Experiments completed! Results saved to: {output_file}")
    
#     # Print summary
#     successful_experiments = [exp for exp in results["experiments"] if exp["status"] == "success"]
#     if successful_experiments:
#         print(f"\nSummary of {len(successful_experiments)} successful experiments:")
#         print("-" * 60)
#         for exp in successful_experiments:
#             desc = exp["description"]
#             acc = exp["performance"]["accuracy"]
#             f1 = exp["performance"]["f1_macro"]
#             print(f"{desc}: Accuracy={acc:.4f}, F1-Macro={f1:.4f}")
    
#     failed_experiments = [exp for exp in results["experiments"] if exp["status"] == "failed"]
#     if failed_experiments:
#         print(f"\nFailed experiments: {len(failed_experiments)}")
#         for exp in failed_experiments:
#             print(f"- {exp['description']}: {exp['error']}")
    
#     return results 