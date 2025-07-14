"""Main entry point for text classification experiments and pipeline execution."""

import os
from dotenv import load_dotenv
from DataLoader import DataSourceType
from FeatureExtractor import FeatureExtractorType  
from SupervisedModel import SupervisedModelType
from Pipeline import run_pipeline
from Experiments import run_experiments

# Load environment variables from .env file
load_dotenv()


if __name__ == "__main__":
    # Configuration: Choose between single run or experiment mode
    RUN_EXPERIMENTS = True  # Set to False for single run mode
    
    if RUN_EXPERIMENTS:
        # Define experiment configurations - Logistic Regression Feature Comparison
        experiment_configs = [
            {
                "data_source_type": DataSourceType.CSV_FILE,
                "feature_extractor_type": FeatureExtractorType.COUNT_VECTORIZER,
                "model_type": SupervisedModelType.LOGISTIC_REGRESSION,
                "use_class_weights": True,
                "loader_kwargs": {"file_path": "dataset.csv", "text_column": "customer_review", "target_column": "return", "sep": "\t"},
                "extractor_kwargs": {"max_features": 10000},
                "model_kwargs": {},
                "description": "Logistic Regression + Count Vectorizer (with class weights)"
            },
            {
                "data_source_type": DataSourceType.CSV_FILE,
                "feature_extractor_type": FeatureExtractorType.TFIDF_VECTORIZER,
                "model_type": SupervisedModelType.LOGISTIC_REGRESSION,
                "use_class_weights": True,
                "loader_kwargs": {"file_path": "dataset.csv", "text_column": "customer_review", "target_column": "return", "sep": "\t"},
                "extractor_kwargs": {"max_features": 10000, "min_df": 2, "max_df": 0.8},
                "model_kwargs": {},
                "description": "Logistic Regression + TF-IDF Vectorizer (with class weights)"
            },
            {
                "data_source_type": DataSourceType.CSV_FILE,
                "feature_extractor_type": FeatureExtractorType.HUGGINGFACE_TRANSFORMER,
                "model_type": SupervisedModelType.LOGISTIC_REGRESSION,
                "use_class_weights": True,
                "loader_kwargs": {"file_path": "dataset.csv", "text_column": "customer_review", "target_column": "return", "sep": "\t"},
                "extractor_kwargs": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
                "model_kwargs": {},
                "description": "Logistic Regression + Sentence Transformer (with class weights)"
            }
        ]
        
        # Run experiments
        run_experiments(experiment_configs, "experiment_results.json")
    
    else:
        # Single run configuration
        DATA_SOURCE = DataSourceType.CSV_FILE  # or DataSourceType.NEWSGROUPS
        FEATURE_EXTRACTOR = FeatureExtractorType.TFIDF_VECTORIZER  # or FeatureExtractorType.COUNT_VECTORIZER or FeatureExtractorType.HUGGINGFACE_TRANSFORMER
        MODEL = SupervisedModelType.LOGISTIC_REGRESSION  # or SupervisedModelType.PYTORCH_NEURAL_NETWORK or SupervisedModelType.KNN_CLASSIFIER
        USE_CLASS_WEIGHTS = False  # Enable class weights to handle imbalanced data
        
        # Configure data loader arguments based on selection
        loader_kwargs = {}
        if DATA_SOURCE == DataSourceType.CSV_FILE:
            loader_kwargs = {
                "file_path": "dataset.csv",
                "text_column": "customer_review",
                "target_column": "return",
                "sep": "\t"
            }
        elif DATA_SOURCE == DataSourceType.NEWSGROUPS:
            loader_kwargs = {
                "categories": ['alt.atheism', 'soc.religion.christian'],
                "subset": "train",
                "shuffle": True,
                "random_state": 42
            }
        
        # Configure extractor and model arguments based on selection
        extractor_kwargs = {}
        model_kwargs = {}
        
        if FEATURE_EXTRACTOR == FeatureExtractorType.COUNT_VECTORIZER:
            extractor_kwargs = {"max_features": 10000}
        elif FEATURE_EXTRACTOR == FeatureExtractorType.TFIDF_VECTORIZER:
            extractor_kwargs = {"max_features": 10000, "min_df": 2, "max_df": 0.8}
        elif FEATURE_EXTRACTOR == FeatureExtractorType.HUGGINGFACE_TRANSFORMER:
            extractor_kwargs = {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
        
        if MODEL == SupervisedModelType.PYTORCH_NEURAL_NETWORK:
            model_kwargs = {"hidden_size": 128, "epochs": 50}
        elif MODEL == SupervisedModelType.KNN_CLASSIFIER:
            model_kwargs = {"n_neighbors": 5, "weights": "uniform"}
        
        # Run the main pipeline
        results = run_pipeline(
            data_source_type=DATA_SOURCE,
            feature_extractor_type=FEATURE_EXTRACTOR,
            model_type=MODEL,
            use_class_weights=USE_CLASS_WEIGHTS,
            loader_kwargs=loader_kwargs,
            extractor_kwargs=extractor_kwargs,
            model_kwargs=model_kwargs
        )
        
        print(f"\nPipeline completed successfully!")
        print(f"Final Results: Accuracy={results['accuracy']:.4f}, F1-Macro={results['f1_macro']:.4f}")
