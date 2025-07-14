#!/usr/bin/env python3
"""Test script for HyperparamPicker system."""

import os
from dotenv import load_dotenv

from DataLoader import DataSourceType
from FeatureExtractor import FeatureExtractorType
from Model import ModelType
from HyperparamPicker import (
    create_hyperparam_picker,
    HyperparamSearchConfig,
    MultiObjectiveConfig,
    ParameterSpec
)
from HyperparamPicker.factory import create_tfidf_logistic_regression_config, run_hyperparameter_optimization
from ax.core import ParameterType

# Load environment variables
load_dotenv()

def test_tfidf_hyperparameter_optimization():
    """Test hyperparameter optimization with TF-IDF + Logistic Regression."""
    
    print("Testing TF-IDF + Logistic Regression hyperparameter optimization...")
    
    # Configuration
    config = create_tfidf_logistic_regression_config()
    
    # Run optimization with small number of trials for testing
    results = run_hyperparameter_optimization(
        data_source_type=config["data_source_type"],
        feature_extractor_type=config["feature_extractor_type"],
        model_type=config["model_type"],
        loader_kwargs=config["loader_kwargs"],
        base_extractor_kwargs=config["base_extractor_kwargs"],
        base_model_kwargs=config["base_model_kwargs"],
        total_trials=5,  # Small number for testing
        max_parallel_trials=1,  # Sequential for testing
        objectives=[("accuracy", False), ("training_time", True)],
        experiment_name="tfidf_logistic_test"
    )
    
    print("\nOptimization Results:")
    print(f"Total trials: {len(results.optimization_history)}")
    print(f"Best accuracy trial: {results.best_single_objective.get('accuracy', 'Not found')}")
    print(f"Best training time trial: {results.best_single_objective.get('training_time', 'Not found')}")
    
    # Export results
    results.export_results("tfidf_optimization_results.json")
    
    return results

def test_custom_hyperparameter_space():
    """Test creating a custom hyperparameter search space."""
    
    print("\nTesting custom hyperparameter search space...")
    
    # Create custom search configuration
    search_config = HyperparamSearchConfig()
    
    # Custom TF-IDF parameters
    search_config.feature_extractor_params[FeatureExtractorType.TFIDF_VECTORIZER] = {
        "max_features": ParameterSpec(
            name="max_features",
            param_type="choice",
            values=[1000, 5000, 10000],
            parameter_type=ParameterType.INT,
            is_ordered=True
        ),
        "min_df": ParameterSpec(
            name="min_df",
            param_type="range",
            lower=1,
            upper=5,
            parameter_type=ParameterType.INT
        )
    }
    
    # Custom model parameters
    search_config.model_params[ModelType.LOGISTIC_REGRESSION] = {
        "C": ParameterSpec(
            name="C",
            param_type="range",
            lower=0.01,
            upper=100.0,
            parameter_type=ParameterType.FLOAT,
            log_scale=True
        ),
        "max_iter": ParameterSpec(
            name="max_iter",
            param_type="choice",
            values=[100, 500, 1000],
            parameter_type=ParameterType.INT,
            is_ordered=True
        )
    }
    
    # Custom multi-objective configuration
    multi_objective_config = MultiObjectiveConfig(
        objectives=[("accuracy", False), ("f1_macro", False), ("training_time", True)],
        objective_thresholds={
            "accuracy": 0.75,
            "f1_macro": 0.70,
            "training_time": 180
        }
    )
    
    # Create picker with custom configuration
    picker = create_hyperparam_picker(
        search_config=search_config,
        multi_objective_config=multi_objective_config,
        experiment_name="custom_hyperparameter_test"
    )
    
    # Run optimization
    results = picker.optimize(
        data_source_type=DataSourceType.CSV_FILE,
        feature_extractor_type=FeatureExtractorType.TFIDF_VECTORIZER,
        model_type=ModelType.LOGISTIC_REGRESSION,
        loader_kwargs={
            "file_path": "dataset.csv",
            "text_column": "customer_review",
            "target_column": "return",
            "sep": "\t"
        },
        base_extractor_kwargs={},
        base_model_kwargs={},
        total_trials=4,
        max_parallel_trials=1,
        use_class_weights=True
    )
    
    print("\nCustom Hyperparameter Optimization Results:")
    print(f"Total trials: {len(results.optimization_history)}")
    print(f"Pareto frontier size: {len(results.pareto_frontier)}")
    
    # Get recommended configurations
    recommendations = results.get_recommended_configs()
    print(f"Number of recommended configurations: {len(recommendations)}")
    
    # Export results
    results.export_results("custom_hyperparameter_optimization_results.json")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting HyperparamPicker System Tests")
    print("=" * 60)
    
    try:
        # Test 1: TF-IDF + Logistic Regression
        tfidf_results = test_tfidf_hyperparameter_optimization()
        
        # Test 2: Custom hyperparameter space
        custom_results = test_custom_hyperparameter_space()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("📊 Results exported to JSON files for analysis")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 HyperparamPicker System Test Complete!") 