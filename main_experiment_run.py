"""Experiment suite execution for text classification pipeline."""

import os
from dotenv import load_dotenv
from DataLoader import DataSourceType
from FeatureExtractor import FeatureExtractorType  
from SupervisedModel import SupervisedModelType
from Experiments import run_experiments

# Load environment variables from .env file
load_dotenv()


def setup_persistence_environment():
    """Setup and validate persistence environment variables."""
    
    # GCP bucket for ML artifacts
    gcp_bucket = os.getenv("GCP_ML_BUCKET")
    if not gcp_bucket:
        print("⚠ Warning: GCP_ML_BUCKET not set in environment variables")
        print("  Using default bucket name. Set in .env file for production:")
        print("  GCP_ML_BUCKET=your-project-id-ml-artifacts")
        gcp_bucket = "default-ml-artifacts"
    
    # Other persistence settings
    use_gcp = os.getenv("USE_GCP_PERSISTENCE", "true").lower() == "true"
    save_artifacts = os.getenv("SAVE_ML_ARTIFACTS", "true").lower() == "true"
    force_retrain = os.getenv("FORCE_RETRAIN", "false").lower() == "true"
    
    print(f"🔧 Persistence Configuration:")
    print(f"  GCP Bucket: {gcp_bucket}")
    print(f"  Use GCP Storage: {use_gcp}")
    print(f"  Save Artifacts: {save_artifacts}")
    print(f"  Force Retrain: {force_retrain}")
    
    return {
        "gcp_bucket": gcp_bucket,
        "use_gcp_persistence": use_gcp,
        "save_artifacts": save_artifacts,
        "force_retrain": force_retrain
    }


def run_experiment_suite(persistence_config):
    """Run experiment suite with persistence."""
    print("\n" + "="*60)
    print("EXPERIMENT SUITE MODE")
    print("="*60)
    
    # Define experiment configurations with persistence
    experiment_configs = [
        {
            "data_source_type": DataSourceType.NEWSGROUPS,
            "feature_extractor_type": FeatureExtractorType.COUNT_VECTORIZER,
            "model_type": SupervisedModelType.LOGISTIC_REGRESSION,
            "use_class_weights": True,
            "loader_kwargs": {"categories": ['alt.atheism', 'soc.religion.christian']},
            "extractor_kwargs": {"max_features": 10000},
            "model_kwargs": {},
            "description": "Logistic Regression + Count Vectorizer (with class weights)",
            # Add persistence to each experiment
            **persistence_config,
            "artifact_prefix": "exp1_"
        },
        {
            "data_source_type": DataSourceType.NEWSGROUPS,
            "feature_extractor_type": FeatureExtractorType.TFIDF_VECTORIZER,
            "model_type": SupervisedModelType.LOGISTIC_REGRESSION,
            "use_class_weights": True,
            "loader_kwargs": {"categories": ['alt.atheism', 'soc.religion.christian']},
            "extractor_kwargs": {"max_features": 10000, "min_df": 2, "max_df": 0.8},
            "model_kwargs": {},
            "description": "Logistic Regression + TF-IDF Vectorizer (with class weights)",
            **persistence_config,
            "artifact_prefix": "exp2_"
        },
        {
            "data_source_type": DataSourceType.NEWSGROUPS,
            "feature_extractor_type": FeatureExtractorType.TFIDF_VECTORIZER,
            "model_type": SupervisedModelType.PYTORCH_NEURAL_NETWORK,
            "use_class_weights": True,
            "loader_kwargs": {"categories": ['alt.atheism', 'soc.religion.christian']},
            "extractor_kwargs": {"max_features": 5000, "min_df": 2},
            "model_kwargs": {"hidden_size": 128, "epochs": 30},
            "description": "PyTorch NN + TF-IDF Vectorizer (with class weights)",
            **persistence_config,
            "artifact_prefix": "exp3_"
        }
    ]
    
    # Run experiments
    results = run_experiments(experiment_configs, "experiment_results_with_persistence.json")
    
    print(f"\n🎉 Experiment Suite Complete!")
    print(f"Results saved to: experiment_results_with_persistence.json")
    
    return results


def demonstrate_persistence_features():
    """Demonstrate advanced persistence features."""
    print("\n" + "="*60)
    print("PERSISTENCE FEATURES DEMO")
    print("="*60)
    
    persistence_config = setup_persistence_environment()
    
    print("\n📝 Usage Examples:")
    print("1. Train and save artifacts:")
    print("   python main_experiment_run.py  # Trains models and saves to GCP/local storage")
    
    print("\n2. Load existing artifacts (skip training):")
    print("   FORCE_RETRAIN=false python main_experiment_run.py")
    
    print("\n3. Force retrain (ignore existing artifacts):")
    print("   FORCE_RETRAIN=true python main_experiment_run.py")
    
    print("\n4. Use local storage instead of GCP:")
    print("   USE_GCP_PERSISTENCE=false python main_experiment_run.py")
    
    print("\n5. Disable artifact saving:")
    print("   SAVE_ML_ARTIFACTS=false python main_experiment_run.py")
    
    print("\n🔧 Environment Variables (.env file):")
    print("   GCP_ML_BUCKET=your-project-id-ml-artifacts")
    print("   USE_GCP_PERSISTENCE=true")
    print("   SAVE_ML_ARTIFACTS=true") 
    print("   FORCE_RETRAIN=false")
    print("   GCP_DATA_BUCKET=your-project-id-data")
    
    print("\n💡 Production Benefits:")
    print("   • Reuse fitted extractors across experiments")
    print("   • Share models between team members via cloud storage")
    print("   • Skip expensive retraining for inference")
    print("   • Version control your ML artifacts")
    print("   • Seamless deployment with pre-trained components")
    
    return persistence_config


if __name__ == "__main__":
    print("🚀 Experiment Suite ML Pipeline with Cloud Persistence")
    print("=" * 60)
    
    # Setup persistence configuration
    persistence_config = setup_persistence_environment()
    
    # Configuration
    SHOW_PERSISTENCE_DEMO = True  # Set to True to show persistence features
    
    # Show persistence demo if enabled
    if SHOW_PERSISTENCE_DEMO:
        demonstrate_persistence_features()
    
    # Run experiment suite with persistence
    run_experiment_suite(persistence_config)
    
    print("\n" + "="*60)
    print("✅ Experiment suite execution complete!")
    
    # Show next steps
    print("\n🔄 Next Steps:")
    if persistence_config['save_artifacts']:
        print("   • Check your GCP bucket or local storage for saved artifacts")
        print("   • Run again with FORCE_RETRAIN=false to load existing artifacts")
        print("   • Use saved artifacts for production inference")
    
    print("   • Experiment with different configurations")
    print("   • Run single experiments using: python main_single_run.py")
    print("   • Deploy to GCP VMs using: python deploy.py")
    print("   • Run hyperparameter optimization: python test_hyperparameter_optimization.py") 