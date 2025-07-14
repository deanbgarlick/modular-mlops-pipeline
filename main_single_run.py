"""Single experiment execution for text classification pipeline."""

import os
from dotenv import load_dotenv
from DataLoader import DataSourceType
from FeatureExtractor import FeatureExtractorType  
from SupervisedModel import SupervisedModelType
from Pipeline import run_pipeline

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


def run_single_experiment(persistence_config):
    """Run a single ML experiment with persistence."""
    print("\n" + "="*60)
    print("SINGLE EXPERIMENT MODE")
    print("="*60)
    
    # Single run configuration
    DATA_SOURCE = DataSourceType.NEWSGROUPS  # or DataSourceType.CSV_FILE or DataSourceType.GCP_CSV_FILE
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
    elif DATA_SOURCE == DataSourceType.GCP_CSV_FILE:
        loader_kwargs = {
            "bucket_name": os.getenv("GCP_DATA_BUCKET", "your-data-bucket"),
            "file_path": "training_data.csv",
            "text_column": "text",
            "target_column": "label"
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
    
    # Run the main pipeline with persistence
    results = run_pipeline(
        data_source_type=DATA_SOURCE,
        feature_extractor_type=FEATURE_EXTRACTOR,
        model_type=MODEL,
        use_class_weights=USE_CLASS_WEIGHTS,
        loader_kwargs=loader_kwargs,
        extractor_kwargs=extractor_kwargs,
        model_kwargs=model_kwargs,
        # Persistence configuration
        **persistence_config,
        artifact_prefix="single_exp_"
    )
    
    print(f"\n🎉 Single Experiment Complete!")
    print(f"Final Results: Accuracy={results['accuracy']:.4f}, F1-Macro={results['f1_macro']:.4f}")
    
    # Show artifact information
    if results.get('saved_paths'):
        saved_paths = results['saved_paths']
        print(f"\n💾 Saved Artifacts:")
        print(f"  Extractor: {saved_paths['extractor_path']}")
        print(f"  Model: {saved_paths['model_path']}")
        print(f"  Storage: {saved_paths['bucket']}")
        print(f"  Artifact ID: {saved_paths['artifact_name']}")
    
    return results


def demonstrate_persistence_features():
    """Demonstrate advanced persistence features."""
    print("\n" + "="*60)
    print("PERSISTENCE FEATURES DEMO")
    print("="*60)
    
    persistence_config = setup_persistence_environment()
    
    print("\n📝 Usage Examples:")
    print("1. Train and save artifacts:")
    print("   python main_single_run.py  # Trains models and saves to GCP/local storage")
    
    print("\n2. Load existing artifacts (skip training):")
    print("   FORCE_RETRAIN=false python main_single_run.py")
    
    print("\n3. Force retrain (ignore existing artifacts):")
    print("   FORCE_RETRAIN=true python main_single_run.py")
    
    print("\n4. Use local storage instead of GCP:")
    print("   USE_GCP_PERSISTENCE=false python main_single_run.py")
    
    print("\n5. Disable artifact saving:")
    print("   SAVE_ML_ARTIFACTS=false python main_single_run.py")
    
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
    print("🚀 Single Experiment ML Pipeline with Cloud Persistence")
    print("=" * 60)
    
    # Setup persistence configuration
    persistence_config = setup_persistence_environment()
    
    # Configuration
    SHOW_PERSISTENCE_DEMO = True  # Set to True to show persistence features
    
    # Show persistence demo if enabled
    if SHOW_PERSISTENCE_DEMO:
        demonstrate_persistence_features()
    
    # Run single experiment with persistence
    run_single_experiment(persistence_config)
    
    print("\n" + "="*60)
    print("✅ Single experiment execution complete!")
    
    # Show next steps
    print("\n🔄 Next Steps:")
    if persistence_config['save_artifacts']:
        print("   • Check your GCP bucket or local storage for saved artifacts")
        print("   • Run again with FORCE_RETRAIN=false to load existing artifacts")
        print("   • Use saved artifacts for production inference")
    
    print("   • Experiment with different configurations")
    print("   • Run experiment suite using: python main_experiment_run.py")
    print("   • Deploy to GCP VMs using: python deploy.py")
    print("   • Run hyperparameter optimization: python test_hyperparameter_optimization.py") 