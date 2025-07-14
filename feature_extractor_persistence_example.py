#!/usr/bin/env python3
"""Example script demonstrating feature extractor persistence capabilities."""

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from FeatureExtractor import (
    FeatureExtractorType,
    create_feature_extractor,
    PickleLocalExtractorPersistence,
    PickleGCPExtractorPersistence,
    HuggingFaceExtractorPersistence
)


def demo_tfidf_local_persistence():
    """Demonstrate TF-IDF extractor with local persistence."""
    print("=" * 60)
    print("Demo: TF-IDF Extractor with Local Persistence")
    print("=" * 60)
    
    # Load sample data
    print("Loading 20 newsgroups data...")
    newsgroups = fetch_20newsgroups(
        categories=['alt.atheism', 'soc.religion.christian'],
        subset='train',
        shuffle=True,
        random_state=42
    )
    
    # Split data
    texts = pd.Series(newsgroups.data[:100])  # Use smaller subset for demo
    train_texts = pd.Series(texts[:80])
    test_texts = pd.Series(texts[80:])
    
    # Create extractor with local persistence
    local_persistence = PickleLocalExtractorPersistence(base_path="demo_extractors")
    
    extractor = create_feature_extractor(
        FeatureExtractorType.TFIDF_VECTORIZER,
        persistence=local_persistence,
        max_features=1000,
        min_df=2,
        max_df=0.8
    )
    
    # Fit and transform
    print("Fitting TF-IDF extractor...")
    X_train_transformed, X_test_transformed = extractor.fit_transform(train_texts, test_texts)
    
    print(f"Training features shape: {X_train_transformed.shape}")
    print(f"Test features shape: {X_test_transformed.shape}")
    print(f"Feature info: {extractor.get_feature_info()}")
    
    # Save the fitted extractor
    print("\nSaving fitted extractor...")
    extractor.save("tfidf_demo_extractor.pkl")
    
    # Load the extractor in a new instance
    print("Loading extractor from saved file...")
    new_extractor = create_feature_extractor(
        FeatureExtractorType.TFIDF_VECTORIZER,
        persistence=local_persistence
    )
    new_extractor.load("tfidf_demo_extractor.pkl")
    
    # Test the loaded extractor
    print("Testing loaded extractor...")
    new_test_features = new_extractor.transform(["This is a test document about religion."])
    print(f"New prediction features shape: {new_test_features.shape}")
    
    # Use class method for loading
    print("Using load_from_path class method...")
    loaded_extractor = create_feature_extractor.__self__.__class__.load_from_path(
        "tfidf_demo_extractor.pkl",
        persistence=local_persistence
    )
    print("✓ Successfully loaded using class method")


def demo_count_vectorizer_gcp_persistence():
    """Demonstrate Count Vectorizer with GCP persistence."""
    print("\n" + "=" * 60)
    print("Demo: Count Vectorizer with GCP Persistence (Mock)")
    print("=" * 60)
    
    # Note: This demo uses local persistence but shows the GCP interface
    # In production, you would use actual GCP credentials and bucket
    try:
        gcp_persistence = PickleGCPExtractorPersistence(
            bucket_name="demo-bucket-name",
            prefix="feature_extractors/demo/"
        )
        
        extractor = create_feature_extractor(
            FeatureExtractorType.COUNT_VECTORIZER,
            persistence=gcp_persistence,
            max_features=5000
        )
        
        print("✓ GCP persistence extractor created successfully")
        print("  (Note: Would require actual GCP credentials to save/load)")
        print(f"  Bucket: {gcp_persistence.bucket_name}")
        print(f"  Prefix: {gcp_persistence.prefix}")
        
    except ImportError as e:
        print(f"GCP dependencies not available: {e}")
        print("Install with: pip install google-cloud-storage")


def demo_huggingface_specialized_persistence():
    """Demonstrate HuggingFace extractor with specialized persistence."""
    print("\n" + "=" * 60)
    print("Demo: HuggingFace Extractor with Specialized Persistence")
    print("=" * 60)
    
    try:
        # Create HuggingFace persistence (local fallback)
        hf_persistence = HuggingFaceExtractorPersistence(
            bucket_name="demo-bucket",  # Required parameter (not used for local storage)
            use_gcp=False
        )
        
        extractor = create_feature_extractor(
            FeatureExtractorType.HUGGINGFACE_TRANSFORMER,
            persistence=hf_persistence,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print("✓ HuggingFace extractor with specialized persistence created")
        print("  (Note: Would require sentence-transformers package to run)")
        
    except ImportError as e:
        print(f"HuggingFace dependencies not available: {e}")
        print("Install with: pip install sentence-transformers")


def demo_persistence_comparison():
    """Compare different persistence options."""
    print("\n" + "=" * 60)
    print("Persistence Options Comparison")
    print("=" * 60)
    
    persistence_options = [
        ("Local Pickle", "PickleLocalExtractorPersistence", "Fast, local storage"),
        ("GCP Pickle", "PickleGCPExtractorPersistence", "Cloud storage, scalable"),
        ("AWS Pickle", "PickleAWSExtractorPersistence", "Cloud storage, AWS ecosystem"),
        ("HuggingFace Specialized", "HuggingFaceExtractorPersistence", "Optimized for transformers")
    ]
    
    print("Available persistence types:")
    for name, class_name, description in persistence_options:
        print(f"  • {name:20} ({class_name})")
        print(f"    {description}")
    
    print("\nKey Features:")
    print("  • Automatic persistence type selection based on extractor type")
    print("  • Save/load fitted extractors with all internal state")
    print("  • Support for sklearn vectorizers and HuggingFace transformers")
    print("  • Cloud storage integration (GCP, AWS)")
    print("  • Backward compatibility with direct pickle objects")


if __name__ == "__main__":
    print("🔧 Feature Extractor Persistence Demo")
    print("=====================================")
    
    try:
        # Run demos
        demo_tfidf_local_persistence()
        demo_count_vectorizer_gcp_persistence()
        demo_huggingface_specialized_persistence()
        demo_persistence_comparison()
        
        print("\n" + "=" * 60)
        print("✅ Feature Extractor Persistence Demo Complete!")
        print("✅ Check the 'demo_extractors/' directory for saved extractors")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Make sure you have the required dependencies installed.") 