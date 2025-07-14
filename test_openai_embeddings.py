#!/usr/bin/env python3
"""Test script for OpenAI embeddings feature extractor."""

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from FeatureExtractor import create_feature_extractor, FeatureExtractorType

# Load environment variables from .env file
load_dotenv()

def test_openai_embeddings():
    """Test the OpenAI embeddings extractor."""
    
    # Sample text data
    train_texts = [
        "This is a positive review about the product.",
        "I love this item, it's amazing!",
        "This product is terrible and I hate it.",
        "Not satisfied with this purchase.",
        "Great quality and fast delivery!"
    ]
    
    test_texts = [
        "Excellent product, highly recommend!",
        "Poor quality, would not buy again.",
        "Average product, nothing special."
    ]
    
    # Convert to pandas Series
    X_train = pd.Series(train_texts)
    X_test = pd.Series(test_texts)
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found. Please check your .env file.")
    
    print("Testing OpenAI Embeddings Extractor...")
    print("=" * 50)
    
    try:
        # Create the OpenAI embeddings extractor
        extractor = create_feature_extractor(
            FeatureExtractorType.OPENAI_EMBEDDINGS,
            api_key=api_key,
            model_name="text-embedding-3-small",
            batch_size=5
        )
        
        print("✓ Extractor created successfully")
        
        # Fit and transform the data
        print("\nProcessing embeddings...")
        X_train_transformed, X_test_transformed = extractor.fit_transform(X_train, X_test)
        
        print(f"✓ Training embeddings shape: {X_train_transformed.shape}")
        print(f"✓ Test embeddings shape: {X_test_transformed.shape}")
        
        # Get feature info
        feature_info = extractor.get_feature_info()
        print(f"\nFeature Info:")
        for key, value in feature_info.items():
            print(f"  {key}: {value}")
        
        # Test transform on new data
        print("\nTesting transform on new data...")
        new_texts = ["This is a new text to transform"]
        new_embeddings = extractor.transform(new_texts)
        print(f"✓ New text embeddings shape: {new_embeddings.shape}")
        
        # Show some sample embedding values
        print(f"\nSample embedding values (first 5 dimensions):")
        print(f"First training sample: {X_train_transformed[0][:5]}")
        print(f"First test sample: {X_test_transformed[0][:5]}")
        
        print(f"\n✓ OpenAI embeddings extractor is working correctly!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_openai_embeddings() 