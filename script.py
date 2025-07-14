"""
Simple production script for TF-IDF + Logistic Regression text classification.
Uses only the data loader from existing code, everything else is standalone.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json

# Import only the data loader from existing code
from DataLoader import DataSourceType, create_data_loader


class TextClassificationModel:
    """Simple text classification model using TF-IDF + Logistic Regression."""
    
    def __init__(self, max_features=10000, min_df=2, max_df=0.8, use_class_weights=True):
        """
        Initialize the model.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
            use_class_weights: Whether to use class weights for imbalanced data
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_class_weights = use_class_weights
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        self.is_trained = False
        self.class_weights = None
        self.target_names = None
        
    def fit(self, X_train, y_train, target_names):
        """
        Train the model on training data.
        
        Args:
            X_train: Training text data (pandas Series or list)
            y_train: Training labels (pandas Series or list)
            target_names: List of class names
        """
        print("Training TF-IDF + Logistic Regression model...")
        
        # Store target names
        self.target_names = target_names
        
        # Calculate class weights if enabled
        if self.use_class_weights:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            self.class_weights = dict(zip(classes, weights))
            print(f"Using class weights: {self.class_weights}")
            
            # Set class weights in classifier
            self.classifier.set_params(class_weight=self.class_weights)
        
        # Transform text to TF-IDF features
        print("Transforming text to TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"Created {X_train_tfidf.shape[1]} TF-IDF features")
        
        # Train classifier
        print("Training logistic regression classifier...")
        self.classifier.fit(X_train_tfidf, y_train)
        
        self.is_trained = True
        print("Model training completed!")
        
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test text data (pandas Series or list)
            
        Returns:
            numpy array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_tfidf)
    
    def predict_proba(self, X_test):
        """
        Get prediction probabilities.
        
        Args:
            X_test: Test text data (pandas Series or list)
            
        Returns:
            numpy array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.classifier.predict_proba(X_test_tfidf)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test text data
            y_test: True test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Print results
        print(f"\nModel Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.target_names))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': y_pred
        }
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'target_names': self.target_names,
            'class_weights': self.class_weights,
            'config': {
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'use_class_weights': self.use_class_weights
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.target_names = model_data['target_names']
        self.class_weights = model_data['class_weights']
        
        # Load config
        config = model_data['config']
        self.max_features = config['max_features']
        self.min_df = config['min_df']
        self.max_df = config['max_df']
        self.use_class_weights = config['use_class_weights']
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def main():
    """Main training and evaluation pipeline."""
    
    # Configuration
    CSV_FILE = "dataset.csv"
    TEXT_COLUMN = "customer_review"
    TARGET_COLUMN = "return"
    SEP = "\t"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Load data using existing data loader
    print("Loading data...")
    data_loader = create_data_loader(
        DataSourceType.CSV_FILE,
        file_path=CSV_FILE,
        text_column=TEXT_COLUMN,
        target_column=TARGET_COLUMN,
        sep=SEP
    )
    
    df, target_names = data_loader.load_data()
    print(target_names)
    raise Exception("Stop here")
    
    # Print dataset info
    print(f"Loaded {len(df)} samples with {len(target_names)} classes")
    print(f"Target distribution:")
    print(df['target'].value_counts())
    
    # Split data
    print(f"\nSplitting data (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['target'], 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=df['target']
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and train model
    model = TextClassificationModel(
        max_features=10000,
        min_df=2,
        max_df=0.8,
        use_class_weights=True
    )
    
    model.fit(X_train, y_train, target_names)
    
    # Evaluate model
    results = model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model("trained_model.pkl")
    
    # Save results
    with open("training_results.json", "w") as f:
        json.dump({
            'accuracy': results['accuracy'],
            'f1_macro': results['f1_macro'],
            'f1_weighted': results['f1_weighted'],
            'dataset_info': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'classes': target_names,
                'class_distribution': df['target'].value_counts().to_dict()
            }
        }, f, indent=2)
    
    print("\nTraining completed successfully!")
    print("Model saved to: trained_model.pkl")
    print("Results saved to: training_results.json")
    
    # Example prediction
    print("\nExample prediction:")
    sample_text = ["This product is terrible and I want to return it"]
    prediction = model.predict(sample_text)[0]
    probabilities = model.predict_proba(sample_text)[0]
    
    print(f"Sample text: {sample_text[0]}")
    print(f"Predicted class: {target_names[prediction]}")
    print(f"Prediction probabilities: {dict(zip(target_names, probabilities))}")


if __name__ == "__main__":
    main() 