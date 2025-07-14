# Modular Text Classification Pipeline

A clean, extensible machine learning pipeline for binary text classification using different feature extraction methods and model architectures.

## 🏗️ Architecture

The project uses **Strategy Pattern** and **Factory Functions** to create a modular, extensible system:

```
├── main.py                     # Main pipeline orchestrator
├── FeatureExtractor/           # Feature extractor package
│   ├── __init__.py            # Package exports
│   ├── base.py                # Abstract base class & enum
│   ├── count_vectorizer.py    # Count vectorizer implementation
│   ├── huggingface.py         # HuggingFace transformer implementation
│   └── factory.py             # Factory function for creating extractors
├── Model/                      # Model package
│   ├── __init__.py            # Package exports
│   ├── base.py                # Abstract base class & enum
│   ├── logistic_regression.py # Sklearn logistic regression wrapper
│   ├── pytorch_neural_network.py # PyTorch neural network implementation
│   └── factory.py             # Factory function for creating models
├── requirements.txt           # Python dependencies
└── test_data.csv             # Sample data (optional)
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with default configuration (Count Vectorizer + Logistic Regression):**
   ```bash
   python main.py
   ```

3. **Switch configurations:**
   Edit `main.py` lines 178-179:
   ```python
   FEATURE_EXTRACTOR = FeatureExtractorType.COUNT_VECTORIZER
   MODEL = ModelType.PYTORCH_NEURAL_NETWORK
   ```

## 🔧 Available Feature Extractors

### Count Vectorizer
- **Type:** `FeatureExtractorType.COUNT_VECTORIZER`
- **Features:** Sparse count-based vectors (10,000 features)
- **Performance:** ~97% accuracy, very fast
- **Use case:** Traditional ML, interpretable features

### HuggingFace Transformers
- **Type:** `FeatureExtractorType.HUGGINGFACE_TRANSFORMER`
- **Features:** Dense embeddings (384 dimensions)
- **Performance:** ~89% accuracy, semantic understanding
- **Use case:** Semantic similarity, modern NLP

## 🤖 Available Models

### Logistic Regression
- **Type:** `ModelType.LOGISTIC_REGRESSION`
- **Implementation:** Sklearn wrapper
- **Performance:** ~97% accuracy with count vectorizer
- **Use case:** Fast, interpretable, reliable baseline

### PyTorch Neural Network
- **Type:** `ModelType.PYTORCH_NEURAL_NETWORK`
- **Implementation:** 3-layer feedforward network
- **Performance:** ~96% accuracy with count vectorizer
- **Use case:** Deep learning, non-linear patterns

## 🎯 Results Comparison

| Feature Extractor | Model | Accuracy | F1-Score | Training Time |
|-------------------|-------|----------|----------|---------------|
| Count Vectorizer | Logistic Regression | 97.22% | 97.17% | Fast ⚡ |
| Count Vectorizer | PyTorch NN | 96.30% | 96.21% | Moderate 🔥 |
| HuggingFace | Logistic Regression | 88.89% | 88.64% | Slow 🐌 |

## 🔌 Adding New Components

### New Feature Extractors
1. **Create new extractor class** inheriting from `FeatureExtractor`
2. **Add enum value** to `FeatureExtractorType`
3. **Update factory function** in `FeatureExtractor/factory.py`
4. **Implement required methods:**
   - `fit_transform(X_train, X_test)`
   - `transform(X)`
   - `get_feature_info()`

### New Models
1. **Create new model class** inheriting from `Model`
2. **Add enum value** to `ModelType`
3. **Update factory function** in `Model/factory.py`
4. **Implement required methods:**
   - `fit(X_train, y_train)`
   - `predict(X)`
   - `predict_proba(X)`
   - `get_model_info()`

## 🎮 Usage Examples

```python
# Count Vectorizer + Logistic Regression (default)
main(
    feature_extractor_type=FeatureExtractorType.COUNT_VECTORIZER,
    model_type=ModelType.LOGISTIC_REGRESSION,
    extractor_kwargs={"max_features": 10000}
)

# HuggingFace + PyTorch Neural Network
main(
    feature_extractor_type=FeatureExtractorType.HUGGINGFACE_TRANSFORMER,
    model_type=ModelType.PYTORCH_NEURAL_NETWORK,
    extractor_kwargs={"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
    model_kwargs={"hidden_size": 256, "epochs": 100}
)
```

## 📊 Dataset

Uses **20 Newsgroups** dataset with 2 categories:
- `alt.atheism` (480 samples)
- `soc.religion.christian` (599 samples)

**Pipeline steps:**
1. Load data → 2. Train/test split → 3. Feature extraction → 4. Model creation → 5. Training → 6. Evaluation

## 🛠️ Key Features

- ✅ **Modular Design** - Mix and match any feature extractor with any model
- ✅ **Type Safety** - Enum-based configuration prevents errors
- ✅ **Clean Interface** - Consistent API across all components
- ✅ **Factory Functions** - Centralized object creation
- ✅ **Package Organization** - Clean separation of concerns
- ✅ **Extensible** - Easy to add new extractors and models
- ✅ **Performance Metrics** - Comprehensive evaluation
- ✅ **Deep Learning Ready** - PyTorch neural network support

## 🚀 Advanced Usage

The modular design allows for easy experimentation:

```python
# Test different combinations
combinations = [
    (FeatureExtractorType.COUNT_VECTORIZER, ModelType.LOGISTIC_REGRESSION),
    (FeatureExtractorType.COUNT_VECTORIZER, ModelType.PYTORCH_NEURAL_NETWORK),
    (FeatureExtractorType.HUGGINGFACE_TRANSFORMER, ModelType.LOGISTIC_REGRESSION),
]

for extractor_type, model_type in combinations:
    main(feature_extractor_type=extractor_type, model_type=model_type)
```

This architecture makes it trivial to:
- **Compare different approaches** on the same dataset
- **Add new feature extractors** (TF-IDF, Word2Vec, BERT, etc.)
- **Add new models** (Random Forest, SVM, Transformer, etc.)
- **Experiment with hyperparameters**
- **Scale to production** with clean, maintainable code 