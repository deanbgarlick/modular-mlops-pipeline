# Modular Text Classification Pipeline with Cloud Deployment

A production-ready, extensible machine learning pipeline for text classification with cloud deployment capabilities, model persistence, comprehensive experiment management, and **automated hyperparameter optimization**.

## 🚀 Quick Start

### Local Development
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run single experiment:**
   ```bash
   python main.py
   ```

3. **Run experiment suite:**
   ```bash
   # Edit main.py: set RUN_EXPERIMENTS = True
   python main.py
   ```

4. **Run hyperparameter optimization:**
   ```bash
   python test_hyperparameter_optimization.py
   ```

### Cloud Deployment (GCP)
1. **Setup service accounts:**
   ```bash
   ./setup_all_service_accounts.sh
   ```

2. **Deploy to GCP VM:**
   ```bash
   python deploy.py
   ```

3. **Monitor training:**
   ```bash
   python deploy.py --action stream --name your-vm-name
   ```

## 🏗️ Architecture

The project uses **Strategy Pattern** and **Factory Functions** with comprehensive cloud integration:

```
├── main.py                        # Pipeline orchestrator & experiment runner
├── deploy.py                      # GCP VM deployment system
├── DataLoader/                    # Data loading strategies
│   ├── newsgroups_loader.py      # 20 Newsgroups dataset
│   ├── csv_loader.py             # Local CSV files
│   └── gcp_csv_loader.py         # GCP Cloud Storage CSV files
├── FeatureExtractor/              # Feature extraction strategies
│   ├── count_vectorizer.py       # Count-based features
│   ├── tfidf_vectorizer.py       # TF-IDF features
│   └── huggingface.py           # Transformer embeddings
├── SupervisedModel/               # ML model implementations
│   ├── logistic_regression.py    # Sklearn logistic regression
│   ├── pytorch_neural_network.py # PyTorch neural networks
│   ├── knn_classifier.py         # K-Nearest Neighbors
│   └── persistence.py           # Model save/load system
├── Pipeline/                      # Pipeline orchestration
├── Experiments/                   # Experiment management
├── HyperparamPicker/              # Automated hyperparameter optimization
│   ├── core.py                   # Main optimization engine (Ax integration)
│   ├── config.py                 # Parameter space configuration
│   ├── results.py                # Results analysis and Pareto frontier
│   ├── metrics.py                # Custom Ax metrics
│   └── runner.py                 # Pipeline integration for trials
└── Infrastructure/               # Cloud deployment scripts
    ├── setup_service_account.sh  # ML training service account
    ├── setup_deployment_sa.sh    # Deployment service account
    ├── setup_logging_bucket.sh   # GCS logging infrastructure
    └── startup-script-local.sh   # VM initialization script
```

## 🔧 Feature Extractors

### Count Vectorizer
- **Type:** `FeatureExtractorType.COUNT_VECTORIZER`
- **Features:** Sparse count-based vectors (configurable max features)
- **Performance:** ~97% accuracy, very fast
- **Use case:** Traditional ML, interpretable features

### TF-IDF Vectorizer  
- **Type:** `FeatureExtractorType.TFIDF_VECTORIZER`
- **Features:** TF-IDF weighted sparse vectors
- **Performance:** ~98% accuracy, fast
- **Use case:** Document classification, keyword importance

### HuggingFace Transformers
- **Type:** `FeatureExtractorType.HUGGINGFACE_TRANSFORMER`
- **Features:** Dense transformer embeddings (384 dimensions)
- **Performance:** ~89% accuracy, semantic understanding
- **Use case:** Semantic similarity, modern NLP

## 🤖 Available Models

### Logistic Regression
- **Type:** `SupervisedModelType.LOGISTIC_REGRESSION`
- **Implementation:** Sklearn wrapper with class weighting
- **Performance:** ~97% accuracy with TF-IDF
- **Use case:** Fast, interpretable, reliable baseline

### PyTorch Neural Network
- **Type:** `SupervisedModelType.PYTORCH_NEURAL_NETWORK`
- **Implementation:** Configurable feedforward network
- **Performance:** ~96% accuracy with count vectorizer
- **Use case:** Deep learning, non-linear patterns

### K-Nearest Neighbors
- **Type:** `SupervisedModelType.KNN_CLASSIFIER`
- **Implementation:** Sklearn KNN with configurable parameters
- **Performance:** Variable based on feature space
- **Use case:** Non-parametric classification, simple baseline

## 💾 Model Persistence System

Comprehensive save/load system with multiple storage backends:

### Automatic Persistence Selection
```python
# Automatically selects appropriate persistence based on model type
model = create_supervised_model(SupervisedModelType.PYTORCH_NEURAL_NETWORK)
model.save("my_model")  # Automatically uses TorchGCPBucketPersistence

model = create_supervised_model(SupervisedModelType.LOGISTIC_REGRESSION) 
model.save("my_model")  # Automatically uses PickleGCPBucketPersistence
```

### Available Persistence Classes

#### Pickle-based (Sklearn models)
- `PickleGCPBucketPersistence` - Google Cloud Storage
- `PickleAWSBucketPersistence` - Amazon S3
- `PickleLocalFilePersistence` - Local filesystem

#### PyTorch-based (Neural networks)
- `TorchGCPBucketPersistence` - Google Cloud Storage
- `TorchAWSBucketPersistence` - Amazon S3  
- `TorchLocalFilePersistence` - Local filesystem

### Custom Persistence
```python
from SupervisedModel.persistence import TorchGCPBucketPersistence

# Custom persistence configuration
persistence = TorchGCPBucketPersistence(
    bucket_name="my-ml-models",
    prefix="production/models/"
)

model = create_supervised_model(
    SupervisedModelType.PYTORCH_NEURAL_NETWORK,
    persistence=persistence
)
```

## 📊 Data Sources

### 20 Newsgroups Dataset
- **Type:** `DataSourceType.NEWSGROUPS`
- **Built-in:** Uses sklearn's fetch_20newsgroups
- **Categories:** Configurable subset selection
- **Use case:** Standardized text classification benchmark

### Local CSV Files
- **Type:** `DataSourceType.CSV_FILE`
- **Format:** Configurable columns and separators
- **Use case:** Custom datasets, local development

### GCP Cloud Storage CSV
- **Type:** `DataSourceType.GCP_CSV_FILE`
- **Format:** CSV files stored in Google Cloud Storage
- **Authentication:** Service account or application default credentials
- **Use case:** Production datasets, shared storage

## ☁️ Cloud Deployment (GCP)

### Deployment Features
- **Automated VM creation** with proper networking and firewall rules
- **Service account management** with minimal required permissions
- **Real-time log streaming** during training
- **Automatic result storage** in Google Cloud Storage
- **Cost optimization** with preemptible instances
- **Auto-shutdown** after training completion

### Deployment Commands
```bash
# Basic deployment
python deploy.py

# Custom machine type
python deploy.py --machine-type n1-standard-4

# Monitor existing VM
python deploy.py --action monitor --name ml-training-123456

# Stream logs in real-time
python deploy.py --action stream --name ml-training-123456

# Get all logs
python deploy.py --action logs --name ml-training-123456
```

### Infrastructure Setup
```bash
# Setup all required service accounts and permissions
./setup_all_service_accounts.sh

# Setup logging infrastructure
./setup_logging_bucket.sh

# Retrieve training results
./get_logs.sh vm-name
```

## ⚖️ Class Weights for Imbalanced Data

Built-in support for handling class imbalance:

```python
# Enable automatic class weighting
results = run_pipeline(
    data_source_type=DataSourceType.NEWSGROUPS,
    feature_extractor_type=FeatureExtractorType.TFIDF_VECTORIZER,
    model_type=SupervisedModelType.LOGISTIC_REGRESSION,
    use_class_weights=True  # Handles imbalance automatically
)
```

## 🧪 Experiment Management

### Single Experiment
```python
# Configure in main.py
DATA_SOURCE = DataSourceType.NEWSGROUPS
FEATURE_EXTRACTOR = FeatureExtractorType.TFIDF_VECTORIZER  
MODEL = SupervisedModelType.LOGISTIC_REGRESSION
USE_CLASS_WEIGHTS = True

# Run
python main.py
```

### Experiment Suite
```python
# Set RUN_EXPERIMENTS = True in main.py
experiment_configs = [
    {
        "data_source_type": DataSourceType.NEWSGROUPS,
        "feature_extractor_type": FeatureExtractorType.COUNT_VECTORIZER,
        "model_type": SupervisedModelType.LOGISTIC_REGRESSION,
        "use_class_weights": True,
        "description": "Baseline: Count + LogReg"
    },
    {
        "data_source_type": DataSourceType.NEWSGROUPS,
        "feature_extractor_type": FeatureExtractorType.TFIDF_VECTORIZER,
        "model_type": SupervisedModelType.PYTORCH_NEURAL_NETWORK,
        "use_class_weights": True,
        "description": "Advanced: TF-IDF + PyTorch"
    }
]
```

## 🎯 Performance Results

| Feature Extractor | Model | Accuracy | F1-Score | Training Time | Notes |
|-------------------|-------|----------|----------|---------------|-------|
| TF-IDF Vectorizer | Logistic Regression | 98.1% | 98.0% | Fast ⚡ | Best overall |
| Count Vectorizer | Logistic Regression | 97.2% | 97.1% | Fast ⚡ | Good baseline |
| Count Vectorizer | PyTorch NN | 96.3% | 96.2% | Moderate 🔥 | Deep learning |
| HuggingFace | Logistic Regression | 88.9% | 88.6% | Slow 🐌 | Semantic features |

## 🔍 Automated Hyperparameter Optimization

### 🧠 Powered by Facebook's Ax Library

The system includes comprehensive hyperparameter optimization using [Facebook AI Research's Ax](https://ax.dev/) library, featuring:

- **Single & Multi-objective optimization**
- **Bayesian optimization** with intelligent search strategies
- **Parallel trial execution** for faster optimization
- **Pareto frontier analysis** for multi-objective problems
- **Automatic parameter space configuration**
- **Production-ready result analysis**

### 🎛️ Simple Usage

```python
from HyperparamPicker.factory import run_hyperparameter_optimization
from DataLoader import DataSourceType
from FeatureExtractor import FeatureExtractorType  
from SupervisedModel import SupervisedModelType

# Quick hyperparameter optimization
results = run_hyperparameter_optimization(
    data_source_type=DataSourceType.NEWSGROUPS,
    feature_extractor_type=FeatureExtractorType.TFIDF_VECTORIZER,
    model_type=SupervisedModelType.LOGISTIC_REGRESSION,
    loader_kwargs={"categories": ['alt.atheism', 'soc.religion.christian']},
    total_trials=20,
    max_parallel_trials=4,
    objectives=[("accuracy", False), ("training_time", True)]  # Maximize accuracy, minimize time
)

# Get best configurations
best_configs = results.get_recommended_configs()
```

### 🎯 Single-Objective Optimization

Optimize for a single metric (accuracy, F1-score, training time):

```python
from HyperparamPicker import create_hyperparam_picker, MultiObjectiveConfig

# Single objective: maximize accuracy
picker = create_hyperparam_picker(
    multi_objective_config=MultiObjectiveConfig(
        objectives=[("accuracy", False)]  # False = maximize
    )
)

results = picker.optimize(
    data_source_type=DataSourceType.NEWSGROUPS,
    feature_extractor_type=FeatureExtractorType.COUNT_VECTORIZER,
    model_type=SupervisedModelType.LOGISTIC_REGRESSION,
    loader_kwargs={},
    base_extractor_kwargs={},
    base_model_kwargs={},
    total_trials=30
)
```

### 🎯 Multi-Objective Optimization

Optimize multiple competing objectives simultaneously:

```python
# Multi-objective: maximize accuracy AND F1, minimize training time
picker = create_hyperparam_picker(
    multi_objective_config=MultiObjectiveConfig(
        objectives=[
            ("accuracy", False),      # Maximize accuracy
            ("f1_macro", False),      # Maximize F1-score  
            ("training_time", True)   # Minimize training time
        ],
        objective_thresholds={
            "accuracy": 0.80,         # Minimum acceptable accuracy
            "training_time": 300      # Maximum acceptable training time (seconds)
        }
    )
)

results = picker.optimize(
    # ... same parameters as above ...
    total_trials=50,
    max_parallel_trials=8
)

# Analyze Pareto frontier
print(f"Pareto frontier size: {len(results.pareto_frontier)}")
print(f"Best accuracy: {results.best_single_objective['accuracy']}")
print(f"Fastest training: {results.best_single_objective['training_time']}")
```

### ⚙️ Custom Parameter Spaces

Define custom hyperparameter search spaces:

```python
from HyperparamPicker import HyperparamSearchConfig, ParameterSpec
from ax.core import ParameterType

# Custom search configuration
search_config = HyperparamSearchConfig()

# TF-IDF parameters
search_config.feature_extractor_params[FeatureExtractorType.TFIDF_VECTORIZER] = {
    "max_features": ParameterSpec(
        name="max_features",
        param_type="choice",
        values=[1000, 5000, 10000, 20000, 50000],
        parameter_type=ParameterType.INT,
        is_ordered=True
    ),
    "min_df": ParameterSpec(
        name="min_df", 
        param_type="range",
        lower=1,
        upper=10,
        parameter_type=ParameterType.INT
    ),
    "max_df": ParameterSpec(
        name="max_df",
        param_type="range", 
        lower=0.5,
        upper=0.95,
        parameter_type=ParameterType.FLOAT
    )
}

# Logistic Regression parameters
search_config.model_params[SupervisedModelType.LOGISTIC_REGRESSION] = {
    "C": ParameterSpec(
        name="C",
        param_type="range",
        lower=0.001,
        upper=10.0,
        parameter_type=ParameterType.FLOAT,
        log_scale=True  # Logarithmic scale for regularization
    ),
    "max_iter": ParameterSpec(
        name="max_iter",
        param_type="choice", 
        values=[100, 200, 500, 1000, 2000],
        parameter_type=ParameterType.INT,
        is_ordered=True
    )
}

# Use custom configuration
picker = create_hyperparam_picker(search_config=search_config)
```

### 📊 Available Optimization Objectives

| Objective | Description | Optimization Direction |
|-----------|-------------|----------------------|
| `accuracy` | Classification accuracy | Maximize |
| `f1_macro` | Macro-averaged F1-score | Maximize |
| `f1_weighted` | Weighted F1-score | Maximize |
| `precision` | Precision score | Maximize |
| `recall` | Recall score | Maximize |
| `training_time` | Model training time (seconds) | Minimize |
| `prediction_time` | Average prediction time | Minimize |
| `model_size` | Model memory footprint | Minimize |

### 🔧 Supported Parameter Types

#### Range Parameters
```python
ParameterSpec(
    name="learning_rate",
    param_type="range",
    lower=0.001,
    upper=0.1,
    parameter_type=ParameterType.FLOAT,
    log_scale=True  # Use logarithmic scaling
)
```

#### Choice Parameters
```python
ParameterSpec(
    name="optimizer",
    param_type="choice", 
    values=["adam", "sgd", "adamw"],
    parameter_type=ParameterType.STRING
)
```

#### Fixed Parameters
```python
ParameterSpec(
    name="random_state",
    param_type="fixed",
    value=42,
    parameter_type=ParameterType.INT
)
```

### 📈 Results Analysis

```python
# Run optimization
results = picker.optimize(...)

# Optimization history
print(f"Total trials completed: {len(results.optimization_history)}")

# Best single-objective results
for objective, trial in results.best_single_objective.items():
    print(f"Best {objective}: Trial {trial.index}")

# Pareto frontier (multi-objective)
print(f"Pareto frontier: {len(results.pareto_frontier)} optimal solutions")

# Get recommendations based on preferences
preferences = {"accuracy": 0.7, "training_time": 0.3}
recommended = results.get_recommended_configs(preferences)

# Export results
results.export_results("hyperparameter_results.json")
results.save_visualization("pareto_plot.png")
```

### 🏃‍♂️ Default Parameter Spaces

The system comes with pre-configured parameter spaces for all models:

#### **TF-IDF Vectorizer**
- `max_features`: [1000, 5000, 10000, 20000, 50000]
- `min_df`: Range(1, 10)
- `max_df`: Range(0.5, 0.95)

#### **Count Vectorizer**  
- `max_features`: [1000, 5000, 10000, 20000, 50000]
- `min_df`: Range(1, 10)
- `max_df`: Range(0.5, 0.95)

#### **HuggingFace Transformers**
- `model_name`: ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]
- `batch_size`: [50, 100, 200]

#### **Logistic Regression**
- `C`: Range(0.001, 10.0, log_scale=True)
- `max_iter`: [100, 200, 500, 1000]

#### **PyTorch Neural Network**
- `hidden_size`: Range(64, 512)
- `learning_rate`: Range(0.0001, 0.1, log_scale=True)
- `epochs`: [25, 50, 100]
- `batch_size`: [32, 64, 128]

#### **K-Nearest Neighbors**
- `n_neighbors`: Range(3, 20)
- `weights`: ["uniform", "distance"]

### 🚀 Production Hyperparameter Optimization

Combine with cloud deployment for large-scale optimization:

```python
# Local hyperparameter search
results = run_hyperparameter_optimization(
    data_source_type=DataSourceType.GCP_CSV_FILE,
    feature_extractor_type=FeatureExtractorType.TFIDF_VECTORIZER,
    model_type=SupervisedModelType.LOGISTIC_REGRESSION,
    loader_kwargs={"bucket_name": "my-data", "file_path": "training_data.csv"},
    total_trials=100,
    max_parallel_trials=16,  # Parallel optimization
    objectives=[("accuracy", False), ("f1_macro", False), ("training_time", True)]
)

# Deploy best configuration to GCP VM
best_config = results.get_recommended_configs()[0]
# Use best_config parameters in deploy.py
```

The hyperparameter optimization system provides production-ready automated ML tuning with minimal configuration, supporting both single and multi-objective optimization scenarios.

## 🔌 Extensibility

### Adding New Feature Extractors
1. Create class inheriting from `FeatureExtractor`
2. Add enum value to `FeatureExtractorType`
3. Update factory function
4. Implement required methods: `fit_transform()`, `transform()`, `get_feature_info()`

### Adding New Models
1. Create class inheriting from `SupervisedModel`
2. Add enum value to `SupervisedModelType`
3. Update factory function
4. Implement required methods: `fit()`, `predict()`, `predict_proba()`, `get_model_info()`

### Adding New Data Sources
1. Create class inheriting from `DataLoader`
2. Add enum value to `DataSourceType`
3. Update factory function
4. Implement required methods: `load_data()`, `get_loader_info()`

## 🛠️ Environment Configuration

Create `.env` file for deployment configuration:
```bash
# GCP Configuration
GCLOUD_PROJECT_ID=your-project-id
GCLOUD_REGION=us-central1
GCLOUD_ZONE=us-central1-a

# Service Accounts
ML_SERVICE_ACCOUNT=ml-training@your-project-id.iam.gserviceaccount.com
DEPLOYMENT_SERVICE_ACCOUNT=ml-deployment@your-project-id.iam.gserviceaccount.com

# Authentication (for local deployment)
GOOGLE_APPLICATION_CREDENTIALS=deployment-key.json
```

## 🚀 Production Deployment Workflow

1. **Local Development:**
   ```bash
   pip install -r requirements.txt
   python main.py  # Test pipeline locally
   ```

2. **Setup Cloud Infrastructure:**
   ```bash
   ./setup_all_service_accounts.sh
   ./setup_logging_bucket.sh
   ```

3. **Deploy Training:**
   ```bash
   python deploy.py --machine-type n1-standard-2
   ```

4. **Monitor & Retrieve Results:**
   ```bash
   python deploy.py --action stream --name your-vm-name
   ./get_logs.sh your-vm-name
   ```

## 🛡️ Security & Best Practices

- **Service Account Isolation:** Separate accounts for deployment and training
- **Minimal Permissions:** Each service account has only required permissions
- **Secure Key Management:** JSON keys stored locally, not in repository
- **Network Security:** VMs use private IPs with Cloud NAT for outbound access
- **Automatic Cleanup:** VMs auto-terminate after training completion
- **Cost Control:** Preemptible instances enabled by default

## 🔍 Monitoring & Debugging

- **Real-time logs:** `python deploy.py --action stream --name vm-name`
- **VM status:** `python deploy.py --action monitor --name vm-name`
- **Training results:** Automatically saved to GCS bucket
- **Error debugging:** Complete startup logs preserved in cloud storage
- **Resource monitoring:** GCP Console for VM metrics and costs
- **Hyperparameter optimization:** Bayesian optimization with Pareto frontier analysis

## 🛠️ Key Features

- ✅ **Modular Design** - Mix and match any feature extractor with any model
- ✅ **Type Safety** - Enum-based configuration prevents errors
- ✅ **Clean Interface** - Consistent API across all components
- ✅ **Factory Functions** - Centralized object creation
- ✅ **Package Organization** - Clean separation of concerns
- ✅ **Extensible** - Easy to add new extractors and models
- ✅ **Performance Metrics** - Comprehensive evaluation
- ✅ **Deep Learning Ready** - PyTorch neural network support
- ✅ **Imbalance Handling** - Automatic class weighting for better F1-scores
- ✅ **Model Persistence** - Save/load models across multiple storage backends
- ✅ **Cloud Deployment** - Automated GCP VM deployment with monitoring
- ✅ **Hyperparameter Optimization** - Automated Bayesian optimization with Ax
- ✅ **Multi-objective Optimization** - Pareto frontier analysis for competing objectives
- ✅ **Production Ready** - Complete MLOps pipeline with security and monitoring

This architecture provides a complete MLOps solution with local development capabilities, cloud deployment automation, comprehensive monitoring, automated hyperparameter optimization, and production-ready security practices. 