# Modular Text Classification Pipeline with Cloud Deployment

A production-ready, extensible machine learning pipeline for text classification with cloud deployment capabilities, model persistence, and comprehensive experiment management.

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

This architecture provides a complete MLOps solution with local development capabilities, cloud deployment automation, comprehensive monitoring, and production-ready security practices. 