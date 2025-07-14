#!/bin/bash
set -e

# Logging - redirect all output to log file and console
exec > >(tee -a /var/log/startup-script.log)
exec 2>&1

echo "Starting ML training setup at $(date)"

# Update and install dependencies
echo "Updating system and installing dependencies..."
apt-get update
apt-get install -y python3.10 python3.10-venv python3-pip git curl

# Create python3 symlink for consistency
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create working directory
mkdir -p /opt/ml-training
cd /opt/ml-training

# Get metadata from gcloud
echo "Retrieving metadata..."
REPO_URL=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo-url" -H "Metadata-Flavor: Google")
BRANCH=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/branch" -H "Metadata-Flavor: Google" 2>/dev/null || echo "main")
AUTO_SHUTDOWN=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/auto-shutdown" -H "Metadata-Flavor: Google" 2>/dev/null || echo "true")
ML_MODE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/ml-mode" -H "Metadata-Flavor: Google" 2>/dev/null || echo "dispatcher")

echo "Configuration:"
echo "  Repository: $REPO_URL"
echo "  Branch: $BRANCH"
echo "  Auto-shutdown: $AUTO_SHUTDOWN"
echo "  ML Mode: $ML_MODE"

# Clone the repository
echo "Cloning repository..."
git clone -b $BRANCH $REPO_URL .

# Create virtual environment
echo "Setting up Python virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
if [ -f requirements.txt ]; then
    echo "Installing from requirements.txt"
    pip install -r requirements.txt
else
    echo "No requirements.txt found, installing common ML packages"
    pip install pandas numpy scikit-learn torch transformers sentence-transformers google-cloud-storage python-dotenv
fi

# Create .env file with basic GCP configuration
echo "Creating .env file..."
cat > .env << 'EOF'
# GCP Configuration
# Add your bucket names here or override in your code
GCP_BUCKET_NAME=your-ml-data-bucket
GCP_MODEL_BUCKET=your-ml-models-bucket
EOF

# Display Python and package versions for debugging
echo "Python environment info:"
echo "System Python: $(python3.10 --version)"
echo "Virtual env Python: $(python --version)"
pip list

# Determine which ML script to run based on mode
case "$ML_MODE" in
    "single")
        SCRIPT_TO_RUN="main_single_run.py"
        echo "🔄 Running single experiment mode"
        ;;
    "suite")
        SCRIPT_TO_RUN="main_experiment_run.py"
        echo "🔄 Running experiment suite mode"
        ;;
    "dispatcher"|*)
        SCRIPT_TO_RUN="main.py"
        echo "🔄 Running dispatcher mode (main.py)"
        ;;
esac

# Run the selected script
echo "Starting $SCRIPT_TO_RUN at $(date)"
python $SCRIPT_TO_RUN

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "$SCRIPT_TO_RUN completed successfully at $(date)"
else
    echo "$SCRIPT_TO_RUN failed with exit code $EXIT_CODE at $(date)"
fi

# Save logs to bucket before shutdown
echo "Saving logs and results..."

# Get VM name and create log directory
VM_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
PROJECT_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_BUCKET="${PROJECT_ID}-ml-logs"

# Check if bucket exists
if ! gsutil ls gs://$LOG_BUCKET/ >/dev/null 2>&1; then
    echo "Warning: Bucket gs://$LOG_BUCKET does not exist!"
    echo "Run './setup_logging_bucket.sh' before deploying VMs."
    echo "Continuing without bucket logging..."
    LOG_BUCKET=""
fi

# Upload logs and results if bucket exists
if [ -n "$LOG_BUCKET" ]; then
    echo "Uploading startup script logs..."
    gsutil cp /var/log/startup-script.log gs://$LOG_BUCKET/${VM_NAME}_${TIMESTAMP}_startup.log

    echo "Uploading training results..."
    if [ -f "training_results.json" ]; then
        gsutil cp training_results.json gs://$LOG_BUCKET/${VM_NAME}_${TIMESTAMP}_results.json
    fi

    if [ -f "model.pkl" ] || [ -f "model.pt" ]; then
        echo "Uploading trained models..."
        gsutil cp model.* gs://$LOG_BUCKET/${VM_NAME}_${TIMESTAMP}/ 2>/dev/null || true
    fi

    # Upload any other important files
    if [ -d "outputs" ]; then
        gsutil -m cp -r outputs gs://$LOG_BUCKET/${VM_NAME}_${TIMESTAMP}/ || true
    fi

    echo "Logs and results saved to: gs://$LOG_BUCKET/${VM_NAME}_${TIMESTAMP}/"
else
    echo "No logging bucket configured - logs remain on VM only"
fi

# Auto-shutdown if requested
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "Auto-shutdown enabled. Shutting down in 60 seconds..."
    echo "You can SSH in now if you need to check anything: gcloud compute ssh $VM_NAME --zone=us-central1-a"
    if [ -n "$LOG_BUCKET" ]; then
        echo "Logs saved to: gs://$LOG_BUCKET/${VM_NAME}_${TIMESTAMP}/"
    else
        echo "Logs available on VM only (no bucket configured)"
    fi
    sleep 60
    shutdown -h now
fi

echo "Startup script completed at $(date)" 