#!/bin/bash
set -e

# Logging - redirect all output to log file and console
exec > >(tee -a /var/log/startup-script.log)
exec 2>&1

echo "Starting ML training setup at $(date)"

# Update and install dependencies
echo "Updating system and installing dependencies..."
apt-get update
apt-get install -y python3.9 python3.9-venv python3-pip git curl

# Create working directory
mkdir -p /opt/ml-training
cd /opt/ml-training

# Get metadata from gcloud
echo "Retrieving metadata..."
REPO_URL=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo-url" -H "Metadata-Flavor: Google")
BRANCH=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/branch" -H "Metadata-Flavor: Google" 2>/dev/null || echo "main")
AUTO_SHUTDOWN=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/auto-shutdown" -H "Metadata-Flavor: Google" 2>/dev/null || echo "true")

echo "Configuration:"
echo "  Repository: $REPO_URL"
echo "  Branch: $BRANCH"
echo "  Auto-shutdown: $AUTO_SHUTDOWN"

# Clone the repository
echo "Cloning repository..."
git clone -b $BRANCH $REPO_URL .

# Create virtual environment
echo "Setting up Python virtual environment..."
python3.9 -m venv venv
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
python --version
pip list

# Run main.py
echo "Starting main.py at $(date)"
python main.py

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "main.py completed successfully at $(date)"
else
    echo "main.py failed with exit code $EXIT_CODE at $(date)"
fi

# Auto-shutdown if requested
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "Auto-shutdown enabled. Shutting down in 60 seconds..."
    echo "You can SSH in now if you need to check anything: gcloud compute ssh <vm-name> --zone=<zone>"
    sleep 60
    shutdown -h now
fi

echo "Startup script completed at $(date)" 