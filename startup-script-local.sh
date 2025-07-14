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
AUTO_SHUTDOWN=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/auto-shutdown" -H "Metadata-Flavor: Google" 2>/dev/null || echo "true")

echo "Configuration:"
echo "  Auto-shutdown: $AUTO_SHUTDOWN"
echo "  Working directory: $(pwd)"

# Create a simple main.py for testing
echo "Creating test training script..."
cat > main.py << 'EOF'
#!/usr/bin/env python3
"""Simple ML Training Test Script"""

import time
import json
import os
from datetime import datetime

print("🚀 Starting ML Training Test")
print(f"📅 Start time: {datetime.now()}")
print(f"📁 Working directory: {os.getcwd()}")

# Simulate training for 30 seconds
print("🔄 Training in progress...")
for i in range(6):
    print(f"   Epoch {i+1}/6 - Loss: {1.0 - i*0.15:.3f}")
    time.sleep(5)

# Create training results
results = {
    "training_completed": True,
    "final_loss": 0.125,
    "epochs": 6,
    "duration_seconds": 30,
    "timestamp": datetime.now().isoformat()
}

print(f"💾 Saving results: {results}")
with open("training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Training completed successfully!")
print(f"📅 End time: {datetime.now()}")
EOF

# Create virtual environment
echo "Setting up Python virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install basic dependencies
echo "Installing Python dependencies..."
pip install google-cloud-storage python-dotenv

# Create .env file with basic GCP configuration
echo "Creating .env file..."
cat > .env << 'EOF'
# GCP Configuration
GCP_BUCKET_NAME=your-ml-data-bucket
GCP_MODEL_BUCKET=your-ml-models-bucket
EOF

# Display Python and package versions for debugging
echo "Python environment info:"
echo "System Python: $(python3.10 --version)"
echo "Virtual env Python: $(python --version)"
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