#!/bin/bash
# Retrieve training logs and results from GCS bucket

PROJECT_ID=${GCLOUD_PROJECT_ID:-$(gcloud config get-value project)}
LOG_BUCKET="${PROJECT_ID}-ml-logs"

echo "🔍 Searching for training logs in gs://$LOG_BUCKET/"

# Check if bucket exists
if ! gsutil ls gs://$LOG_BUCKET/ >/dev/null 2>&1; then
    echo "❌ Logging bucket gs://$LOG_BUCKET does not exist!"
    echo "🔧 Run './setup_logging_bucket.sh' to create it first."
    exit 1
fi

if [ "$1" ]; then
    # Specific VM name provided
    VM_NAME=$1
    echo "📋 Looking for logs from VM: $VM_NAME"
    gsutil ls gs://$LOG_BUCKET/ | grep $VM_NAME || echo "No logs found for VM: $VM_NAME"
else
    # List all logs
    echo "📋 All available training logs:"
    gsutil ls gs://$LOG_BUCKET/ || echo "No logs found in bucket"
fi

echo ""
echo "💡 Usage examples:"
echo "  ./get_logs.sh                    # List all logs"
echo "  ./get_logs.sh my-vm-name         # List logs for specific VM"
echo ""
echo "📥 To download logs:"
echo "  gsutil cp gs://$LOG_BUCKET/vm-name_timestamp_startup.log ."
echo "  gsutil cp gs://$LOG_BUCKET/vm-name_timestamp_results.json ."
echo ""
echo "📁 To download everything from a training run:"
echo "  gsutil -m cp -r gs://$LOG_BUCKET/vm-name_timestamp/ ./training_output/" 