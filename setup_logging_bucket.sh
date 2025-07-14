#!/bin/bash
# Setup logging bucket for ML training runs

PROJECT_ID=${GCLOUD_PROJECT_ID:-$(gcloud config get-value project)}
LOG_BUCKET="${PROJECT_ID}-ml-logs"

echo "📦 Setting up logging bucket: gs://$LOG_BUCKET"

# Create the bucket (regional for better performance/cost)
echo "Creating bucket..."
gsutil mb -p $PROJECT_ID -c STANDARD -l us-central1 gs://$LOG_BUCKET

# Set lifecycle policy to delete old logs after 90 days (optional cost saving)
echo "Setting lifecycle policy..."
cat > lifecycle.json << 'EOF'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 90}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://$LOG_BUCKET
rm lifecycle.json

# Enable versioning (keeps multiple versions of files)
echo "Enabling versioning..."
gsutil versioning set on gs://$LOG_BUCKET

# Set public access prevention (security)
echo "Setting security policies..."
gsutil pap set enforced gs://$LOG_BUCKET

# Grant access to ML training service account
echo "Granting access to ML training service account..."
gsutil iam ch serviceAccount:ml-training-sa@${PROJECT_ID}.iam.gserviceaccount.com:objectAdmin gs://$LOG_BUCKET

echo "✅ Logging bucket setup complete!"
echo ""
echo "📊 Bucket: gs://$LOG_BUCKET"
echo "🌍 Region: us-central1"
echo "🔄 Versioning: Enabled"
echo "🗑️  Lifecycle: Delete logs after 90 days"
echo "🔒 Security: Public access prevented"
echo ""
echo "🚀 Ready for training runs! Logs will be automatically saved to this bucket." 