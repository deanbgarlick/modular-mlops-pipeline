#!/bin/bash
# Setup service account for ML training VM with bucket access

PROJECT_ID=${GCLOUD_PROJECT_ID:-"your-project-id"}
SA_NAME="ml-training-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET_NAME=${ML_BUCKET_NAME:-"your-ml-bucket"}

echo "🔧 Setting up service account for ML training..."

# Create service account
gcloud iam service-accounts create $SA_NAME \
    --display-name="ML Training Service Account" \
    --description="Service account for ML training VMs to access buckets"

# Grant bucket permissions
echo "📦 Granting bucket permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/storage.objectAdmin"

# Optional: Grant logging permissions (for better monitoring)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/logging.logWriter"

# Create and download key (optional, for local testing)
echo "🔑 Creating service account key..."
gcloud iam service-accounts keys create ml-training-key.json \
    --iam-account=$SA_EMAIL

echo "✅ Service account setup complete!"
echo "📧 Service Account Email: $SA_EMAIL"
echo "🪣 Bucket Access: Granted for project buckets"
echo ""
echo "To use with VM deployment:"
echo "  export ML_SERVICE_ACCOUNT='$SA_EMAIL'"
echo ""
echo "To test locally:"
echo "  export GOOGLE_APPLICATION_CREDENTIALS='./ml-training-key.json'" 