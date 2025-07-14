#!/bin/bash
# Setup deployment service account for running deploy.py

PROJECT_ID=${GCLOUD_PROJECT_ID:-"your-project-id"}
SA_NAME="ml-deployment-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "🚀 Setting up deployment service account for deploy.py..."

# Create deployment service account
gcloud iam service-accounts create $SA_NAME \
    --display-name="ML Deployment Service Account" \
    --description="Service account for running deploy.py to manage VMs"

echo "🔧 Granting compute permissions..."

# Grant compute instance management permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/compute.instanceAdmin.v1"

# Grant permission to read/write compute images and disks
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/compute.storageAdmin"

# Grant service account user role (to assign service accounts to VMs)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/iam.serviceAccountUser"

# Grant monitoring permissions (for logs)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/logging.viewer"

# Optional: Grant project viewer (for general project access)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/viewer"

echo "🔑 Creating service account key..."
# Create and download key for local/CI usage
gcloud iam service-accounts keys create deployment-key.json \
    --iam-account=$SA_EMAIL

echo "✅ Deployment service account setup complete!"
echo "📧 Deployment SA Email: $SA_EMAIL"
echo "🖥️  Permissions: Compute instance management, service account assignment"
echo ""
echo "To use this service account:"
echo "  export GOOGLE_APPLICATION_CREDENTIALS='./deployment-key.json'"
echo "  python deploy.py"
echo ""
echo "For CI/CD pipelines:"
echo "  # Upload deployment-key.json as secret"
echo "  # Set GOOGLE_APPLICATION_CREDENTIALS environment variable"
echo ""
echo "⚠️  Security Note:"
echo "  - Keep deployment-key.json secure and private"
echo "  - Consider using Workload Identity in GKE/Cloud Run instead"
echo "  - Rotate keys regularly" 