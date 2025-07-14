#!/bin/bash
# Setup all service accounts for ML training deployment system

PROJECT_ID=${GCLOUD_PROJECT_ID:-"your-project-id"}

echo "🚀 Setting up complete ML training deployment service accounts..."
echo "📋 Project: $PROJECT_ID"
echo ""

# Check if project ID is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Error: Please set GCLOUD_PROJECT_ID environment variable"
    echo "   export GCLOUD_PROJECT_ID='your-actual-project-id'"
    exit 1
fi

echo "1️⃣ Setting up ML Training Service Account (for VMs)..."
./setup_service_account.sh
echo ""

echo "2️⃣ Setting up Deployment Service Account (for deploy.py)..."
./setup_deployment_sa.sh
echo ""

echo "📝 Creating .env configuration..."

# Create or update .env file
cat > .env << EOF
# GCP Project Configuration
GCLOUD_PROJECT_ID=${PROJECT_ID}

# Service Accounts
ML_SERVICE_ACCOUNT=ml-training-sa@${PROJECT_ID}.iam.gserviceaccount.com
DEPLOYMENT_SERVICE_ACCOUNT=ml-deployment-sa@${PROJECT_ID}.iam.gserviceaccount.com

# VM Configuration  
DEFAULT_ZONE=us-central1-a
DEFAULT_MACHINE_TYPE=n1-standard-1
DEFAULT_PREEMPTIBLE=true
DEFAULT_AUTO_SHUTDOWN=true

# ML Training Configuration
ML_BUCKET_NAME=${ML_BUCKET_NAME:-"${PROJECT_ID}-ml-bucket"}
REPO_URL=https://github.com/yourusername/repo.git
REPO_BRANCH=main

# Authentication (uncomment to use service account key)
# GOOGLE_APPLICATION_CREDENTIALS=./deployment-key.json
EOF

echo "🎉 Complete setup finished!"
echo ""
echo "📝 Summary:"
echo "   • ML Training SA: ml-training-sa@${PROJECT_ID}.iam.gserviceaccount.com"
echo "   • Deployment SA: ml-deployment-sa@${PROJECT_ID}.iam.gserviceaccount.com"
echo "   • Configuration: .env file created"
echo ""
echo "🚀 Ready to deploy:"
echo "   python deploy.py  # (uses .env configuration automatically)"
echo ""
echo "🔒 Security files created:"
echo "   • deployment-key.json (for running deploy.py)"
echo "   • ml-training-key.json (for local ML testing)"  
echo "   • .env (configuration - already in .gitignore)"
echo "   ⚠️  Keep these files secure!" 