#!/bin/bash
# Load environment variables from .env file

if [ ! -f .env ]; then
    echo "❌ .env file not found. Run ./setup_all_service_accounts.sh first"
    exit 1
fi

echo "🔧 Loading environment variables from .env..."

# Method 1: Export all variables (safest)
set -a
source .env
set +a

echo "✅ Environment loaded successfully!"
echo ""
echo "📋 Loaded variables:"
echo "   GCLOUD_PROJECT_ID: ${GCLOUD_PROJECT_ID}"
echo "   ML_SERVICE_ACCOUNT: ${ML_SERVICE_ACCOUNT}"
echo "   DEFAULT_ZONE: ${DEFAULT_ZONE}"
echo "   DEFAULT_MACHINE_TYPE: ${DEFAULT_MACHINE_TYPE}"
echo ""
echo "🚀 Ready to use:"
echo "   python deploy.py"
echo "   gcloud config set project \$GCLOUD_PROJECT_ID" 