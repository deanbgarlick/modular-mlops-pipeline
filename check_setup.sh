#!/bin/bash

# ========================================
# Setup validation script
# ========================================

echo "Checking ML Training VM Deployment Setup"
echo "========================================"

# Check if required files exist
echo "1. Checking required files..."
files=("deploy.sh" "startup-script.sh" "requirements.txt" "main.py")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
    fi
done

# Check if scripts are executable
echo ""
echo "2. Checking script permissions..."
if [ -x "deploy.sh" ]; then
    echo "✓ deploy.sh is executable"
else
    echo "✗ deploy.sh not executable - run: chmod +x deploy.sh"
fi

# Check gcloud authentication
echo ""
echo "3. Checking gcloud authentication..."
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "✓ gcloud authenticated"
    echo "Active account: $(gcloud auth list --filter=status:ACTIVE --format="value(account)")"
else
    echo "✗ gcloud not authenticated - run: gcloud auth login"
fi

# Check current project
echo ""
echo "4. Checking current project..."
current_project=$(gcloud config get-value project 2>/dev/null)
if [ -n "$current_project" ]; then
    echo "✓ Current project: $current_project"
else
    echo "✗ No project set - run: gcloud config set project YOUR-PROJECT-ID"
fi

# Check if Compute Engine API is enabled
echo ""
echo "5. Checking Compute Engine API..."
if gcloud services list --enabled --filter="name:compute.googleapis.com" --format="value(name)" | grep -q .; then
    echo "✓ Compute Engine API is enabled"
else
    echo "✗ Compute Engine API not enabled - run: gcloud services enable compute.googleapis.com"
fi

# Check configuration in deploy.sh
echo ""
echo "6. Checking deploy.sh configuration..."
if grep -q "your-project-id" deploy.sh; then
    echo "✗ Update PROJECT_ID in deploy.sh"
else
    echo "✓ PROJECT_ID appears to be configured"
fi

if grep -q "yourusername" deploy.sh; then
    echo "✗ Update REPO_URL in deploy.sh"
else
    echo "✓ REPO_URL appears to be configured"
fi

# Check if main.py exists and imports look correct
echo ""
echo "7. Checking main.py..."
if [ -f "main.py" ]; then
    if grep -q "from SupervisedModel" main.py && grep -q "from DataLoader" main.py; then
        echo "✓ main.py has expected imports"
    else
        echo "⚠ main.py may be missing some imports"
    fi
else
    echo "✗ main.py not found"
fi

# Summary
echo ""
echo "Setup Summary:"
echo "=============="
echo "Before deploying, make sure to:"
echo "1. Update PROJECT_ID, ZONE, and REPO_URL in deploy.sh"
echo "2. Push your code to the repository specified in REPO_URL"
echo "3. Ensure your repository is public or the VM has access"
echo "4. Test locally that main.py runs without errors"
echo ""
echo "Then deploy with:"
echo "./deploy.sh deploy"
echo ""
echo "Monitor with:"
echo "./deploy.sh logs <vm-name>" 