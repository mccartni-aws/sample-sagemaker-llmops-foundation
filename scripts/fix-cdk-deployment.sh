#!/bin/bash

# Fix CDK Deployment Script
# This script clears CDK cache and deploys to the correct account

set -e

echo "🔧 CDK Deployment Fix Script"
echo "================================"

# Check if we're in the right directory
if [ ! -f "sm-cdk/app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Step 1: Verify AWS credentials
echo "📋 Step 1: Verifying AWS credentials..."
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ Error: AWS credentials are not valid or expired"
    echo ""
    echo "Please set up your AWS credentials first:"
    echo "  export AWS_ACCESS_KEY_ID=your_access_key"
    echo "  export AWS_SECRET_ACCESS_KEY=your_secret_key"
    echo "  export AWS_SESSION_TOKEN=your_session_token"
    echo ""
    echo "Or use AWS SSO:"
    echo "  aws sso login --profile your-profile"
    echo "  export AWS_PROFILE=your-profile"
    exit 1
fi

# Get current account
CURRENT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "✅ Current AWS Account: $CURRENT_ACCOUNT"

# Display current account for verification
echo "ℹ️  Deploying to account: $CURRENT_ACCOUNT"
read -p "Is this the correct account? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled. Please switch to the correct AWS account."
    exit 1
fi

# Step 2: Clear CDK cache
echo ""
echo "🧹 Step 2: Clearing CDK cache..."
cd sm-cdk

# Remove CDK cache files
rm -rf cdk.out/
rm -f cdk.context.json
echo "✅ CDK cache cleared"

# Step 3: Set environment variables
echo ""
echo "🌍 Step 3: Setting environment variables..."
export CDK_DEFAULT_ACCOUNT=$CURRENT_ACCOUNT
export CDK_DEFAULT_REGION=us-east-1
echo "✅ Environment variables set:"
echo "   CDK_DEFAULT_ACCOUNT=$CDK_DEFAULT_ACCOUNT"
echo "   CDK_DEFAULT_REGION=$CDK_DEFAULT_REGION"

# Step 4: Activate virtual environment
echo ""
echo "🐍 Step 4: Activating virtual environment..."
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found. Please run 'make setup' first."
    exit 1
fi
source .venv/bin/activate
echo "✅ Virtual environment activated"

# Step 5: Bootstrap CDK (if needed)
echo ""
echo "🚀 Step 5: Checking CDK bootstrap..."
if ! cdk bootstrap aws://$CURRENT_ACCOUNT/us-east-1 --no-previous-parameters 2>/dev/null; then
    echo "⚠️  Bootstrap may have failed, but continuing with deployment..."
fi

# Step 6: Synthesize CDK
echo ""
echo "🔨 Step 6: Synthesizing CDK stack..."
if ! cdk synth; then
    echo "❌ Error: CDK synthesis failed"
    exit 1
fi
echo "✅ CDK synthesis successful"

# Step 7: Deploy
echo ""
echo "🚀 Step 7: Deploying CDK stack..."
echo "This will deploy all 3 stacks: Domain, Infrastructure, and Observability"
echo ""
read -p "Do you want to proceed with deployment? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cdk deploy --all --require-approval never
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Check SageMaker Studio for the new organization template"
    echo "2. Update GitHub token in AWS Secrets Manager"
    echo "3. Verify Service Catalog portfolio is visible"
else
    echo "Deployment cancelled by user"
fi

cd ..
echo ""
echo "✅ Script completed"
