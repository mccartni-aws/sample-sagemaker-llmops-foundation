#!/bin/bash

# LLMOps Platform Complete Redeploy Script
# This script will tear down all existing stacks and redeploy them cleanly

set -e  # Exit on any error

echo "🚀 LLMOps Platform Complete Redeploy Script"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "sm-cdk/app.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check AWS credentials
print_status "Checking AWS credentials..."
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    print_error "AWS credentials are not configured or have expired"
    print_warning "Please run: aws sso login --profile your-profile-name"
    print_warning "Or configure your credentials with: aws configure"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
print_success "AWS credentials valid for account: $ACCOUNT_ID"

# Navigate to CDK directory
cd sm-cdk

# Activate virtual environment
print_status "Activating Python virtual environment..."
if [ ! -d ".venv" ]; then
    print_warning "Virtual environment not found, creating one..."
    python3 -m venv .venv
fi

source .venv/bin/activate
print_success "Virtual environment activated"

# Install/update dependencies
print_status "Installing/updating CDK dependencies..."
pip install -e . > /dev/null 2>&1
print_success "Dependencies installed"

# List existing stacks
print_status "Checking existing CloudFormation stacks..."
EXISTING_STACKS=$(aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE UPDATE_ROLLBACK_COMPLETE --query 'StackSummaries[?starts_with(StackName, `LlmOpsSm`)].StackName' --output text)

if [ -n "$EXISTING_STACKS" ]; then
    print_warning "Found existing LLMOps stacks:"
    for stack in $EXISTING_STACKS; do
        echo "  - $stack"
    done

    print_status "Destroying existing stacks..."

    # Destroy stacks in reverse dependency order
    STACKS_TO_DESTROY="LlmOpsSm-Observability-dev LlmOpsSm-Infrastructure-dev LlmOpsSm-Domain-dev"

    for stack in $STACKS_TO_DESTROY; do
        if aws cloudformation describe-stacks --stack-name "$stack" > /dev/null 2>&1; then
            print_status "Destroying stack: $stack"
            cdk destroy "$stack" --force || {
                print_warning "CDK destroy failed for $stack, trying direct CloudFormation delete..."
                aws cloudformation delete-stack --stack-name "$stack"
                print_status "Waiting for stack deletion to complete..."
                aws cloudformation wait stack-delete-complete --stack-name "$stack" || {
                    print_error "Failed to delete stack $stack"
                    print_warning "You may need to manually delete this stack from the AWS Console"
                }
            }
            print_success "Stack $stack destroyed"
        else
            print_status "Stack $stack does not exist, skipping..."
        fi
    done
else
    print_status "No existing LLMOps stacks found"
fi

# Clean up CDK context and outputs
print_status "Cleaning up CDK context..."
rm -f cdk.context.json
rm -rf cdk.out/
print_success "CDK context cleaned"

# Bootstrap CDK (in case it's needed)
print_status "Ensuring CDK is bootstrapped..."
cdk bootstrap > /dev/null 2>&1 || print_warning "CDK bootstrap failed or already done"

# Build Lambda layers with dependencies
print_status "Building Lambda layers with Python dependencies..."
print_status "Building Python dependencies layer..."
cd layers/python-layer
rm -rf python/
mkdir -p python/lib/python3.10/site-packages
pip install -r requirements.txt -t python/lib/python3.10/site-packages/ --platform linux_x86_64 --only-binary=:all: || \
pip install -r requirements.txt -t python/lib/python3.10/site-packages/
cd ../..
print_success "Lambda layers built successfully"

# Synthesize to check for errors
print_status "Synthesizing CDK stacks to check for errors..."
cdk synth > /dev/null 2>&1
print_success "CDK synthesis successful"

# Deploy stacks in correct order
print_status "Starting fresh deployment..."

# Deploy Domain stack first
print_status "Deploying Domain stack (SageMaker Domain + Execution Role)..."
cdk deploy LlmOpsSm-Domain-dev --require-approval never
print_success "Domain stack deployed successfully"

# Wait a moment for resources to stabilize
print_status "Waiting 30 seconds for resources to stabilize..."
sleep 30

# Deploy Infrastructure stack
print_status "Deploying Infrastructure stack (Service Catalog + Lambda + Step Functions)..."
cdk deploy LlmOpsSm-Infrastructure-dev --require-approval never
print_success "Infrastructure stack deployed successfully"

# Deploy Observability stack
print_status "Deploying Observability stack (CloudWatch + Monitoring)..."
cdk deploy LlmOpsSm-Observability-dev --require-approval never
print_success "Observability stack deployed successfully"

# Get important outputs
print_status "Retrieving deployment outputs..."

DOMAIN_ID=$(aws cloudformation describe-stacks --stack-name LlmOpsSm-Domain-dev --query 'Stacks[0].Outputs[?OutputKey==`DomainId`].OutputValue' --output text 2>/dev/null || echo "Not found")
DOMAIN_URL=$(aws cloudformation describe-stacks --stack-name LlmOpsSm-Domain-dev --query 'Stacks[0].Outputs[?OutputKey==`DomainUrl`].OutputValue' --output text 2>/dev/null || echo "Not found")
PORTFOLIO_ID=$(aws cloudformation describe-stacks --stack-name LlmOpsSm-Infrastructure-dev --query 'Stacks[0].Outputs[?OutputKey==`ServiceCatalogPortfolioId`].OutputValue' --output text 2>/dev/null || echo "Not found")

echo ""
print_success "🎉 LLMOps Platform Deployment Complete!"
echo "=========================================="
echo ""
echo "📋 Deployment Summary:"
echo "  • SageMaker Domain ID: $DOMAIN_ID"
echo "  • SageMaker Studio URL: $DOMAIN_URL"
echo "  • Service Catalog Portfolio ID: $PORTFOLIO_ID"
echo ""
echo "🔍 Next Steps:"
echo "  1. Wait 5-10 minutes for AWS to propagate all changes"
echo "  2. Go to SageMaker Studio: $DOMAIN_URL"
echo "  3. Navigate to: Projects → Create Project"
echo "  4. Look for 'Organization Templates' section"
echo "  5. You should see 'ESG Benchmarking MLOps Project'"
echo ""
echo "🛠️  If templates don't appear, run:"
echo "     ./fix-sagemaker-visibility.sh"
echo ""
print_success "Deployment completed successfully! 🚀"

# Return to original directory
cd ..
