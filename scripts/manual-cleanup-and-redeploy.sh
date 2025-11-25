#!/bin/bash

# Manual cleanup and redeploy script for Service Catalog issues
set -e

echo "🧹 Manual Service Catalog Cleanup and Redeploy"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check AWS credentials
print_status "Checking AWS credentials..."
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    print_error "AWS credentials are not configured or have expired"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
print_success "AWS credentials valid for account: $ACCOUNT_ID"

# Manual Service Catalog cleanup
print_status "Manually cleaning up Service Catalog resources..."

# Get portfolio ID
PORTFOLIO_ID=$(aws servicecatalog list-portfolios --query 'PortfolioDetails[?DisplayName==`ESG Benchmarking MLOps Templates`].Id' --output text 2>/dev/null || echo "")

if [ -n "$PORTFOLIO_ID" ]; then
    print_status "Found portfolio: $PORTFOLIO_ID"

    # List and disassociate products
    PRODUCTS=$(aws servicecatalog search-products-as-admin --portfolio-id "$PORTFOLIO_ID" --query 'ProductViewDetails[].ProductViewSummary.ProductId' --output text 2>/dev/null || echo "")

    if [ -n "$PRODUCTS" ]; then
        for product_id in $PRODUCTS; do
            print_status "Disassociating product: $product_id"
            aws servicecatalog disassociate-product-from-portfolio --portfolio-id "$PORTFOLIO_ID" --product-id "$product_id" 2>/dev/null || print_warning "Failed to disassociate product $product_id"
        done
    fi

    # List and disassociate tag options
    TAG_OPTIONS=$(aws servicecatalog list-tag-options-for-resource --resource-id "$PORTFOLIO_ID" --query 'TagOptionDetails[].Id' --output text 2>/dev/null || echo "")

    if [ -n "$TAG_OPTIONS" ]; then
        for tag_option_id in $TAG_OPTIONS; do
            print_status "Disassociating tag option: $tag_option_id"
            aws servicecatalog disassociate-tag-option-from-resource --resource-id "$PORTFOLIO_ID" --tag-option-id "$tag_option_id" 2>/dev/null || print_warning "Failed to disassociate tag option $tag_option_id"
        done
    fi

    # Delete portfolio
    print_status "Deleting portfolio: $PORTFOLIO_ID"
    aws servicecatalog delete-portfolio --id "$PORTFOLIO_ID" 2>/dev/null || print_warning "Failed to delete portfolio"
fi

# Clean up any orphaned tag options
print_status "Cleaning up orphaned tag options..."
TAG_OPTIONS=$(aws servicecatalog list-tag-options --query 'TagOptionDetails[?Key==`sagemaker:studio-visibility`].Id' --output text 2>/dev/null || echo "")

if [ -n "$TAG_OPTIONS" ]; then
    for tag_option_id in $TAG_OPTIONS; do
        print_status "Deleting tag option: $tag_option_id"
        aws servicecatalog delete-tag-option --id "$tag_option_id" 2>/dev/null || print_warning "Failed to delete tag option $tag_option_id"
    done
fi

# Force delete CloudFormation stacks
print_status "Force deleting any remaining CloudFormation stacks..."

STACKS_TO_DELETE="LlmOpsSm-Infrastructure-dev LlmOpsSm-Domain-dev LlmOpsSm-Observability-dev"

for stack in $STACKS_TO_DELETE; do
    if aws cloudformation describe-stacks --stack-name "$stack" > /dev/null 2>&1; then
        print_status "Force deleting stack: $stack"
        aws cloudformation delete-stack --stack-name "$stack"
        print_status "Waiting for stack deletion..."
        aws cloudformation wait stack-delete-complete --stack-name "$stack" 2>/dev/null || {
            print_warning "Stack $stack may still exist or be in a failed state"
            # Try to continue anyway
        }
    fi
done

print_success "Manual cleanup completed"

# Navigate to CDK directory
cd sm-cdk

# Activate virtual environment
source .venv/bin/activate

# Clean CDK context
print_status "Cleaning CDK context..."
rm -f cdk.context.json
rm -rf cdk.out/

# Bootstrap CDK
print_status "Bootstrapping CDK..."
cdk bootstrap > /dev/null 2>&1 || print_warning "CDK bootstrap may have failed"

# Deploy fresh stacks
print_status "Starting fresh deployment..."

# Deploy Domain stack
print_status "Deploying Domain stack..."
cdk deploy LlmOpsSm-Domain-dev --require-approval never
print_success "Domain stack deployed"

# Wait for stabilization
print_status "Waiting 30 seconds for resources to stabilize..."
sleep 30

# Deploy Infrastructure stack
print_status "Deploying Infrastructure stack..."
cdk deploy LlmOpsSm-Infrastructure-dev --require-approval never
print_success "Infrastructure stack deployed"

# Deploy Observability stack
print_status "Deploying Observability stack..."
cdk deploy LlmOpsSm-Observability-dev --require-approval never
print_success "Observability stack deployed"

# Get outputs
print_status "Retrieving deployment outputs..."

DOMAIN_ID=$(aws cloudformation describe-stacks --stack-name LlmOpsSm-Domain-dev --query 'Stacks[0].Outputs[?OutputKey==`DomainId`].OutputValue' --output text 2>/dev/null || echo "Not found")
DOMAIN_URL=$(aws cloudformation describe-stacks --stack-name LlmOpsSm-Domain-dev --query 'Stacks[0].Outputs[?OutputKey==`DomainUrl`].OutputValue' --output text 2>/dev/null || echo "Not found")
PORTFOLIO_ID=$(aws cloudformation describe-stacks --stack-name LlmOpsSm-Infrastructure-dev --query 'Stacks[0].Outputs[?OutputKey==`ServiceCatalogPortfolioId`].OutputValue' --output text 2>/dev/null || echo "Not found")

echo ""
print_success "🎉 LLMOps Platform Successfully Deployed!"
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
print_success "Deployment completed successfully! 🚀"

cd ..
