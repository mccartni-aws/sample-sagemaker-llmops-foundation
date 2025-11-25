#!/bin/bash

# Script to fix AWS credentials configuration in GitHub workflow files
# This script helps identify and fix the missing role-to-assume parameter

set -e

echo "=================================================="
echo "GitHub Workflow AWS Credentials Fix Script"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}This script will help you fix the AWS credentials configuration in your GitHub workflow.${NC}"
echo ""
echo "The error you're seeing:"
echo "  'Credentials could not be loaded, please check your action inputs'"
echo ""
echo "Is caused by a missing 'role-to-assume' parameter in the workflow file."
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

echo "=================================================="
echo "Step 1: Verify Seed Code Templates"
echo "=================================================="
echo ""

# Check seed-code workflow files
BUILD_WORKFLOW="seed-code/esg-benchmarking/model_build/.github/workflows/build_sagemaker_pipeline.yml"
DEPLOY_WORKFLOW="seed-code/esg-benchmarking/model_deploy/.github/workflows/deploy_endpoint.yml"

if [ -f "$BUILD_WORKFLOW" ]; then
    echo -e "${GREEN}✓${NC} Found build workflow template: $BUILD_WORKFLOW"

    # Check if it has the correct configuration
    if grep -q "role-to-assume:" "$BUILD_WORKFLOW"; then
        echo -e "${GREEN}✓${NC} Build workflow has correct OIDC configuration"
    else
        echo -e "${RED}✗${NC} Build workflow is missing role-to-assume parameter"
    fi
else
    echo -e "${RED}✗${NC} Build workflow template not found"
fi

echo ""

if [ -f "$DEPLOY_WORKFLOW" ]; then
    echo -e "${GREEN}✓${NC} Found deploy workflow template: $DEPLOY_WORKFLOW"

    # Check if it has the correct configuration
    if grep -q "role-to-assume:" "$DEPLOY_WORKFLOW"; then
        echo -e "${GREEN}✓${NC} Deploy workflow has correct OIDC configuration"
    else
        echo -e "${RED}✗${NC} Deploy workflow is missing role-to-assume parameter"
    fi
else
    echo -e "${RED}✗${NC} Deploy workflow template not found"
fi

echo ""
echo "=================================================="
echo "Step 2: Instructions to Fix Your GitHub Repository"
echo "=================================================="
echo ""

echo -e "${YELLOW}Your GitHub repository workflow needs to be updated manually.${NC}"
echo ""
echo "Follow these steps:"
echo ""
echo "1. Go to your GitHub repository: https://github.com/<YOUR_ORG>/test4-build-sjbspc"
echo ""
echo "2. Navigate to: .github/workflows/build_sagemaker_pipeline.yml"
echo ""
echo "3. Click 'Edit' and find the 'Configure AWS credentials' step"
echo ""
echo "4. Replace the INCORRECT configuration:"
echo ""
echo "   - name: Configure AWS credentials"
echo "     uses: aws-actions/configure-aws-credentials@v4"
echo "     with:"
echo "       aws-region: us-west-1"
echo "       audience: sts.amazonaws.com"
echo "       output-env-credentials: true"
echo ""
echo "5. With the CORRECT configuration:"
echo ""
echo "   - name: Configure AWS credentials"
echo "     uses: aws-actions/configure-aws-credentials@e3dd6a429d7300a6a4c196c26e071d42e0343502 # v4.0.2"
echo "     with:"
echo "       role-to-assume: \${{ vars.OIDC_ROLE_GITHUB_WORKFLOW_BUILD }}"
echo "       aws-region: \${{ vars.REGION }}"
echo ""
echo "6. Commit the changes"
echo ""

echo "=================================================="
echo "Step 3: Verify GitHub Repository Variables"
echo "=================================================="
echo ""

echo "Ensure these variables are set in your GitHub repository:"
echo ""
echo "Go to: Repository Settings → Secrets and variables → Actions → Variables"
echo ""
echo "Required variables:"
echo "  • OIDC_ROLE_GITHUB_WORKFLOW_BUILD - IAM role ARN for build workflow"
echo "  • REGION - AWS region (e.g., us-west-1)"
echo "  • SAGEMAKER_PROJECT_NAME - Your SageMaker project name"
echo "  • SAGEMAKER_PROJECT_ID - Your SageMaker project ID"
echo "  • SAGEMAKER_PIPELINE_ROLE_ARN - SageMaker pipeline execution role ARN"
echo "  • ARTIFACT_BUCKET - S3 bucket for artifacts"
echo "  • MODEL_PACKAGE_GROUP_NAME - Model package group name"
echo "  • INPUT_DATA_PATH - S3 path to input data"
echo "  • MLFLOW_TRACKING_ARN - MLflow tracking server ARN"
echo "  • TRIGGER_PIPELINE_EXECUTION - Set to 'true' to enable"
echo ""

echo "=================================================="
echo "Step 4: Verify IAM Role Trust Policy"
echo "=================================================="
echo ""

echo "Your IAM role must have a trust policy that allows GitHub OIDC:"
echo ""
echo "{"
echo "  \"Version\": \"2012-10-17\","
echo "  \"Statement\": ["
echo "    {"
echo "      \"Effect\": \"Allow\","
echo "      \"Principal\": {"
echo "        \"Federated\": \"arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com\""
echo "      },"
echo "      \"Action\": \"sts:AssumeRoleWithWebIdentity\","
echo "      \"Condition\": {"
echo "        \"StringEquals\": {"
echo "          \"token.actions.githubusercontent.com:aud\": \"sts.amazonaws.com\""
echo "        },"
echo "        \"StringLike\": {"
echo "          \"token.actions.githubusercontent.com:sub\": \"repo:<YOUR_ORG>/<YOUR_REPO>:*\""
echo "        }"
echo "      }"
echo "    }"
echo "  ]"
echo "}"
echo ""

echo "=================================================="
echo "Documentation"
echo "=================================================="
echo ""
echo "For detailed information, see:"
echo "  • docs/QUICK_FIX_AWS_CREDENTIALS.md"
echo "  • docs/GITHUB_WORKFLOW_AWS_CREDENTIALS_FIX.md"
echo ""
echo -e "${GREEN}Script completed!${NC}"
