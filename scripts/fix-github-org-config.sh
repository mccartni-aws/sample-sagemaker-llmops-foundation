#!/bin/bash

# Script to fix GitHub organization configuration for Lambda functions
# This updates the TARGET_GITHUB_ORG environment variable in all Lambda functions

set -e

echo "=========================================="
echo "GitHub Organization Configuration Fix"
echo "=========================================="
echo ""

# Check if TARGET_GITHUB_ORG is provided
if [ -z "$1" ]; then
    echo "❌ ERROR: GitHub organization name is required!"
    echo ""
    echo "Usage: $0 <github-org-or-username>"
    echo ""
    echo "Examples:"
    echo "  $0 mccartni              # For personal account"
    echo "  $0 my-organization       # For organization"
    echo ""
    echo "💡 To find your GitHub username:"
    echo "   • Go to https://github.com/settings/profile"
    echo "   • Your username is shown at the top"
    echo ""
    echo "💡 To find your GitHub organization:"
    echo "   • Go to https://github.com/settings/organizations"
    echo "   • Use the organization name (not display name)"
    echo ""
    exit 1
fi

TARGET_ORG="$1"
REGION="${AWS_REGION:-us-west-1}"

echo "🔧 Configuration:"
echo "   Target GitHub Org/User: $TARGET_ORG"
echo "   AWS Region: $REGION"
echo ""

# Confirm with user
read -p "❓ Is this correct? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted by user"
    exit 1
fi

echo ""
echo "🔍 Finding Lambda functions..."

# Find all Lambda functions with names containing our keywords
LAMBDA_FUNCTIONS=$(aws lambda list-functions \
    --region "$REGION" \
    --query "Functions[?contains(FunctionName, 'CheckProjectStatus') || contains(FunctionName, 'CreateDeployRepo') || contains(FunctionName, 'SyncRepositories') || contains(FunctionName, 'ModelApprovalTrigger')].FunctionName" \
    --output text)

if [ -z "$LAMBDA_FUNCTIONS" ]; then
    echo "❌ No Lambda functions found. Make sure the CDK stack is deployed."
    exit 1
fi

echo "✅ Found Lambda functions:"
for func in $LAMBDA_FUNCTIONS; do
    echo "   • $func"
done
echo ""

# Update each Lambda function
echo "🔄 Updating Lambda functions..."
for func in $LAMBDA_FUNCTIONS; do
    echo "   Updating $func..."

    # Get current environment variables
    CURRENT_ENV=$(aws lambda get-function-configuration \
        --function-name "$func" \
        --region "$REGION" \
        --query 'Environment.Variables' \
        --output json)

    # Update TARGET_GITHUB_ORG
    UPDATED_ENV=$(echo "$CURRENT_ENV" | jq --arg org "$TARGET_ORG" '. + {TARGET_GITHUB_ORG: $org}')

    # Apply the update
    aws lambda update-function-configuration \
        --function-name "$func" \
        --region "$REGION" \
        --environment "Variables=$UPDATED_ENV" \
        --output text > /dev/null

    echo "   ✅ Updated $func"
done

echo ""
echo "=========================================="
echo "✅ Configuration Update Complete!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo "   1. The Lambda functions now use: $TARGET_ORG"
echo "   2. Try creating a new SageMaker Project"
echo "   3. The repositories will be created in: https://github.com/$TARGET_ORG"
echo ""
echo "💡 Note: If you want to make this permanent, set the environment"
echo "   variable before deploying:"
echo "   export TARGET_GITHUB_ORG=$TARGET_ORG"
echo "   make deploy"
echo ""
