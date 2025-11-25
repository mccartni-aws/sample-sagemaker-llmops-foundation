#!/bin/bash

# Check GitHub token permissions for organization access

set -e

echo "Checking GitHub Token Permissions..."
echo ""

# Get the token from Secrets Manager
SECRET_NAME="llmops-sm-github-token"
REGION="${AWS_REGION:-us-west-1}"

echo "Fetching token from Secrets Manager: $SECRET_NAME"
TOKEN=$(aws secretsmanager get-secret-value \
    --secret-id "$SECRET_NAME" \
    --region "$REGION" \
    --query 'SecretString' \
    --output text | jq -r '.token')

if [ -z "$TOKEN" ]; then
    echo "❌ Failed to retrieve token from Secrets Manager"
    exit 1
fi

echo "✅ Token retrieved"
echo ""

# Test organization access
ORG_NAME="mccartni-aws"
echo "Testing access to organization: $ORG_NAME"
echo ""

# Check if we can access the org
ORG_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -H "Authorization: token $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/orgs/$ORG_NAME")

HTTP_CODE=$(echo "$ORG_RESPONSE" | tail -n1)
BODY=$(echo "$ORG_RESPONSE" | sed '$d')

echo "HTTP Status Code: $HTTP_CODE"
echo ""

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Successfully accessed organization: $ORG_NAME"
    echo ""
    echo "Organization details:"
    echo "$BODY" | jq '{login, name, description, public_repos, total_private_repos}'
    echo ""

    # Now test if we can create repos
    echo "Testing repository creation permissions..."
    REPOS_RESPONSE=$(curl -s -w "\n%{http_code}" \
        -H "Authorization: token $TOKEN" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "https://api.github.com/orgs/$ORG_NAME/repos")

    REPOS_HTTP_CODE=$(echo "$REPOS_RESPONSE" | tail -n1)

    if [ "$REPOS_HTTP_CODE" = "200" ]; then
        echo "✅ Can list repositories in organization"

        # Check token scopes
        echo ""
        echo "Checking token scopes..."
        SCOPES=$(curl -s -I \
            -H "Authorization: token $TOKEN" \
            "https://api.github.com/user" | grep -i "x-oauth-scopes:" | cut -d' ' -f2-)

        echo "Token scopes: $SCOPES"
        echo ""

        if echo "$SCOPES" | grep -q "admin:org"; then
            echo "✅ Token has admin:org scope (can create repos)"
        elif echo "$SCOPES" | grep -q "repo"; then
            echo "⚠️  Token has repo scope but NOT admin:org"
            echo "   You may not be able to create repos in the organization"
        else
            echo "❌ Token missing required scopes"
        fi
    else
        echo "❌ Cannot list repositories (HTTP $REPOS_HTTP_CODE)"
        echo "Response: $(echo "$REPOS_RESPONSE" | sed '$d')"
    fi

elif [ "$HTTP_CODE" = "404" ]; then
    echo "❌ Organization not found or token doesn't have access"
    echo ""
    echo "Possible causes:"
    echo "1. Organization name is incorrect"
    echo "2. Token doesn't have access to this organization"
    echo "3. You're not a member of this organization"
    echo ""
    echo "Response:"
    echo "$BODY" | jq '.'

elif [ "$HTTP_CODE" = "401" ]; then
    echo "❌ Authentication failed - token is invalid or expired"
    echo ""
    echo "Response:"
    echo "$BODY" | jq '.'

else
    echo "❌ Unexpected error (HTTP $HTTP_CODE)"
    echo ""
    echo "Response:"
    echo "$BODY" | jq '.'
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""

if [ "$HTTP_CODE" = "404" ]; then
    echo "1. Verify you're a member of the organization:"
    echo "   https://github.com/orgs/$ORG_NAME/people"
    echo ""
    echo "2. Generate a new token with these scopes:"
    echo "   • repo (Full control of private repositories)"
    echo "   • workflow (Update GitHub Action workflows)"
    echo "   • admin:org (Full control of orgs)"
    echo ""
    echo "3. Update the token in Secrets Manager:"
    echo "   aws secretsmanager update-secret \\"
    echo "     --secret-id $SECRET_NAME \\"
    echo "     --secret-string '{\"token\":\"YOUR_NEW_TOKEN\"}' \\"
    echo "     --region $REGION"
fi
