# SageMaker LLMOps Platform Infrastructure

This directory contains the AWS CDK infrastructure code for the SageMaker LLMOps platform. It provides a complete MLOps automation framework that creates GitHub repositories with LLMOps templates when SageMaker projects are created, orchestrating the complete lifecycle from model fine-tuning to automated endpoint deployment.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Post-Deployment Setup](#post-deployment-setup)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Clean-up](#clean-up)

## Overview

The SageMaker LLMOps platform implements a **three-stack CDK architecture** that automates the complete LLM fine-tuning lifecycle through an event-driven workflow:

1. **SageMaker Domain Stack** - Creates SageMaker Studio Domain, execution roles, and S3 buckets
2. **Infrastructure Stack** - Creates EventBridge rules, Lambda functions, Step Functions, and GitHub integration
3. **Observability Stack** - Creates MLflow tracking server, monitoring dashboards, and notification systems

## Platform vs. Template Architecture

This CDK application deploys a **two-layer LLMOps platform**:

### Layer 1: Platform Infrastructure (This CDK App)

**What it creates:**

- SageMaker Studio Domain (shared development environment)
- EventBridge automation (detects project creation, model approval)
- Lambda functions (GitHub repository management)
- Step Functions (orchestration workflows)
- MLflow tracking server (centralized experiment tracking)
- GitHub OIDC integration (secure CI/CD authentication)

**When to redeploy:**

- Adding new Lambda functions
- Changing EventBridge rules
- Updating MLflow configuration
- Modifying GitHub integration

### Layer 2: Project Templates (Service Catalog)

**What they create (per project):**

- Model Package Group (for model versioning)
- Project-specific S3 buckets
- GitHub repositories (via EventBridge → Lambda)
- Seed code (training and deployment pipelines)

**When to update:**

- Adding new use case templates
- Modifying seed code structure
- Changing default configurations

### Why This Split?

**CloudFormation Template (Minimal):**

- Creates AWS resources only (Model Package Group, S3 buckets)
- Registered in Service Catalog
- Instantiated when user creates SageMaker Project

**Step Functions + Lambda (Heavy Lifting):**

- Triggered by EventBridge when project is created
- Handles GitHub API interactions (not possible in CloudFormation)
- Creates repositories, populates seed code, configures CI/CD
- Provides retry logic and error handling

This architecture allows:

- ✅ Multiple project templates without platform changes
- ✅ Complex GitHub integration beyond CloudFormation capabilities
- ✅ Async processing with automatic retries
- ✅ Easy extension to other Git providers (GitLab, Bitbucket)

## Architecture

### Three-Stack Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SageMaker LLMOps Platform                    │
├─────────────────────────────────────────────────────────────────┤
│  🏗️  SageMaker Domain Stack (LlmOpsSm-Domain-{env})            │
│  ├── SageMaker Studio Domain                                   │
│  ├── SageMaker Execution Roles                                 │
│  ├── S3 Buckets (artifacts, models)                           │
│  └── Model Registry Package Groups                             │
├─────────────────────────────────────────────────────────────────┤
│  ⚙️  Infrastructure Stack (LlmOpsSm-Infrastructure-{env})       │
│  ├── EventBridge Rules (model approval events)                 │
│  ├── Lambda Functions (repository management)                  │
│  ├── Step Functions (workflow orchestration)                   │
│  ├── GitHub OIDC Provider                                      │
│  └── Service Catalog (project templates)                       │
├─────────────────────────────────────────────────────────────────┤
│  📊  Observability Stack (LlmOpsSm-Observability-{env})        │
│  ├── MLflow Tracking Server (ECS Fargate)                      │
│  ├── Application Load Balancer                                 │
│  ├── CloudWatch Dashboards                                     │
│  ├── SNS Topics & Email Notifications                          │
│  └── Slack Integration                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Event-Driven Workflow

1. **Project Creation** → EventBridge → Lambda → GitHub repository creation
2. **Model Training** → GitHub Actions → SageMaker Pipelines → Model Registry
3. **Model Approval** → EventBridge → Lambda → GitHub deployment workflow
4. **Endpoint Deployment** → GitHub Actions → CDK → SageMaker Endpoints

## Prerequisites

### AWS Account Requirements

- **AWS Account** with appropriate IAM permissions
- **Required Permissions**: `PowerUserAccess` managed policy or equivalent
- **AWS CLI** configured with credentials
- **CDK Bootstrap** completed in target region

### Development Environment

- **Python**: 3.10+ (recommended) or 3.9+
- **Node.js**: 18.x+ (for CDK CLI)
- **AWS CDK CLI**: v2.188.0+
- **Docker Desktop**: Required for Lambda layers
- **Git**: For repository operations

### GitHub Requirements

- **GitHub Organization**: Where repositories will be created
- **Personal Access Token**: GitHub PAT with `repo`, `workflow`, `write:packages` permissions

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/sample-smai-llmops.git
cd sample-smai-llmops/sm-cdk

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -e .

# Install CDK CLI
npm install -g aws-cdk@latest
```

**Note**: Dependencies are managed via `pyproject.toml`. The command `pip install -e .` installs the package in editable mode with all required dependencies.

**For uv users**: If you have [uv](https://github.com/astral-sh/uv) installed, you can use `uv sync` for faster, reproducible installs using the `uv.lock` file.

### 2. Configuration

Edit `llmops_sm/config.py`:

```python
# GitHub Configuration
GitConfig(
    public_llmops_org="your-template-org",
    public_llmops_org_repo="sample-smai-llmops",
    private_github_organization="your-github-org",
    github_token_secret_name="llmops-sm-github-token",
)

# SageMaker Configuration
SageMakerConfig(
    domain_name="llmops-esg-domain",
    execution_role_name="SageMakerExecutionRole-LLMOps",
    project_template_name="esg-benchmarking-template",
    model_package_group_name="esg-benchmarking-models",
)
```

### 3. Bootstrap and Deploy

```bash
# Set required environment variables
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export CDK_DEFAULT_REGION="us-east-1"

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy all stacks
cdk deploy --all --require-approval never
```

### 4. Post-Deployment Configuration

```bash
# Update GitHub token in AWS Secrets Manager
aws secretsmanager update-secret \
  --secret-id llmops-sm-github-token \
  --secret-string '{"token":"your-github-pat-token"}'
```

## Configuration

### Core Configuration Files

#### `llmops_sm/config.py`

Main configuration file containing all platform settings:

```python
@dataclass
class GitConfig:
    public_llmops_org: str                    # Source template organization
    public_llmops_org_repo: str               # Source repository name
    public_llmops_org_repo_folder: str        # Template folder path
    public_repo_branch: str                   # Source branch
    oidc_role_github_workflow: str            # GitHub Actions role name
    private_github_organization: str          # Target GitHub organization
    private_deploy_repo_default_branch: str   # Default branch for repos
    github_token_secret_name: str             # Secret name for GitHub token

@dataclass
class SageMakerConfig:
    domain_name: str                          # SageMaker Domain name
    execution_role_name: str                  # SageMaker execution role
    project_template_name: str                # Service Catalog template
    model_package_group_name: str             # Model Registry group
```

#### Environment-Specific Settings

```python
# Development environment
ENVIRONMENT = "dev"

# AWS Resource Naming
STACK_PREFIX = "LlmOpsSm"
DOMAIN_STACK_NAME = f"{STACK_PREFIX}-Domain-{ENVIRONMENT}"
INFRASTRUCTURE_STACK_NAME = f"{STACK_PREFIX}-Infrastructure-{ENVIRONMENT}"
OBSERVABILITY_STACK_NAME = f"{STACK_PREFIX}-Observability-{ENVIRONMENT}"
```

### GitHub Integration Setup

#### 1. Create GitHub Personal Access Token

1. Go to **GitHub → Settings → Developer settings → Personal access tokens**
2. Click **"Generate new token (classic)"**
3. Select scopes:
   - `repo` (Full control of private repositories)
   - `workflow` (Update GitHub Action workflows)
   - `write:packages` (Upload packages to GitHub Package Registry)
4. Copy the generated token

#### 2. Store Token in AWS Secrets Manager

```bash
# Create secret with GitHub token
aws secretsmanager create-secret \
  --name llmops-sm-github-token \
  --description "GitHub Personal Access Token for LLMOps platform" \
  --secret-string '{"token":"your-github-pat-token"}'

# Or update existing secret
aws secretsmanager update-secret \
  --secret-id llmops-sm-github-token \
  --secret-string '{"token":"your-github-pat-token"}'
```

## Deployment

### Deployment Options

#### Option 1: Single Command Deployment (Recommended)

```bash
# Deploy all stacks in correct order
cdk deploy --all --require-approval never

# With approval prompts
cdk deploy --all
```

#### Option 2: Individual Stack Deployment

```bash
# Deploy in dependency order
cdk deploy LlmOpsSm-Domain-dev
cdk deploy LlmOpsSm-Infrastructure-dev
cdk deploy LlmOpsSm-Observability-dev
```

#### Option 3: Using Make Commands

```bash
# From project root
make deploy              # Deploy all stacks
make deploy-domain       # Deploy domain stack only
make deploy-infrastructure  # Deploy infrastructure stack only
make deploy-observability   # Deploy observability stack only
```

### Deployment Verification

```bash
# Check stack status
aws cloudformation describe-stacks --stack-name LlmOpsSm-Domain-dev
aws cloudformation describe-stacks --stack-name LlmOpsSm-Infrastructure-dev
aws cloudformation describe-stacks --stack-name LlmOpsSm-Observability-dev

# Get important outputs
aws cloudformation describe-stacks \
  --stack-name LlmOpsSm-Domain-dev \
  --query 'Stacks[0].Outputs'

# Check SageMaker domain
aws sagemaker list-domains

# Verify MLflow service
aws ecs describe-services --cluster mlflow-cluster-dev --services mlflow-tracking-dev
```

## Post-Deployment Setup

### 1. SageMaker Studio Access

```bash
# Get SageMaker Studio URL
aws cloudformation describe-stacks \
  --stack-name LlmOpsSm-Domain-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`DomainUrl`].OutputValue' \
  --output text
```

### 2. MLflow Tracking Server

```bash
# Get MLflow URL
aws cloudformation describe-stacks \
  --stack-name LlmOpsSm-Observability-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`MLflowTrackingServerUrl`].OutputValue' \
  --output text
```

### 3. Service Catalog Templates

1. Navigate to **AWS Console → Service Catalog → Portfolios**
2. Find **"ESG Benchmarking MLOps Templates"** portfolio
3. Verify template is available and properly configured

### 4. Notification Setup

#### Email Notifications

```bash
# Verify email identity in SES
aws sesv2 put-email-identity --email-identity your-email@domain.com

# Update notification email
aws secretsmanager update-secret \
  --secret-id llmops-notification-email-dev \
  --secret-string '{"email":"your-team@domain.com"}'
```

#### Slack Integration

```bash
# Update Slack webhook URL
aws secretsmanager update-secret \
  --secret-id llmops-slack-webhook-dev \
  --secret-string '{"webhook_url":"https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"}'
```

## Troubleshooting

### Common Deployment Issues

#### 1. CDK Bootstrap Issues

```bash
# Re-bootstrap CDK
cdk bootstrap --force

# Check bootstrap stack
aws cloudformation describe-stacks --stack-name CDKToolkit
```

#### 2. Lambda Layer Build Issues

```bash
# Ensure Docker is running
docker --version

# Rebuild Lambda layers
cd layers/python-layer
rm -rf python/
mkdir -p python/lib/python3.10/site-packages
pip install -r requirements.txt -t python/lib/python3.10/site-packages/
```

#### 3. GitHub Integration Issues

```bash
# Verify GitHub token
aws secretsmanager get-secret-value --secret-id llmops-sm-github-token

# Test GitHub API access
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user
```

#### 4. SageMaker Domain Issues

```bash
# Check domain status
aws sagemaker describe-domain --domain-id YOUR_DOMAIN_ID

# List execution roles
aws iam list-roles --query 'Roles[?contains(RoleName, `SageMaker`)]'
```

### Log Locations

- **CDK Deployment**: CloudFormation console → Stack events
- **Lambda Functions**: CloudWatch → `/aws/lambda/{function-name}`
- **Step Functions**: Step Functions console → Execution history
- **MLflow Service**: CloudWatch → `/aws/ecs/mlflow-tracking-dev`

### Debug Commands

```bash
# Check CDK diff
cdk diff --all

# Synthesize templates
cdk synth --all

# List CDK stacks
cdk list

# Check stack resources
aws cloudformation list-stack-resources --stack-name LlmOpsSm-Infrastructure-dev
```

## Development

### Project Structure

```
sm-cdk/
├── app.py                          # CDK application entry point
├── main.py                         # Alternative entry point
├── requirements.txt                # Python dependencies
├── cdk.json                        # CDK configuration
├── pyproject.toml                  # Python project configuration
├── setup.py                        # Package setup
├── llmops_sm/                      # Core platform code
│   ├── __init__.py
│   ├── config.py                   # Configuration settings
│   ├── constructs/                 # CDK constructs
│   │   ├── __init__.py
│   │   ├── depedency_layer.py      # Python dependencies layer
│   │   ├── depedency_layer.py      # Python dependencies layer
│   │   └── model_approval_lambda_construcy.py
│   ├── stacks/                     # CDK stacks
│   │   ├── main_stack.py           # Infrastructure stack
│   │   ├── observability_stack.py  # MLflow and monitoring
│   │   └── sagemaker_domain_stack.py # SageMaker domain
│   └── utils/                      # Utility functions
├── lambda/                         # Lambda function code
│   ├── check-project-status/
│   ├── create-deploy-repository/
│   ├── model-approval-trigger/
│   └── sync-repositories/
├── layers/                         # Lambda layers
│   └── python-layer/               # Python dependencies (requests, etc.)
└── templates/                      # Service Catalog templates
    └── esg-project-template.yaml
```

## Clean-up

### Complete Platform Cleanup

```bash
# Destroy all stacks (in reverse order)
cdk destroy LlmOpsSm-Observability-dev
cdk destroy LlmOpsSm-Infrastructure-dev
cdk destroy LlmOpsSm-Domain-dev

# Or destroy all at once
cdk destroy --all
```

### Manual Resource Cleanup

```bash
# Clean up S3 buckets (if not empty)
aws s3 rm s3://your-artifact-bucket --recursive
aws s3 rb s3://your-artifact-bucket

# Delete SageMaker endpoints
aws sagemaker list-endpoints --query 'Endpoints[].EndpointName' --output text | \
  xargs -I {} aws sagemaker delete-endpoint --endpoint-name {}

# Clean up ECR repositories
aws ecr describe-repositories --query 'repositories[].repositoryName' --output text | \
  xargs -I {} aws ecr delete-repository --repository-name {} --force
```

### Using Automation Scripts

```bash
# From project root
./scripts/manual-cleanup-and-redeploy.sh  # Complete cleanup and redeploy
./scripts/redeploy-llmops-platform.sh     # Standard redeploy
```

## Advanced Configuration

### Multi-Environment Setup

```python
# llmops_sm/config.py
ENVIRONMENTS = {
    "dev": {
        "domain_name": "llmops-dev-domain",
        "instance_types": ["ml.t3.medium"],
    },
    "prod": {
        "domain_name": "llmops-prod-domain",
        "instance_types": ["ml.m5.large", "ml.m5.xlarge"],
    }
}
```

### Custom Notification Channels

```python
# Add Microsoft Teams integration
class TeamsNotificationConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str):
        super().__init__(scope, construct_id)

        # Teams webhook integration
        self.teams_webhook = secretsmanager.Secret(
            self, "TeamsWebhook",
            description="Microsoft Teams webhook URL"
        )
```

---

## 🚀 Ready to Deploy?

1. [Configure GitHub Integration](#github-integration-setup)
2. [Update Configuration](#configuration)
3. [Deploy Platform](#deployment)
4. [Verify Setup](#post-deployment-setup)

**Total Deployment Time**: ~15-20 minutes

**Next Steps**: Create your first SageMaker project and start training ESG models!

---
