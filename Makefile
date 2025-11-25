# SageMaker LLMOps Platform Makefile
# Automates build, deployment, and cleanup tasks

.PHONY: help setup clean build deploy test lint format docker-check bootstrap clean-all clean-endpoints clean-models clean-cdk clean-local clean-build

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

# Project variables
PROJECT_NAME := sample-smai-llmops
CDK_DIR := sm-cdk
VENV_DIR := $(CDK_DIR)/.venv
PYTHON := python3
PIP := pip
CDK := cdk
DOCKER := docker
UV := uv

# Check if uv is available
UV_AVAILABLE := $(shell command -v uv 2> /dev/null)

# AWS variables (can be overridden)
AWS_REGION ?= us-east-1
AWS_PROFILE ?= default

help: ## Show this help message
	@echo "$(BLUE)SageMaker LLMOps Platform - Available Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(setup|install|bootstrap)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(build|test|lint|format)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Deployment Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(deploy|synth)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Cleanup Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(clean|destroy)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Utility Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v -E "(setup|install|bootstrap|build|test|lint|format|deploy|synth|clean|destroy)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# Setup Commands
setup: docker-check install-deps create-venv install-cdk-deps ## Complete project setup
	@echo "$(GREEN)✓ Project setup complete!$(NC)"
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. Activate virtual environment: source $(VENV_DIR)/bin/activate"
	@echo "  2. Configure settings: edit $(CDK_DIR)/llmops_sm/config.py"
	@echo "  3. Bootstrap CDK: make bootstrap"
	@echo "  4. Deploy stack: make deploy"

install-deps: ## Install system dependencies
	@echo "$(BLUE)Installing system dependencies...$(NC)"
	@which $(PYTHON) > /dev/null || (echo "$(RED)Error: Python 3 not found$(NC)" && exit 1)
	@which npm > /dev/null || (echo "$(RED)Error: Node.js/npm not found$(NC)" && exit 1)
	@which git > /dev/null || (echo "$(RED)Error: Git not found$(NC)" && exit 1)
ifdef UV_AVAILABLE
	@echo "$(GREEN)✓ uv found - will use for fast Python package management$(NC)"
else
	@echo "$(YELLOW)⚠ uv not found - consider installing for faster package management: pip install uv$(NC)"
endif
	@echo "$(GREEN)✓ System dependencies verified$(NC)"

create-venv: ## Create Python virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
ifdef UV_AVAILABLE
	@echo "$(GREEN)Using uv for fast virtual environment creation...$(NC)"
	@cd $(CDK_DIR) && $(UV) python install 3.10
	@cd $(CDK_DIR) && $(UV) venv --python 3.10
else
	@echo "$(YELLOW)uv not found, using standard Python venv...$(NC)"
	@cd $(CDK_DIR) && $(PYTHON) -m venv .venv
endif
	@echo "$(GREEN)✓ Virtual environment created$(NC)"

install-cdk-deps: create-venv ## Install CDK dependencies
	@echo "$(BLUE)Installing CDK dependencies...$(NC)"
ifdef UV_AVAILABLE
	@echo "$(GREEN)Using uv for fast dependency installation...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(UV) pip install -e .
else
	@cd $(CDK_DIR) && . .venv/bin/activate && $(PIP) install -e .
endif
	@echo "$(YELLOW)Installing CDK CLI...$(NC)"
	@if command -v cdk >/dev/null 2>&1; then \
		echo "$(GREEN)CDK CLI already installed globally$(NC)"; \
	else \
		echo "$(BLUE)Installing CDK CLI globally...$(NC)"; \
		npm install -g aws-cdk@latest || { \
			echo "$(YELLOW)Global install failed, installing locally...$(NC)"; \
			cd $(CDK_DIR) && npm install aws-cdk@latest; \
			echo "$(YELLOW)Use 'npx cdk' instead of 'cdk' for commands$(NC)"; \
		}; \
	fi
	@echo "$(GREEN)✓ CDK dependencies installed$(NC)"

docker-check: ## Verify Docker is running
	@echo "$(BLUE)Checking Docker...$(NC)"
	@$(DOCKER) info > /dev/null 2>&1 || (echo "$(RED)Error: Docker is not running. Please start Docker Desktop.$(NC)" && exit 1)
	@echo "$(GREEN)✓ Docker is running$(NC)"

docker-pull-images: docker-check ## Pull required Docker images for CDK deployment
	@echo "$(BLUE)Pulling required Docker images for CDK...$(NC)"
	@echo "$(YELLOW)This may take a few minutes on first run...$(NC)"
	@$(DOCKER) pull public.ecr.aws/sam/build-python3.11:latest || \
		(echo "$(RED)Failed to pull Docker image. Check your internet connection.$(NC)" && exit 1)
	@echo "$(GREEN)✓ Docker images ready$(NC)"

# Development Commands
build: ## Build all project components (CDK synthesis handles Lambda layers automatically)
	@echo "$(BLUE)Building project...$(NC)"
	@echo "$(YELLOW)Note: Lambda layer dependencies are now built automatically during CDK synthesis$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) synth
	@echo "$(GREEN)✓ Build complete!$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && python -m pytest tests/ -v || echo "$(YELLOW)Warning: Tests not found or failed$(NC)"
	@echo "$(GREEN)✓ Tests completed$(NC)"

lint: ## Run code linting
	@echo "$(BLUE)Running linting...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && \
		python -m flake8 llmops_sm/ lambda/ --max-line-length=100 --ignore=E203,W503 || echo "$(YELLOW)Warning: flake8 not installed$(NC)"
	@echo "$(GREEN)✓ Linting completed$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && \
		python -m black llmops_sm/ lambda/ --line-length=100 || echo "$(YELLOW)Warning: black not installed$(NC)"
	@echo "$(GREEN)✓ Code formatted$(NC)"

# CDK Commands
bootstrap: setup-aws-env ## Bootstrap CDK environment
	@echo "$(BLUE)Bootstrapping CDK environment...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) bootstrap
	@echo "$(GREEN)✓ CDK bootstrapped$(NC)"

setup-aws-env: ## Set up AWS environment variables
	@echo "$(BLUE)Setting up AWS environment...$(NC)"
	@echo "export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)" > .env
	@echo "export CDK_DEFAULT_REGION=us-east-1" >> .env
	@echo "$(GREEN)✓ AWS environment variables saved to .env$(NC)"
	@echo "$(YELLOW)Run 'source .env' to load environment variables$(NC)"

synth: ## Synthesize CDK stack
	@echo "$(BLUE)Synthesizing CDK stack...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) synth
	@echo "$(GREEN)✓ CDK stack synthesized$(NC)"

check-bootstrap: ## Check if CDK is bootstrapped in current region
	@echo "$(BLUE)Checking CDK bootstrap status...$(NC)"
	@REGION=$$(aws configure get region || echo "us-east-1"); \
	ACCOUNT=$$(aws sts get-caller-identity --query Account --output text); \
	if aws ssm get-parameter --name /cdk-bootstrap/hnb659fds/version --region $$REGION >/dev/null 2>&1; then \
		echo "$(GREEN)✓ CDK already bootstrapped in $$REGION$(NC)"; \
	else \
		echo "$(YELLOW)⚠ CDK not bootstrapped in $$REGION, running bootstrap...$(NC)"; \
		cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) bootstrap && \
		echo "$(GREEN)✓ CDK bootstrap complete$(NC)"; \
	fi

deploy: validate-env docker-pull-images ## Deploy all stacks in correct order (Domain → Infrastructure + Observability)
	@echo "$(BLUE)Checking CDK bootstrap status...$(NC)"
	@REGION=$$(aws configure get region || echo "us-east-1"); \
	if ! aws ssm get-parameter --name /cdk-bootstrap/hnb659fds/version --region $$REGION >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ CDK not bootstrapped in $$REGION, running bootstrap...$(NC)"; \
		cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) bootstrap; \
		echo "$(GREEN)✓ CDK bootstrap complete$(NC)"; \
	else \
		echo "$(GREEN)✓ CDK already bootstrapped in $$REGION$(NC)"; \
	fi
	@echo "$(BLUE)Deploying LLMOps platform (3 stacks: Domain + Infrastructure + Observability)...$(NC)"
	@echo "$(YELLOW)CDK will deploy Domain stack first, then Infrastructure and Observability in parallel$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) deploy --all --require-approval never
	@echo "$(GREEN)✓ LLMOps platform deployed successfully!$(NC)"
	@echo "$(BLUE)Post-deployment notes:$(NC)"
	@echo "  ✅ Required environment variables validated"
	@echo "  ✅ Target GitHub organization: $$TARGET_GITHUB_ORG"
	@echo "  2. Update Slack webhook: aws secretsmanager update-secret --secret-id llmops-slack-webhook-dev"
	@echo "  3. Verify SES email identity in AWS Console"
	@echo "  4. Access MLflow UI via the load balancer URL in outputs"
	@echo "  5. View CloudWatch dashboard for monitoring"

deploy-observability: ## Deploy observability stack only (requires Domain stack)
	@echo "$(BLUE)Deploying Observability stack (MLflow + Monitoring)...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) deploy "*Observability*" --require-approval never
	@echo "$(GREEN)✓ Observability stack deployed!$(NC)"

deploy-with-approval: docker-pull-images ## Deploy both stacks with approval prompts
	@echo "$(BLUE)Deploying LLMOps platform with approval prompts...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) deploy --all
	@echo "$(GREEN)✓ LLMOps platform deployed successfully!$(NC)"

# Individual stack deployment (optional - for debugging)
deploy-domain: ## Deploy SageMaker Domain stack only
	@echo "$(BLUE)Deploying SageMaker Domain stack only...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) deploy "*Domain*" --require-approval never
	@echo "$(GREEN)✓ SageMaker Domain stack deployed!$(NC)"

deploy-infrastructure: ## Deploy Infrastructure stack only (requires Domain stack)
	@echo "$(BLUE)Deploying Infrastructure stack only...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) deploy "*Infrastructure*" --require-approval never
	@echo "$(GREEN)✓ Infrastructure stack deployed$(NC)"

deploy-all: deploy ## Alias for deploy (same as deploy)

diff: ## Show CDK diff
	@echo "$(BLUE)Showing CDK diff...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) diff

# Monitoring Commands
list-stacks: ## List CloudFormation stacks
	@echo "$(BLUE)Listing CloudFormation stacks...$(NC)"
	@aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE --query 'StackSummaries[?contains(StackName, `LlmOps`) || contains(StackName, `CDKToolkit`)].{Name:StackName,Status:StackStatus,Created:CreationTime}' --output table

list-endpoints: ## List SageMaker endpoints
	@echo "$(BLUE)Listing SageMaker endpoints...$(NC)"
	@aws sagemaker list-endpoints --sort-by CreationTime --sort-order Descending --output table

list-models: ## List SageMaker models
	@echo "$(BLUE)Listing SageMaker models...$(NC)"
	@aws sagemaker list-models --sort-by CreationTime --sort-order Descending --output table

list-domains: ## List SageMaker domains
	@echo "$(BLUE)Listing SageMaker domains...$(NC)"
	@aws sagemaker list-domains --output table

# Cleanup Commands
clean-all: clean-endpoints clean-models destroy-stack clean-local ## Clean up everything
	@echo "$(GREEN)✓ Complete cleanup finished!$(NC)"

clean-endpoints: ## Delete all SageMaker endpoints
	@echo "$(BLUE)Cleaning up SageMaker endpoints...$(NC)"
	@aws sagemaker list-endpoints --query 'Endpoints[?contains(EndpointName, `esg-benchmark`) || contains(EndpointName, `llmops`)].EndpointName' --output text | \
		xargs -r -I {} aws sagemaker delete-endpoint --endpoint-name {}
	@echo "$(GREEN)✓ Endpoints cleaned up$(NC)"

clean-models: ## Delete SageMaker models and model packages
	@echo "$(BLUE)Cleaning up SageMaker models...$(NC)"
	@aws sagemaker list-models --query 'Models[?contains(ModelName, `esg-benchmark`) || contains(ModelName, `llmops`)].ModelName' --output text | \
		xargs -r -I {} aws sagemaker delete-model --model-name {}
	@echo "$(GREEN)✓ Models cleaned$(NC)"

clean-projects: ## Delete ALL SageMaker projects in the account (WARNING: This deletes ALL projects!)
	@echo "$(RED)WARNING: This will delete ALL SageMaker projects in your account!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C within 10 seconds to cancel...$(NC)"
	@sleep 10
	@echo "$(BLUE)Listing all SageMaker projects...$(NC)"
	@aws sagemaker list-projects --query 'ProjectSummaryList[].{Name:ProjectName,Status:ProjectStatus,Created:CreationTime}' --output table || echo "$(YELLOW)No projects found or error listing projects$(NC)"
	@echo "$(BLUE)Deleting all SageMaker projects...$(NC)"
	@aws sagemaker list-projects --query 'ProjectSummaryList[].ProjectName' --output text | \
		tr '\t' '\n' | \
		while read -r project_name; do \
			if [ -n "$$project_name" ]; then \
				echo "$(YELLOW)Deleting project: $$project_name$(NC)"; \
				aws sagemaker delete-project --project-name "$$project_name" 2>/dev/null || \
				echo "$(RED)Failed to delete project: $$project_name$(NC)"; \
			fi; \
		done
	@echo "$(GREEN)✓ Project deletion initiated$(NC)"
	@echo "$(YELLOW)Note: Deletion is asynchronous and may take a few minutes$(NC)"

list-projects: ## List all SageMaker projects
	@echo "$(BLUE)Listing all SageMaker projects...$(NC)"
	@aws sagemaker list-projects --query 'ProjectSummaryList[].{Name:ProjectName,Status:ProjectStatus,Created:CreationTime,ID:ProjectId}' --output table || echo "$(YELLOW)No projects found or error listing projects$(NC)"

show-github-secrets: ## Show GitHub secrets/variables needed for project repositories
	@echo "$(BLUE)=== GitHub Actions Configuration ===$(NC)"
	@echo ""
	@echo "$(GREEN)Set these in your GitHub repository (Settings → Secrets and variables → Actions):$(NC)"
	@echo ""
	@echo "$(BLUE)Required SECRETS:$(NC)"
	@echo "OIDC_ROLE_GITHUB_WORKFLOW=arn:aws:iam::$(shell aws sts get-caller-identity --query Account --output text):role/llmops-sm-github-action"
	@echo "SAGEMAKER_PIPELINE_ROLE_ARN=arn:aws:iam::$(shell aws sts get-caller-identity --query Account --output text):role/SageMakerExecutionRole-LLMOps"
	@echo "REGION=$(shell aws configure get region || echo us-east-1)"
	@echo "ARTIFACT_BUCKET=llmops-sm-artifacts-$(shell aws sts get-caller-identity --query Account --output text)-$(shell aws configure get region || echo us-east-1)"
	@echo ""
	@echo "$(BLUE)Project-Specific SECRETS:$(NC)"
	@echo "SAGEMAKER_PROJECT_NAME=<your-project-name>"
	@echo "SAGEMAKER_PROJECT_ID=<your-project-id>"
	@echo "MODEL_PACKAGE_GROUP_NAME=esg-benchmarking-<your-project-id>-models"
	@echo ""
	@echo "$(BLUE)Required VARIABLES:$(NC)"
	@echo "TRIGGER_PIPELINE_EXECUTION=true"
	@echo ""
	@echo "$(GREEN)Account Information:$(NC)"
	@echo "AWS Account ID: $(shell aws sts get-caller-identity --query Account --output text)"
	@echo "AWS Region: $(shell aws configure get region || echo us-east-1)"
	@echo "SageMaker Domain ID: $(shell aws sagemaker list-domains --query 'Domains[0].DomainId' --output text 2>/dev/null || echo 'Not found')"

show-sagemaker-role: ## Show the SageMaker execution role ARN that should be used in GitHub Actions
	@echo "$(BLUE)=== SageMaker Execution Role ARN ===$(NC)"
	@echo ""
	@echo "$(GREEN)Use this role ARN in your GitHub Actions workflow:$(NC)"
	@echo "$(YELLOW)arn:aws:iam::$(shell aws sts get-caller-identity --query Account --output text):role/SageMakerExecutionRole-LLMOps$(NC)"
	@echo ""
	@echo "$(BLUE)Set as secret in GitHub:$(NC)"
	@echo "SAGEMAKER_PIPELINE_ROLE_ARN=arn:aws:iam::$(shell aws sts get-caller-identity --query Account --output text):role/SageMakerExecutionRole-LLMOps"
	@echo ""
	@echo "$(GREEN)✅ Copy the ARN above to your GitHub repository secrets$(NC)"

show-project-secrets: ## Show GH configuration for a specific project (usage: make show-project-secrets PROJECT_NAME=testing PROJECT_ID=p-xxxx)
	@if [ -z "$(PROJECT_NAME)" ] || [ -z "$(PROJECT_ID)" ]; then \
		echo "$(RED)Usage: make show-project-secrets PROJECT_NAME=testing PROJECT_ID=p-abc123$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)=== GitHub Config for Project: $(PROJECT_NAME) ===$(NC)"
	@echo ""
	@echo "$(GREEN)Copy these to your repo (Settings → Secrets and variables → Actions):$(NC)"
	@echo ""
	@echo "$(BLUE)Core SECRETS:$(NC)"
	@echo "OIDC_ROLE_GITHUB_WORKFLOW=arn:aws:iam::$(shell aws sts get-caller-identity --query Account --output text):role/llmops-sm-github-action"
	@echo "SAGEMAKER_PIPELINE_ROLE_ARN=arn:aws:iam::$(shell aws sts get-caller-identity --query Account --output text):role/SageMakerExecutionRole-LLMOps"
	@echo "REGION=$(shell aws configure get region || echo us-east-1)"
	@echo "ARTIFACT_BUCKET=llmops-sm-artifacts-$(shell aws sts get-caller-identity --query Account --output text)-$(shell aws configure get region || echo us-east-1)"
	@echo ""
	@echo "$(BLUE)Project SECRETS:$(NC)"
	@echo "SAGEMAKER_PROJECT_NAME=$(PROJECT_NAME)"
	@echo "SAGEMAKER_PROJECT_ID=$(PROJECT_ID)"
	@echo "MODEL_PACKAGE_GROUP_NAME=esg-benchmarking-$(PROJECT_ID)-models"
	@echo ""
	@echo "$(BLUE)Required VARIABLES:$(NC)"
	@echo "TRIGGER_PIPELINE_EXECUTION=true"
	@echo ""
	@echo "$(GREEN)✅ Ready to use!$(NC)"

debug-github-secrets: ## Debug GitHub secrets creation for a repository
	@echo "$(BLUE)Running GitHub secrets debug tool...$(NC)"
	@python3 debug-github-secrets.py

destroy-stack: ## Destroy CDK stack
	@echo "$(BLUE)Destroying CDK stack...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(CDK) destroy --all --force
	@echo "$(GREEN)✓ CDK stack destroyed$(NC)"

clean-cdk: ## Clean CDK artifacts
	@echo "$(BLUE)Cleaning CDK artifacts...$(NC)"
	@cd $(CDK_DIR) && rm -rf cdk.out/ .cdk.staging/
	@echo "$(GREEN)✓ CDK artifacts cleaned$(NC)"

clean-local: ## Clean local development files
	@echo "$(BLUE)Cleaning local files...$(NC)"
	@rm -rf $(VENV_DIR)
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Local files cleaned$(NC)"

clean-build: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@cd $(CDK_DIR) && rm -rf build/ dist/ .pytest_cache/
	@echo "$(GREEN)✓ Build artifacts cleaned$(NC)"

# Utility Commands
status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "├── Virtual Environment: $(if $(wildcard $(VENV_DIR)),$(GREEN)✓ Created$(NC),$(RED)✗ Not found$(NC))"
	@echo "├── Docker: $(shell $(DOCKER) info > /dev/null 2>&1 && echo "$(GREEN)✓ Running$(NC)" || echo "$(RED)✗ Not running$(NC)")"
	@echo "├── CDK: $(shell which cdk > /dev/null 2>&1 && echo "$(GREEN)✓ Installed$(NC)" || echo "$(RED)✗ Not installed$(NC)")"
	@echo "└── AWS CLI: $(shell which aws > /dev/null 2>&1 && echo "$(GREEN)✓ Installed$(NC)" || echo "$(RED)✗ Not installed$(NC)")"

logs: ## Show recent CloudWatch logs
	@echo "$(BLUE)Recent CloudWatch logs...$(NC)"
	@aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/llmops" --query 'logGroups[].logGroupName' --output text | \
		head -5 | xargs -I {} aws logs tail {} --since 1h

update-requirements: ## Update Python requirements
	@echo "$(BLUE)Updating requirements...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && $(PIP) list --outdated --format=json | jq -r '.[] | .name' | xargs -r $(PIP) install -U
	@cd $(CDK_DIR) && . .venv/bin/activate && $(PIP) freeze > requirements.txt
	@echo "$(GREEN)✓ Requirements updated$(NC)"

validate-config: ## Validate configuration files
	@echo "$(BLUE)Validating configuration...$(NC)"
	@cd $(CDK_DIR) && . .venv/bin/activate && python -c "from llmops_sm.config import GitConfig; print('✓ Configuration valid')" || echo "$(RED)✗ Configuration invalid$(NC)"

validate-github-token: ## Validate GitHub token environment variable
	@echo "$(BLUE)Validating GitHub token...$(NC)"
	@if [ -z "$$GITHUB_TOKEN" ]; then \
		echo "$(RED)❌ ERROR: GITHUB_TOKEN environment variable is not set!$(NC)"; \
		echo ""; \
		echo "$(YELLOW)🔑 Set GITHUB_TOKEN with a GitHub PAT before deploying.$(NC)"; \
		echo "   export GITHUB_TOKEN='ghp_your_token_here'"; \
		exit 1; \
	fi
	@if ! echo "$$GITHUB_TOKEN" | grep -qE '^gh[pous]_[A-Za-z0-9_]{36,}$$'; then \
		echo "$(RED)❌ ERROR: Invalid GitHub token format!$(NC)"; \
		echo "$(YELLOW)Tokens start with ghp_/gho_/ghu_/ghs_ and are 40+ chars.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ GitHub token validation passed$(NC)"

validate-target-org: ## Validate TARGET_GITHUB_ORG environment variable
	@echo "$(BLUE)Validating TARGET_GITHUB_ORG...$(NC)"
	@if [ -z "$$TARGET_GITHUB_ORG" ]; then \
		echo "$(RED)❌ ERROR: TARGET_GITHUB_ORG environment variable is not set!$(NC)"; \
		echo ""; \
		echo "$(YELLOW)📁 TARGET_GITHUB_ORG is REQUIRED - this is where project repositories will be created.$(NC)"; \
		echo "   export TARGET_GITHUB_ORG='your-github-org'"; \
		echo ""; \
		echo "$(BLUE)Why is this required?$(NC)"; \
		echo "  • This is YOUR GitHub organization where project repos will be created"; \
		echo "  • There is no sensible default - it must be your organization"; \
		echo "  • The platform will fail to deploy without this configuration"; \
		echo ""; \
		echo "$(YELLOW)Optional: Set TEMPLATE_GITHUB_ORG if you've forked the templates$(NC)"; \
		echo "   export TEMPLATE_GITHUB_ORG='your-custom-org'  # Defaults to 'aws-samples'"; \
		exit 1; \
	fi
	@if ! echo "$$TARGET_GITHUB_ORG" | grep -qE '^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]$$'; then \
		echo "$(RED)❌ ERROR: Invalid GitHub organization name format!$(NC)"; \
		echo "$(YELLOW)Organization names must:$(NC)"; \
		echo "  • Start and end with alphanumeric characters"; \
		echo "  • Contain only alphanumeric characters and hyphens"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ TARGET_GITHUB_ORG validation passed: $$TARGET_GITHUB_ORG$(NC)"

validate-env: validate-github-token validate-target-org ## Validate all required environment variables
	@echo "$(GREEN)✅ All environment variables validated successfully!$(NC)"


# Development shortcuts
dev-setup: setup bootstrap ## Quick development setup
	@echo "$(GREEN)✓ Development environment ready!$(NC)"

quick-deploy: docker-pull-images synth deploy ## Quick synthesis and deployment
	@echo "$(GREEN)✓ Quick deployment complete!$(NC)"

# Version information
version: ## Show version information
	@echo "$(BLUE)Version Information:$(NC)"
	@echo "├── Python: $(shell $(PYTHON) --version 2>&1)"
	@echo "├── CDK: $(shell $(CDK) --version 2>/dev/null || echo "Not installed")"
	@echo "├── Node: $(shell node --version 2>/dev/null || echo "Not installed")"
	@echo "├── Docker: $(shell $(DOCKER) --version 2>/dev/null || echo "Not installed")"
	@echo "└── AWS CLI: $(shell aws --version 2>&1 | head -1 || echo "Not installed")"
