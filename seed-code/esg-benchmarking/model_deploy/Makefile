# ESG Model Deployment Makefile
# Automates common tasks for ESG sustainability model deployment

.PHONY: help setup install clean test lint format deploy destroy diff synth bootstrap

# Default target
help:
	@echo "ESG Model Deployment - Available targets:"
	@echo "  setup      - Set up Python virtual environment and install dependencies"
	@echo "  install    - Install Python dependencies"
	@echo "  clean      - Clean up build artifacts and cache files"
	@echo "  test       - Run tests"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code"
	@echo "  bootstrap  - Bootstrap CDK environment"
	@echo "  synth      - Synthesize CDK templates"
	@echo "  diff       - Show CDK diff"
	@echo "  deploy     - Deploy ESG model endpoint"
	@echo "  destroy    - Destroy ESG model endpoint"

# Set up development environment
setup:
	@echo "🔧 Setting up ESG model deployment environment..."
	python3 -m venv .venv
	@echo "✅ Virtual environment created"
	@echo "📦 Installing dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "✅ ESG deployment environment setup complete!"
	@echo "💡 Activate with: source .venv/bin/activate"

# Install dependencies
install:
	@echo "📦 Installing ESG deployment dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Clean up
clean:
	@echo "🧹 Cleaning up ESG deployment artifacts..."
	rm -rf .venv/
	rm -rf __pycache__/
	rm -rf **/__pycache__/
	rm -rf *.pyc
	rm -rf **/*.pyc
	rm -rf cdk.out/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "✅ Cleanup complete"

# Run tests
test:
	@echo "🧪 Running ESG deployment tests..."
	python -m pytest tests/ -v
	@echo "✅ Tests completed"

# Lint code
lint:
	@echo "🔍 Linting ESG deployment code..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "✅ Linting complete"

# Format code
format:
	@echo "🎨 Formatting ESG deployment code..."
	black .
	isort .
	@echo "✅ Code formatting complete"

# Bootstrap CDK
bootstrap:
	@echo "🚀 Bootstrapping CDK for ESG deployment..."
	cdk bootstrap
	@echo "✅ CDK bootstrap complete"

# Synthesize CDK templates
synth:
	@echo "🔄 Synthesizing ESG deployment templates..."
	cdk synth
	@echo "✅ CDK synthesis complete"

# Show CDK diff
diff:
	@echo "📊 Showing ESG deployment diff..."
	cdk diff
	@echo "✅ CDK diff complete"

# Deploy ESG model endpoint
deploy:
	@echo "🚀 Deploying ESG model endpoint..."
	cdk deploy --require-approval never
	@echo "✅ ESG model endpoint deployed successfully!"
	@echo "🌱 Your ESG sustainability model is now ready for production use!"

# Deploy with approval
deploy-with-approval:
	@echo "🚀 Deploying ESG model endpoint (with approval)..."
	cdk deploy
	@echo "✅ ESG model endpoint deployed successfully!"

# Destroy ESG model endpoint
destroy:
	@echo "💥 Destroying ESG model endpoint..."
	cdk destroy --force
	@echo "✅ ESG model endpoint destroyed"

# Development workflow
dev-setup: setup bootstrap
	@echo "🎯 ESG development environment ready!"

# Full deployment workflow
full-deploy: install synth deploy
	@echo "🎉 ESG model deployment complete!"

# Check prerequisites
check-prereqs:
	@echo "🔍 Checking ESG deployment prerequisites..."
	@command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed."; exit 1; }
	@command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed."; exit 1; }
	@command -v cdk >/dev/null 2>&1 || { echo "❌ AWS CDK is required but not installed."; exit 1; }
	@command -v aws >/dev/null 2>&1 || { echo "❌ AWS CLI is required but not installed."; exit 1; }
	@echo "✅ All prerequisites are installed"

# Show environment info
info:
	@echo "📋 ESG Deployment Environment Information:"
	@echo "Python version: $(shell python3 --version)"
	@echo "Node.js version: $(shell node --version)"
	@echo "CDK version: $(shell cdk --version)"
	@echo "AWS CLI version: $(shell aws --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repository')"
