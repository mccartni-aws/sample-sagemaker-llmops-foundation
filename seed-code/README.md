# 🌱 Project Seed Code Templates

> **Template repositories for LLM fine-tuning projects on the SageMaker LLMOps platform**

This directory contains seed code templates that are automatically copied to new GitHub repositories when SageMaker projects are created through the LLMOps platform. These templates provide complete MLOps workflows for various LLM fine-tuning use cases.

## 📋 What Are Seed Code Templates?

Seed code templates are pre-configured project structures that serve as starting points for new ML projects. When you create a SageMaker project, the platform:

1. **Copies the appropriate template** to new GitHub repositories
2. **Configures all necessary integrations** (AWS, GitHub Actions, MLflow)
3. **Sets up CI/CD pipelines** for automated training and deployment

This approach ensures consistency, best practices, and rapid project initialization across your organization.

## 🎯 What This Does

When you create a SageMaker project using the ESG template, the platform automatically:

1. **Creates Two Repositories** 📦

   - `{project-name}-build` - Model training and pipeline code
   - `{project-id}-deploy-repo` - Model deployment and endpoint management

2. **Populates with Seed Code** 🌱

   - Complete training pipelines for ESG model fine-tuning
   - Automated deployment workflows for production endpoints
   - Pre-configured GitHub Actions for CI/CD

3. **Configures Everything** ⚙️
   - GitHub secrets and environment variables
   - AWS IAM roles and permissions
   - SageMaker pipeline configurations

## 📁 Repository Structure

```
seed-code/
└── esg-benchmarking/                    # Example: ESG sustainability use case
    ├── model_build/                     # 🏗️ Training pipeline templates
    │   ├── .github/workflows/           # GitHub Actions for training
    │   ├── ml_pipelines/                # SageMaker pipeline definitions
    │   ├── source_scripts/              # Training, preprocessing, evaluation
    │   └── datasets/                    # Sample ESG datasets
    └── model_deploy/                    # 🚀 Deployment pipeline templates
        ├── .github/workflows/           # GitHub Actions for deployment
        ├── config/                      # Environment configurations
        ├── deploy_endpoint/             # Deployment utilities
        └── tests/                       # Integration and unit tests
```

## 🏗️ Included Example: ESG Benchmarking

This directory includes one complete example template to demonstrate the platform's capabilities:

The ESG benchmarking template provides a complete solution for sustainability reporting:

### **Model Build Repository** 🔧

**Purpose**: Train LLM models for ESG sustainability report generation

**Key Features**:

- ✅ **Fine-tuning Scripts** - Specialized for ESG domain adaptation
- ✅ **SageMaker Pipelines** - Automated training workflows with MLflow tracking
- ✅ **Evaluation Framework** - ESG-specific benchmarking and metrics
- ✅ **GitHub Actions** - Automated CI/CD triggered by code changes
- ✅ **Flexible Data Support** - Works with custom ESG/Financial NLP datasets
- ✅ **Data Preprocessing** - Built-in preprocessing for instruction-following format

**Training Pipeline Flow**:

1. **Data Processing** → Load and preprocess ESG datasets
2. **Model Training** → Fine-tune transformer models (DialoGPT-medium)
3. **Evaluation** → ROUGE, BERT scores, and ESG-specific metrics
4. **Model Registration** → Automatic registration in SageMaker Model Registry

### **Model Deploy Repository** 🚀

**Purpose**: Deploy approved ESG models to production endpoints

**Key Features**:

- ✅ **Endpoint Deployment** - SageMaker endpoint creation and management
- ✅ **Event-Driven Automation** - Triggered by model approval events
- ✅ **A/B Testing** - Canary deployment capabilities
- ✅ **Monitoring** - Real-time model performance tracking
- ✅ **Cost Optimization** - Environment-specific instance sizing

**Deployment Pipeline Flow**:

1. **Model Approval** → EventBridge detects approval in Model Registry
2. **Automated Deployment** → GitHub Actions provisions SageMaker endpoints
3. **Testing & Validation** → Automated inference testing with ESG samples
4. **Production Ready** → Endpoints ready for ESG report generation

## 📚 Documentation Links

- **[Model Build README](esg-benchmarking/model_build/README.md)** - Detailed training pipeline documentation
- **[Model Deploy README](esg-benchmarking/model_deploy/README.md)** - Comprehensive deployment guide
- **[Main Platform README](../README.md)** - Overall platform documentation
