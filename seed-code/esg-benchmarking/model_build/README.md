# 🏗️ ESG Model Build Pipeline

> **Automated LLM fine-tuning for ESG sustainability report generation**

This repository contains the complete training pipeline for ESG (Environmental, Social, Governance) models using SageMaker Pipelines. It automatically fine-tunes transformer models on ESG datasets and registers them in the SageMaker Model Registry for deployment.

## ⚡ Quick Start

Get your ESG model training in 3 simple steps:

### Step 1: Check GitHub Variables 🔐

Go to **Settings → Secrets and variables → Actions** and check these are there, otherwise the GitHub actions workflow will fail:

```bash
# Core secrets (required)
OIDC_ROLE_GITHUB_WORKFLOW=arn:aws:iam::ACCOUNT:role/github-actions-role
SAGEMAKER_PIPELINE_ROLE_ARN=arn:aws:iam::ACCOUNT:role/sagemaker-execution-role
SAGEMAKER_PROJECT_NAME=your-project-name
SAGEMAKER_PROJECT_ID=p-abc123def456
REGION=us-east-1
ARTIFACT_BUCKET=sagemaker-us-east-1-ACCOUNT
MODEL_PACKAGE_GROUP_NAME=esg-benchmarking-models

# Optional secrets
INPUT_DATA_PATH=s3://your-bucket/esg-data/  # Leave empty for sample data
```

### Step 2: Enable Pipeline Execution 🚀

Go to **Settings → Secrets and variables → Actions → Variables** and add:

```bash
TRIGGER_PIPELINE_EXECUTION=true
```

### Step 3: Start Training 🎯

```bash
# Option A: Manual trigger
Go to Actions → "SageMaker Pipeline build" → "Run workflow"

# Option B: Push code changes
# git add . && git commit -m "Start ESG model training"
# git push

```

**That's it!** 🎉 Your ESG model will be trained and registered automatically.

## 🎯 What This Pipeline Does

This automated pipeline handles the complete ESG model training lifecycle:

### 🔄 **Training Workflow**

1. **Data Processing** → Loads ESG datasets (sample or custom)
2. **Model Fine-tuning** → Fine-tunes DialoGPT on ESG sustainability tasks
3. **Evaluation** → Comprehensive metrics including ROUGE, BERT, and ESG-specific scores
4. **Model Registration** → Automatically registers in SageMaker Model Registry
5. **Notifications** → Updates via GitHub Actions and CloudWatch

### 🛠️ **Key Features**

- ✅ **Zero-Config Training** - Works out of the box with sample data
- ✅ **Flexible Dataset Support** - Use any ESG/Financial NLP dataset
- ✅ **MLflow Tracking** - Complete experiment logging and versioning
- ✅ **ESG-Specific Evaluation** - Domain-specific metrics and benchmarks
- ✅ **Automated CI/CD** - GitHub Actions trigger training on code changes
- ✅ **Cost Optimization** - Efficient resource usage and cleanup

## 📁 Repository Structure

```
model_build/
├── 📄 README.md                        # This guide
├── 🔄 .github/workflows/               # GitHub Actions CI/CD
│   └── build_sagemaker_pipeline.yml    # Main training workflow
├── 🔧 ml_pipelines/                    # SageMaker Pipeline definitions
│   ├── run_pipeline.py                 # Pipeline execution script
│   ├── training/pipeline.py            # Main pipeline definition
│   └── requirements.txt                # Pipeline dependencies
└── 📜 source_scripts/                  # Processing and training scripts
    ├── data/                           # Data utilities
    │   └── download_sample_data.py     # Sample dataset downloader
    ├── preprocessing/                  # Data preprocessing
    │   └── preprocess.py              # ESG data preprocessing
    ├── training/                       # Model training
    │   └── train.py                   # HuggingFace transformer training
    └── evaluate/                       # Model evaluation
        └── evaluate.py                # Multi-metric evaluation
```

## 🚀 Training Dataset

### SusGen-30k Dataset from HuggingFace 📊

**✅ Automatically downloaded** - No manual setup required

The pipeline uses the **WHATX/SusGen-30k** dataset from HuggingFace, which contains 30,000 high-quality ESG sustainability report examples covering:

- **Environmental**: Carbon emissions, renewable energy, waste management, climate risk
- **Social**: Employee diversity, board diversity, executive compensation, labor practices
- **Governance**: Data privacy, cybersecurity, supply chain ethics, corporate transparency

**Dataset Details**:

- **Source**: HuggingFace (WHATX/SusGen-30k)
- **Format**: JSON with `instruction`, `input`, `output` fields
- **Size**: ~30,000 training examples
- **Access**: Public dataset (no authentication required)
- **Download**: Automatic during pipeline execution

The preprocessing step automatically:

1. Downloads the dataset from HuggingFace
2. Applies prompt templates (Mistral or Llama3 format)
3. Creates train/validation/test splits (80/10/10)
4. Saves processed data to S3 for training

## 🔍 Monitoring Training

### SageMaker Console

Track detailed training metrics:

```bash
# View pipeline executions
aws sagemaker list-pipeline-executions --pipeline-name "githubactions-YOUR_PROJECT_ID"

# Check training job details
aws sagemaker describe-training-job --training-job-name "YOUR_TRAINING_JOB"

# View model packages
aws sagemaker list-model-packages --model-package-group-name "esg-benchmarking-models"
```

### MLflow Integration

Access comprehensive experiment tracking:

- **Training metrics** - Loss, accuracy, and custom metrics
- **Model artifacts** - Trained models and evaluation results
- **Parameter tracking** - Hyperparameters and configuration
- **Comparison tools** - Compare different training runs

## 🛠️ Local Development

### Setup Environment

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/your-project-build.git
cd your-project-build

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r ml_pipelines/requirements.txt
pip install datasets  # For HuggingFace dataset loading
```

### Test Data Loading

```bash
# Test loading the SusGen-30k dataset
python -c "from datasets import load_dataset; ds = load_dataset('WHATX/SusGen-30k', split='train'); print(f'Loaded {len(ds)} samples')"
```

### Run Pipeline Locally

```bash
python ml_pipelines/run_pipeline.py \
  --module-name training.pipeline \
  --role-arn arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole-LLMOps \
  --tags '[{"Key":"sagemaker:project-name", "Value":"your-project"}]' \
  --kwargs '{
    "region": "us-east-1",
    "role": "arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole-LLMOps",
    "default_bucket": "sagemaker-us-east-1-ACCOUNT",
    "pipeline_name": "local-test-pipeline",
    "model_package_group_name": "esg-benchmarking-models"
  }'

# Note: The pipeline will automatically download WHATX/SusGen-30k from HuggingFace
```

## 🔍 Troubleshooting

### Common Issues

**❌ Pipeline execution disabled**

```bash
# Solution: Set GitHub variable
TRIGGER_PIPELINE_EXECUTION=true
```

**❌ GitHub Actions authentication failed**

```bash
# Check OIDC role trust relationship
aws iam get-role --role-name llmops-sm-github-action
```

**❌ SageMaker permissions error**

```bash
# Verify execution role permissions
aws iam list-attached-role-policies --role-name SageMakerExecutionRole-LLMOps
```

**❌ Training job failed**

```bash
# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix "/aws/sagemaker/TrainingJobs"
```

## 📈 Performance Optimization

### Training Efficiency

- **Batch Size**: Increase for faster training (if memory allows)
- **Gradient Accumulation**: Use for effective larger batch sizes
- **Mixed Precision**: Enable for faster training on modern GPUs
- **Data Loading**: Optimize data preprocessing and loading

### Cost Optimization

- **Instance Types**: Use appropriate instance sizes for your data
- **Spot Instances**: Enable for cost savings (with checkpointing)
- **Early Stopping**: Prevent overfitting and reduce costs
- **Resource Cleanup**: Automatic cleanup of temporary resources

## 🚀 Next Steps

After successful training:

1. **Review Model Metrics** 📊

   - Check Pipeline Results in SageMaker Studio
   - Compare with previous training runs in MLflow

2. **Approve Model** ✅

   - Go to SageMaker Model Registry
   - Change status from "PendingManualApproval" to "Approved"

3. **Deploy Model** 🚀 (see model-deploy repo that is created)

   - Model approval automatically triggers deployment pipeline
   - Monitor deployment in your deploy repository

---
