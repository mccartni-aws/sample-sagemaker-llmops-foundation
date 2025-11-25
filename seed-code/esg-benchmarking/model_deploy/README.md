# 🚀 ESG Model Deploy Pipeline

> **Automated deployment of ESG sustainability models to production endpoints**

This repository contains the complete deployment pipeline for ESG (Environmental, Social, Governance) models. It automatically deploys approved models from SageMaker Model Registry to production-ready SageMaker endpoints with comprehensive monitoring and testing.

## ⚡ Quick Start

Deploy your ESG model in 3 simple steps:

### Step 2: Approve Your ESG Model ✅

1. **Go to SageMaker Console** → Model Registry
2. **Find your model package group**: `esg-benchmarking-models`
3. **Select your trained model** package
4. **Change status** from "PendingManualApproval" to "Approved"
5. **Add approval comments** (e.g., "ESG model meets accuracy requirements")

### Step 2: Check GitHub Variables 🔐

Go to **Settings → Secrets and variables → Actions** and check at least the following are there:

```bash
# Core secrets (required)
OIDC_ROLE_GITHUB_WORKFLOW=arn:aws:iam::ACCOUNT:role/github-actions-role
SAGEMAKER_PROJECT_NAME=your-project-name
SAGEMAKER_PROJECT_ID=p-abc123def456
REGION=us-east-1
ARTIFACT_BUCKET=sagemaker-us-east-1-ACCOUNT
MODEL_PACKAGE_GROUP_NAME=esg-benchmarking-models
DEPLOY_ACCOUNT=123456789012
SAGEMAKER_DOMAIN_ARN=arn:aws:sagemaker:us-east-1:ACCOUNT:domain/d-abc123
MODEL_BUCKET_ARN=arn:aws:s3:::llmops-sm-artifacts-ACCOUNT-us-east-1
```

### Step 3: Watch Automatic Deployment 🎯

**Automatic Deployment** (Recommended):

- Model approval triggers EventBridge event
- Lambda function automatically starts GitHub Actions deployment
- No manual intervention required!

**Manual Deployment** (Alternative):

```bash
# Go to Actions → "ESG Model Deploy Pipeline" → "Run workflow"
```

**That's it!** 🎉 Your ESG model endpoint will be deployed and ready for production use.

## 🎯 What This Pipeline Does

This automated pipeline handles the complete ESG model deployment lifecycle:

### 🔄 **Deployment Workflow**

1. **Model Detection** → Finds approved ESG models in Model Registry
2. **Infrastructure Provisioning** → Creates SageMaker endpoints using CDK
3. **Endpoint Configuration** → Optimizes for ESG text generation workloads
4. **Testing & Validation** → Automated inference testing with ESG samples
5. **Production Ready** → Endpoints ready for ESG sustainability report generation

### 🛠️ **Key Features**

- ✅ **Event-Driven Automation** - Triggered by model approval events
- ✅ **Zero-Downtime Deployment** - Blue/green deployment capabilities
- ✅ **Auto-Scaling** - Scales based on ESG reporting demand
- ✅ **Cost Optimization** - Right-sized instances for ESG workloads
- ✅ **Comprehensive Testing** - Automated inference validation
- ✅ **Monitoring Integration** - CloudWatch metrics and alarms

## 📁 Repository Structure

```
model_deploy/
├── 📄 README.md                        # This guide
├── 🔄 .github/workflows/               # GitHub Actions CI/CD
│   └── deploy_model_pipeline.yml       # Main deployment workflow
├── ⚙️ config/dev/                      # Configuration management
│   └── endpoint-config.yml             # ESG endpoint configuration
├── 🏗️ deploy_endpoint/                 # Core deployment logic
│   ├── deploy_endpoint_stack.py        # CDK deployment stack
│   ├── get_approved_package.py         # Model package discovery
│   └── utils.py                        # Deployment utilities
├── 📱 app.py                           # CDK application entry point
├── 🔧 cdk.json                         # CDK configuration
├── 📦 requirements.txt                 # Core dependencies
├── 🛠️ requirements-dev.txt             # Development dependencies
├── 🔨 Makefile                         # Build automation
└── 🚫 .gitignore                       # Git ignore patterns
```

## 🚀 Deployment Methods

### Method 1: Automatic Deployment (Recommended) 🤖

**How it works**:

1. **Approve ESG model** in SageMaker Model Registry
2. **EventBridge detects** approval event automatically
3. **Lambda function triggers** GitHub Actions workflow
4. **Deployment happens** without any manual intervention

**Benefits**:

- ✅ Fully automated workflow
- ✅ Immediate deployment after approval
- ✅ No manual steps required
- ✅ Consistent deployment process

### Method 2: Manual Deployment 👤

**When to use**:

- Testing deployment process
- Deploying specific model versions
- Troubleshooting deployment issues

**Steps**:

1. **Ensure model is approved** in Model Registry
2. **Go to GitHub Actions** → "ESG Model Deploy Pipeline"
3. **Click "Run workflow"** → Select log level
4. **Monitor deployment** progress in Actions tab

## 🔧 Endpoint Configuration

### ESG-Optimized Settings

The deployment uses ESG-specific configurations in `config/dev/endpoint-config.yml`:

```yaml
# ESG model endpoint configuration
initial_instance_count: 1
initial_variant_weight: 1
instance_type: "ml.m5.large" # Optimized for ESG text generation
variant_name: "AllTraffic"
```

### Instance Type Recommendations

| Environment     | Instance Type   | Use Case                          |
| --------------- | --------------- | --------------------------------- |
| **Development** | `ml.t3.medium`  | Cost-effective testing            |
| **Staging**     | `ml.m5.large`   | Balanced performance              |
| **Production**  | `ml.m5.xlarge`  | High-throughput ESG reporting     |
| **High-Volume** | `ml.c5.2xlarge` | CPU-optimized for text generation |

### Auto-Scaling Configuration

```python
# Automatic scaling based on ESG reporting demand
auto_scaling_config = {
    "min_capacity": 1,
    "max_capacity": 10,
    "target_value": 70.0,  # CPU utilization target
    "scale_in_cooldown": 300,
    "scale_out_cooldown": 300
}
```

## 🧪 Testing Your ESG Endpoint

### Automated Testing

The pipeline includes comprehensive testing:

```python
# ESG inference test examples
test_cases = [
    {
        "instruction": "Generate an environmental sustainability report",
        "input": "Company: GreenTech Solutions, Focus: Solar energy"
    },
    {
        "instruction": "Create a social responsibility summary",
        "input": "Company: TechForGood Inc, Focus: Employee diversity"
    },
    {
        "instruction": "Generate a corporate governance report",
        "input": "Company: SecureBank Ltd, Focus: Data privacy"
    }
]
```

### Manual Testing

Test your deployed endpoint:

```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name esg-sustainability-endpoint

# Test ESG model inference
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name esg-sustainability-endpoint \
  --content-type application/json \
  --body '{
    "instruction": "Generate an environmental sustainability report for a renewable energy company",
    "input": "Company: GreenTech Solutions, Focus: Solar energy"
  }' \
  response.json

# View response
cat response.json
```

## 🔍 Troubleshooting

### Common Issues

**❌ No approved ESG models found**

```bash
# Check model registry for approved models
aws sagemaker list-model-packages \
  --model-package-group-name esg-benchmarking-models \
  --model-approval-status Approved
```

**❌ Endpoint creation failed**

```bash
# Check CloudFormation events
aws cloudformation describe-stack-events \
  --stack-name ESGEndpointStack

# Check SageMaker endpoint logs
aws logs describe-log-groups \
  --log-group-name-prefix /aws/sagemaker/Endpoints
```

**❌ GitHub Actions authentication failed**

```bash
# Verify OIDC trust relationship
aws iam get-role --role-name llmops-sm-github-action

# Check role permissions
aws iam list-attached-role-policies \
  --role-name llmops-sm-github-action
```

**❌ Endpoint stuck in "Creating" status**

```bash
# Check endpoint configuration
aws sagemaker describe-endpoint-config \
  --endpoint-config-name YOUR_CONFIG_NAME

# Monitor endpoint creation
aws sagemaker describe-endpoint \
  --endpoint-name esg-sustainability-endpoint
```

### Debug Commands

```bash
# List recent deployments
aws sagemaker list-endpoints \
  --sort-by CreationTime \
  --sort-order Descending

# Check model package details
aws sagemaker describe-model-package \
  --model-package-name YOUR_MODEL_PACKAGE_ARN

# View deployment logs
aws logs tail /aws/sagemaker/Endpoints/esg-sustainability-endpoint --follow
```

### Getting Help

1. **Check GitHub Actions logs** for deployment errors
2. **Review CloudFormation events** for infrastructure issues
3. **Monitor CloudWatch logs** for endpoint runtime errors
4. **Verify all secrets** are configured correctly
5. **Test with sample data** to validate endpoint functionality

## 🧹 Cleanup

### Endpoint Cleanup

```bash
# Delete ESG endpoint
aws sagemaker delete-endpoint --endpoint-name esg-sustainability-endpoint

# Delete endpoint configuration
aws sagemaker delete-endpoint-config --endpoint-config-name YOUR_CONFIG_NAME

# Delete model
aws sagemaker delete-model --model-name YOUR_MODEL_NAME
```

### CDK Stack Cleanup

```bash
# Destroy CDK stack
cdk destroy

# Or using make
make clean
```

## 🚀 Next Steps

After successful deployment:

1. **Test Your Endpoint** 🧪

   - Run inference tests with ESG sample data
   - Validate response quality and latency
   - Test with your organization's ESG scenarios

---

**Ready to deploy your ESG model?** Just approve your model in the registry and watch the magic happen!
