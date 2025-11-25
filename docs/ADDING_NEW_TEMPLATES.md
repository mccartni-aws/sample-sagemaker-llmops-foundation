# Adding New Project Templates

This guide explains how to add new use case templates to the LLMOps platform.

## Template Components

Each template consists of:

1. **CloudFormation Template** - Creates AWS resources
2. **Seed Code** - Training and deployment pipelines
3. **Service Catalog Registration** - Makes template available in SageMaker Studio

## Step-by-Step Guide

### 1. Create CloudFormation Template

Create `sm-cdk/templates/your-usecase-template.yaml`:

```yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: "Your Use Case SageMaker Project Template"

Parameters:
  SageMakerProjectName:
    Type: String
    Description: Name of the SageMaker project
    MinLength: 1
    MaxLength: 32

  SageMakerProjectId:
    Type: String
    Description: Service generated Id of the project

Resources:
  ModelPackageGroup:
    Type: AWS::SageMaker::ModelPackageGroup
    Properties:
      ModelPackageGroupName: !Sub "${SageMakerProjectName}-models"
      ModelPackageGroupDescription: "Model package group for your use case"
      Tags:
        - Key: "sagemaker:project-id"
          Value: !Ref SageMakerProjectId
        - Key: "sagemaker:project-name"
          Value: !Ref SageMakerProjectName

  ProjectArtifactsBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "${SageMakerProjectName}-${SageMakerProjectId}-artifacts"
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256

Outputs:
  ModelPackageGroupName:
    Description: Name of the created model package group
    Value: !Ref ModelPackageGroup

  ProjectArtifactsBucketName:
    Description: Name of the project artifacts S3 bucket
    Value: !Ref ProjectArtifactsBucket
```

### 2. Create Seed Code

Create directory structure in `seed-code/`:

```
seed-code/your-usecase/
├── model_build/
│   ├── README.md
│   ├── ml_pipelines/
│   │   ├── requirements.txt
│   │   ├── run_pipeline.py
│   │   └── training/
│   │       └── pipeline.py
│   └── source_scripts/
│       ├── data/
│       │   ├── preprocess_data.py
│       │   └── requirements.txt
│       ├── training/
│       │   ├── train.py
│       │   └── requirements.txt
│       └── evaluate/
│           ├── evaluate.py
│           └── requirements.txt
└── model_deploy/
    ├── README.md
    ├── app.py
    ├── cdk.json
    ├── requirements.txt
    └── deploy_endpoint/
        ├── deploy_endpoint_stack.py
        └── get_approved_package.py
```

**Key Files to Create:**

1. **Training Pipeline** (`model_build/ml_pipelines/training/pipeline.py`):

   - Define SageMaker Pipeline steps
   - Configure data processing, training, and evaluation

2. **Training Script** (`model_build/source_scripts/training/train.py`):

   - Implement model training logic
   - Handle hyperparameters and model saving

3. **Deployment Stack** (`model_deploy/deploy_endpoint/deploy_endpoint_stack.py`):
   - Define CDK stack for SageMaker endpoints
   - Configure auto-scaling and monitoring

### 3. Register in Service Catalog

Edit `sm-cdk/llmops_sm/stacks/main_stack.py`:

```python
def _create_sagemaker_project_template(self):
    # ... existing code ...

    # Add your new template
    your_template = servicecatalog.CloudFormationProduct(
        self,
        "YourUseCaseTemplate",
        product_name="Your Use Case MLOps Project",
        description="Description of your use case",
        owner="LLMOps Platform",
        product_versions=[
            servicecatalog.CloudFormationProductVersion(
                product_version_name="1.0",
                cloud_formation_template=servicecatalog.CloudFormationTemplate.from_asset(
                    "templates/your-usecase-template.yaml"
                ),
            )
        ],
    )

    # Associate with portfolio
    portfolio.add_product(your_template)

    # Add visibility tag
    cdk.Tags.of(your_template).add("sagemaker:studio-visibility", "true")
```

### 4. Update Configuration

Edit `sm-cdk/llmops_sm/config.py` to add template-specific settings if needed:

```python
@dataclass
class YourUseCaseConfig:
    """Your use case specific configuration"""

    base_model_name: str = "your-model-name"
    dataset_formats: list = None
    evaluation_metrics: list = None

    def __post_init__(self):
        if self.dataset_formats is None:
            self.dataset_formats = ["jsonl", "csv"]

        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["accuracy", "f1_score"]
```

### 5. Update Lambda Environment Variables

If your template requires specific configuration, update the Lambda environment variables in `main_stack.py`:

```python
common_env = {
    # ... existing variables ...
    "TEMPLATE_FOLDER": self.config.git.public_llmops_org_repo_folder,
    "YOUR_USECASE_CONFIG": "your-config-value",
}
```

### 6. Deploy

```bash
cd sm-cdk
cdk deploy LlmOpsSm-Infrastructure-dev
```

## Platform Remains Unchanged

Note that adding new templates does **not** require changes to:

- SageMaker Domain Stack
- EventBridge rules
- Lambda functions (unless adding new functionality)
- Step Functions workflows
- MLflow tracking server

The platform infrastructure is generic and works with any template.

## Testing Your Template

### 1. Verify Template in Service Catalog

```bash
# List portfolios
aws servicecatalog list-portfolios

# List products in portfolio
aws servicecatalog search-products-as-admin \
  --portfolio-id <portfolio-id>
```

### 2. Create Test Project

1. Open SageMaker Studio
2. Go to Projects → Create Project
3. Select your new template
4. Enter project name and create

### 3. Verify Repository Creation

```bash
# Check if repositories were created
gh repo list <your-org> --limit 100 | grep <project-name>
```

### 4. Test Training Pipeline

```bash
# Clone the build repository
git clone https://github.com/<your-org>/<project-name>-build.git
cd <project-name>-build

# Trigger training
git commit --allow-empty -m "Test training"
git push
```

## Best Practices

### Seed Code Structure

1. **Keep it Generic**: Avoid hardcoding values; use environment variables
2. **Document Well**: Include comprehensive README files
3. **Follow Conventions**: Match the structure of existing templates
4. **Test Thoroughly**: Verify all pipeline steps work end-to-end

### CloudFormation Template

1. **Minimal Resources**: Only create project-specific resources
2. **Proper Tagging**: Include sagemaker:project-id and sagemaker:project-name tags
3. **Security First**: Enable encryption, versioning, and access controls
4. **Clear Outputs**: Export important resource names/ARNs

### Configuration

1. **Environment-Specific**: Support dev/staging/prod configurations
2. **Sensible Defaults**: Provide good default values
3. **Validation**: Add validation logic for required parameters
4. **Documentation**: Document all configuration options

## Troubleshooting

### Template Not Visible in SageMaker Studio

1. Check Service Catalog portfolio permissions
2. Verify `sagemaker:studio-visibility` tag is set
3. Ensure SageMaker execution role has access to portfolio

### Repository Creation Fails

1. Check Lambda function logs in CloudWatch
2. Verify GitHub token has correct permissions
3. Ensure template folder exists in source repository

### Pipeline Execution Fails

1. Review SageMaker Pipeline execution logs
2. Check IAM role permissions
3. Verify S3 bucket access and paths

## Additional Resources

- [SageMaker Projects Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects.html)
- [Service Catalog Documentation](https://docs.aws.amazon.com/servicecatalog/latest/adminguide/introduction.html)
- [SageMaker Pipelines Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [Platform Architecture Guide](../README.md#platform-architecture)
