"""
Configuration settings for SageMaker LLMOps Platform
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GitConfig:
    """GitHub configuration for LLMOps platform

    This configuration defines two key GitHub organizations:
    1. template_github_org: Where the seed code templates are stored (defaults to aws-samples)
    2. target_github_org: Where new project repositories will be created (REQUIRED - must be set by customer)
    """

    # Source template repository settings
    # This is where the platform reads seed code templates from
    template_github_org: str = "aws-samples"  # Your forked repository organization
    template_github_repo: str = "sample-smai-llmops"
    template_code_folder: str = "seed-code"
    template_repo_branch: str = "main"

    # GitHub Actions and OIDC settings
    oidc_role_github_workflow: str = "llmops-sm-github-action"

    # Target organization and repository settings
    # This is where new project repositories will be created for customers
    target_github_org: Optional[str] = None  # REQUIRED - must be specified by customer
    target_repo_default_branch: str = "main"

    # AWS Secrets Manager
    github_token_secret_name: str = "llmops-sm-github-token"


@dataclass
class SageMakerConfig:
    """SageMaker configuration for LLMOps platform"""

    # SageMaker Domain settings
    domain_name: str = "llmops-esg-domain"
    execution_role_name: str = "SageMakerExecutionRole-LLMOps"

    # Project and template settings
    project_template_name: str = "esg-benchmarking-template"
    model_package_group_name: str = "esg-benchmarking-models"

    # Default instance settings
    default_instance_type: str = "ml.m5.large"
    training_instance_type: str = "ml.m5.large"  # "ml.g4dn.xlarge"
    inference_instance_type: str = "ml.t3.medium"

    # S3 settings
    artifacts_bucket_prefix: str = "llmops-sm-artifacts"
    data_bucket_prefix: str = "llmops-sm-data"

    # Network settings
    enable_internet_access: bool = True  # Set to False for VPC-only (more secure)
    use_dedicated_vpc: bool = False  # Set to True to create dedicated VPC with NAT

    # MLflow settings
    mlflow_server_size: str = "Small"  # Small, Medium, Large
    mlflow_weekly_maintenance_window: str = "SAT:03:00"
    enable_mlflow_tracking: bool = True


@dataclass
class ESGConfig:
    """ESG benchmarking specific configuration"""

    # Model settings
    base_model_name: str = "anthropic.claude-sonnet-4-20250514-v1:0"
    fine_tuning_job_prefix: str = "esg-benchmark-ft"

    # Dataset settings
    esg_dataset_formats: list = None
    benchmark_metrics: list = None

    # Evaluation settings
    evaluation_framework: str = "custom"  # or "mlflow", "langfuse"

    def __post_init__(self):
        if self.esg_dataset_formats is None:
            self.esg_dataset_formats = ["jsonl", "csv", "parquet"]

        if self.benchmark_metrics is None:
            self.benchmark_metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "esg_domain_accuracy",
                "sustainability_score",
            ]


@dataclass
class AWSConfig:
    """AWS-specific configuration"""

    # Default AWS settings
    default_region: str = "us-east-1"

    # Lambda settings
    lambda_timeout: int = 900  # 15 minutes
    lambda_memory: int = 512

    # EventBridge settings
    eventbridge_rule_prefix: str = "llmops-sm"

    # Step Functions settings
    step_function_name: str = "llmops-project-setup"

    # IAM settings
    iam_role_prefix: str = "LlmOpsSm"


@dataclass
class PlatformConfig:
    """Main platform configuration combining all configs"""

    git: GitConfig
    sagemaker: SageMakerConfig
    esg: ESGConfig
    aws: AWSConfig

    # Environment settings
    environment: str = "dev"  # dev, staging, prod

    # Feature flags
    enable_model_monitoring: bool = True
    enable_a_b_testing: bool = True
    enable_auto_scaling: bool = True
    enable_mlflow_tracking: bool = True

    # Notification settings
    notification_email: Optional[str] = None
    slack_webhook_url: Optional[str] = None

    # Observability settings
    enable_xray_tracing: bool = True
    cloudwatch_log_retention_days: int = 7  # 30 for prod
    custom_metrics_namespace: str = "LLMOps/Platform"


# Default configuration instance
DEFAULT_CONFIG = PlatformConfig(
    git=GitConfig(), sagemaker=SageMakerConfig(), esg=ESGConfig(), aws=AWSConfig()
)


def get_config(environment: str = "dev") -> PlatformConfig:
    """
    Get configuration for specific environment

    Args:
        environment: Target environment (dev, staging, prod)

    Returns:
        PlatformConfig: Configuration for the specified environment
    """
    config = DEFAULT_CONFIG
    config.environment = environment

    # Environment-specific overrides
    if environment == "prod":
        config.sagemaker.default_instance_type = "ml.m5.xlarge"
        config.sagemaker.training_instance_type = "ml.g4dn.2xlarge"
        config.sagemaker.inference_instance_type = "ml.m5.2xlarge"
        config.aws.lambda_memory = 1024
        config.enable_auto_scaling = True

    elif environment == "staging":
        config.sagemaker.training_instance_type = "ml.g4dn.xlarge"
        config.enable_a_b_testing = False

    return config


def validate_config(config: PlatformConfig) -> bool:
    """
    Validate configuration settings

    Args:
        config: Configuration to validate

    Returns:
        bool: True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate required fields
    if not config.git.target_github_org:
        raise ValueError(
            "target_github_org must be specified. This is your GitHub "
            "organization where project repositories will be created. "
            "Set this via environment variable TARGET_GITHUB_ORG or in "
            "config.py"
        )

    if not config.sagemaker.domain_name:
        raise ValueError("sagemaker domain_name must be specified")

    # Validate instance types
    valid_instance_types = [
        "ml.t3.medium",
        "ml.t3.large",
        "ml.t3.xlarge",
        "ml.m5.large",
        "ml.m5.xlarge",
        "ml.m5.2xlarge",
        "ml.g4dn.xlarge",
        "ml.g4dn.2xlarge",
        "ml.g4dn.4xlarge",
    ]

    if config.sagemaker.training_instance_type not in valid_instance_types:
        raise ValueError(
            f"Invalid training instance type: {config.sagemaker.training_instance_type}"
        )

    # Validate AWS region
    valid_regions = [
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
        "eu-west-1",
        "eu-west-2",
        "eu-central-1",
        "ap-southeast-1",
        "ap-southeast-2",
        "ap-northeast-1",
    ]

    if config.aws.default_region not in valid_regions:
        raise ValueError(f"Invalid AWS region: {config.aws.default_region}")

    return True
