"""
Main LLMOps Infrastructure Stack - SECURITY ENHANCED VERSION
Creates EventBridge rules, Lambda functions, Step Functions, and GitHub integration
with separate, scoped-down IAM roles for build and deploy workflows
"""

import os
import json
from typing import Optional
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_events as events,
    aws_events_targets as targets,
    aws_lambda as lambda_,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_iam as iam,
    aws_secretsmanager as secretsmanager,
    aws_sagemaker as sagemaker,
    aws_servicecatalog as servicecatalog,
    CfnOutput,
    Duration,
)
from constructs import Construct

from llmops_sm.config import PlatformConfig
from llmops_sm.stacks.sagemaker_domain_stack import SageMakerDomainStack


class LlmOpsSmStack(Stack):
    """Main infrastructure stack for LLMOps platform"""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        config: PlatformConfig,
        domain_stack: SageMakerDomainStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.config = config
        self.domain_stack = domain_stack

        # Create GitHub token secret
        self.github_secret = self._create_github_secret()

        # Create GitHub OIDC provider
        self.oidc_provider = self._create_github_oidc_provider()

        # Create separate IAM roles for build and deploy workflows
        self.github_build_role = self._create_github_build_role()
        self.github_deploy_role = self._create_github_deploy_role()

        # Create Lambda functions
        self.lambda_functions = self._create_lambda_functions()

        # Create Step Functions workflow
        self.step_function = self._create_step_function()

        # Create SageMaker Project Template
        self.project_template, self.portfolio = (
            self._create_sagemaker_project_template()
        )

        # Create EventBridge rules
        self._create_eventbridge_rules()

        # Create outputs
        self._create_outputs()

    def _create_github_secret(self) -> secretsmanager.Secret:
        """Create secret for GitHub token"""

        # Get GitHub token from environment variable (validated in app.py)
        github_token = os.environ.get("GITHUB_TOKEN")

        if github_token:
            # Create secret with the actual token value
            secret = secretsmanager.Secret(
                self,
                "GitHubTokenSecret",
                secret_name=self.config.git.github_token_secret_name,
                description="GitHub Personal Access Token for LLMOps platform",
                secret_string_value=cdk.SecretValue.unsafe_plain_text(
                    f'{{"token": "{github_token}"}}'
                ),
            )
            print(f"✅ GitHub token secret created with provided token")
        else:
            # Fallback to empty secret (should not happen due to validation in app.py)
            secret = secretsmanager.Secret(
                self,
                "GitHubTokenSecret",
                secret_name=self.config.git.github_token_secret_name,
                description="GitHub Personal Access Token for LLMOps platform",
                generate_secret_string=secretsmanager.SecretStringGenerator(
                    secret_string_template='{"token": ""}',
                    generate_string_key="token",
                    exclude_characters='"@/\\',
                ),
            )
            print("⚠️  GitHub token not found, created empty secret")

        return secret

    def _create_github_oidc_provider(self) -> iam.OpenIdConnectProvider:
        """Create GitHub OIDC provider for GitHub Actions"""

        oidc_provider = iam.OpenIdConnectProvider(
            self,
            "GitHubOidcProvider",
            url="https://token.actions.githubusercontent.com",
            client_ids=["sts.amazonaws.com"],
            thumbprints=[
                "6938fd4d98bab03faadb97b34396831e3780aea1"
            ],  # GitHub's thumbprint
        )

        return oidc_provider

    def _create_github_build_role(self) -> iam.Role:
        """
        Create scoped-down IAM role for GitHub Actions BUILD workflow
        This role is used by model_build repositories to run SageMaker pipelines
        """

        github_build_role = iam.Role(
            self,
            "GitHubActionsBuildRole",
            role_name=f"{self.config.git.oidc_role_github_workflow}-build",
            assumed_by=iam.OpenIdConnectPrincipal(
                self.oidc_provider,
                conditions={
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub": f"repo:{self.config.git.target_github_org}/*-model-build:*"
                    }
                },
            ),
            description="Scoped role for GitHub Actions BUILD workflows (model training)",
            max_session_duration=Duration.hours(2),
        )

        # SageMaker Pipeline permissions - scoped to specific actions needed for build
        github_build_role.add_to_policy(
            iam.PolicyStatement(
                sid="SageMakerPipelineManagement",
                effect=iam.Effect.ALLOW,
                actions=[
                    # Pipeline operations
                    "sagemaker:CreatePipeline",
                    "sagemaker:UpdatePipeline",
                    "sagemaker:StartPipelineExecution",
                    "sagemaker:DescribePipeline",
                    "sagemaker:DescribePipelineExecution",
                    "sagemaker:ListPipelineExecutions",
                    "sagemaker:ListPipelineExecutionSteps",
                    "sagemaker:ListPipelines",
                    "sagemaker:StopPipelineExecution",
                    # Model package operations (for registration)
                    "sagemaker:CreateModelPackage",
                    "sagemaker:DescribeModelPackage",
                    "sagemaker:ListModelPackages",
                    "sagemaker:UpdateModelPackage",
                    "sagemaker:DescribeModelPackageGroup",
                    "sagemaker:ListModelPackageGroups",
                    # Processing and training jobs (created by pipeline)
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:DescribeProcessingJob",
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:CreateTransformJob",
                    "sagemaker:DescribeTransformJob",
                    # Experiments and trials (for tracking)
                    "sagemaker:CreateExperiment",
                    "sagemaker:DescribeExperiment",
                    "sagemaker:CreateTrial",
                    "sagemaker:DescribeTrial",
                    "sagemaker:CreateTrialComponent",
                    "sagemaker:DescribeTrialComponent",
                    "sagemaker:AssociateTrialComponent",
                    # Tags
                    "sagemaker:AddTags",
                    "sagemaker:ListTags",
                ],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline-execution/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:model-package/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:model-package-group/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:processing-job/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:training-job/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:transform-job/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:experiment/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:experiment-trial/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:experiment-trial-component/*",
                ],
            )
        )

        # S3 permissions - scoped to artifacts bucket only
        github_build_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ArtifactsBucketAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation",
                    "s3:GetBucketVersioning",
                ],
                resources=[
                    self.domain_stack.artifacts_bucket.bucket_arn,
                    f"{self.domain_stack.artifacts_bucket.bucket_arn}/*",
                ],
            )
        )

        # CloudWatch Logs - for pipeline execution logs
        github_build_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogsForPipeline",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*",
                ],
            )
        )

        # IAM PassRole - only for SageMaker execution role
        github_build_role.add_to_policy(
            iam.PolicyStatement(
                sid="PassRoleToSageMaker",
                effect=iam.Effect.ALLOW,
                actions=["iam:PassRole"],
                resources=[self.domain_stack.execution_role.role_arn],
                conditions={
                    "StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}
                },
            )
        )

        # ECR permissions - read-only for pulling training images
        github_build_role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRReadAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:DescribeRepositories",
                    "ecr:DescribeImages",
                ],
                resources=["*"],  # GetAuthorizationToken requires *
            )
        )

        # MLflow tracking (if enabled)
        if self.config.enable_mlflow_tracking:
            github_build_role.add_to_policy(
                iam.PolicyStatement(
                    sid="MLflowTracking",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sagemaker-mlflow:*",
                    ],
                    resources=[
                        f"arn:aws:sagemaker:{self.region}:{self.account}:mlflow-tracking-server/*",
                    ],
                )
            )

        return github_build_role

    def _create_github_deploy_role(self) -> iam.Role:
        """
        Create scoped-down IAM role for GitHub Actions DEPLOY workflow
        This role is used by model_deploy repositories to deploy endpoints via CDK
        """

        github_deploy_role = iam.Role(
            self,
            "GitHubActionsDeployRole",
            role_name=f"{self.config.git.oidc_role_github_workflow}-deploy",
            assumed_by=iam.OpenIdConnectPrincipal(
                self.oidc_provider,
                conditions={
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub": f"repo:{self.config.git.target_github_org}/*-model-deploy:*"
                    }
                },
            ),
            description="Scoped role for GitHub Actions DEPLOY workflows (endpoint deployment)",
            max_session_duration=Duration.hours(2),
        )

        # CloudFormation permissions - for CDK deployments
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudFormationDeployment",
                effect=iam.Effect.ALLOW,
                actions=[
                    "cloudformation:CreateStack",
                    "cloudformation:UpdateStack",
                    "cloudformation:DeleteStack",
                    "cloudformation:DescribeStacks",
                    "cloudformation:DescribeStackEvents",
                    "cloudformation:DescribeStackResources",
                    "cloudformation:GetTemplate",
                    "cloudformation:ListStackResources",
                    "cloudformation:DescribeChangeSet",
                    "cloudformation:CreateChangeSet",
                    "cloudformation:ExecuteChangeSet",
                    "cloudformation:DeleteChangeSet",
                    "cloudformation:ListChangeSets",
                    "cloudformation:ValidateTemplate",
                ],
                resources=[
                    f"arn:aws:cloudformation:{self.region}:{self.account}:stack/esg-*/*",
                    f"arn:aws:cloudformation:{self.region}:{self.account}:stack/CDKToolkit/*",
                ],
            )
        )

        # SageMaker endpoint permissions - scoped to deployment operations
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="SageMakerEndpointDeployment",
                effect=iam.Effect.ALLOW,
                actions=[
                    # Model operations
                    "sagemaker:CreateModel",
                    "sagemaker:DescribeModel",
                    "sagemaker:DeleteModel",
                    # Endpoint config operations
                    "sagemaker:CreateEndpointConfig",
                    "sagemaker:DescribeEndpointConfig",
                    "sagemaker:DeleteEndpointConfig",
                    # Endpoint operations
                    "sagemaker:CreateEndpoint",
                    "sagemaker:UpdateEndpoint",
                    "sagemaker:DescribeEndpoint",
                    "sagemaker:DeleteEndpoint",
                    "sagemaker:InvokeEndpoint",
                    # Model package operations (read-only for approved models)
                    "sagemaker:DescribeModelPackage",
                    "sagemaker:ListModelPackages",
                    "sagemaker:DescribeModelPackageGroup",
                    # Tags
                    "sagemaker:AddTags",
                    "sagemaker:ListTags",
                ],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:model/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint-config/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:model-package/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:model-package-group/*",
                ],
            )
        )

        # Create permission boundary for roles created by deploy workflow
        deploy_permission_boundary = iam.ManagedPolicy(
            self,
            "DeployPermissionBoundary",
            managed_policy_name="LLMOpsDeployBoundary",
            description="Permission boundary for roles created by deploy workflow",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sagemaker:*",
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:ListBucket",
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "ecr:GetAuthorizationToken",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                        "kms:Decrypt",
                        "kms:DescribeKey",
                    ],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.DENY,
                    actions=[
                        "iam:*",
                        "organizations:*",
                        "account:*",
                    ],
                    resources=["*"],
                ),
            ],
        )

        # IAM permissions - scoped with permission boundaries
        # Role creation/deletion with permission boundary requirement
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="IAMRoleManagementWithBoundary",
                effect=iam.Effect.ALLOW,
                actions=[
                    "iam:CreateRole",
                    "iam:GetRole",
                    "iam:DeleteRole",
                    "iam:TagRole",
                    "iam:UntagRole",
                ],
                resources=[
                    f"arn:aws:iam::{self.account}:role/esg-*",
                    f"arn:aws:iam::{self.account}:role/ESG*",
                ],
                conditions={
                    "StringEquals": {
                        "iam:PermissionsBoundary": deploy_permission_boundary.managed_policy_arn
                    }
                },
            )
        )

        # Policy attachment - restricted to specific managed policies
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="IAMPolicyAttachment",
                effect=iam.Effect.ALLOW,
                actions=[
                    "iam:AttachRolePolicy",
                    "iam:DetachRolePolicy",
                ],
                resources=[
                    f"arn:aws:iam::{self.account}:role/esg-*",
                    f"arn:aws:iam::{self.account}:role/ESG*",
                ],
                conditions={
                    "ArnLike": {
                        "iam:PolicyARN": [
                            "arn:aws:iam::aws:policy/AmazonSageMaker*",
                            f"arn:aws:iam::{self.account}:policy/esg-*",
                            f"arn:aws:iam::{self.account}:policy/ESG*",
                        ]
                    }
                },
            )
        )

        # Inline policy management
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="IAMInlinePolicyManagement",
                effect=iam.Effect.ALLOW,
                actions=[
                    "iam:PutRolePolicy",
                    "iam:DeleteRolePolicy",
                    "iam:GetRolePolicy",
                    "iam:ListRolePolicies",
                    "iam:ListAttachedRolePolicies",
                ],
                resources=[
                    f"arn:aws:iam::{self.account}:role/esg-*",
                    f"arn:aws:iam::{self.account}:role/ESG*",
                ],
            )
        )

        # PassRole with strict conditions
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="IAMPassRoleRestricted",
                effect=iam.Effect.ALLOW,
                actions=["iam:PassRole"],
                resources=[
                    f"arn:aws:iam::{self.account}:role/esg-*",
                    f"arn:aws:iam::{self.account}:role/ESG*",
                ],
                conditions={
                    "StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}
                },
            )
        )

        # Policy management - restricted to specific prefix
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="IAMPolicyManagement",
                effect=iam.Effect.ALLOW,
                actions=[
                    "iam:CreatePolicy",
                    "iam:GetPolicy",
                    "iam:DeletePolicy",
                    "iam:GetPolicyVersion",
                    "iam:ListPolicyVersions",
                ],
                resources=[
                    f"arn:aws:iam::{self.account}:policy/esg-*",
                    f"arn:aws:iam::{self.account}:policy/ESG*",
                ],
            )
        )

        # S3 permissions - scoped to artifacts and CDK assets buckets
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3BucketAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation",
                    "s3:GetBucketVersioning",
                    "s3:GetBucketPolicy",
                    "s3:PutBucketPolicy",
                ],
                resources=[
                    # Artifacts bucket
                    self.domain_stack.artifacts_bucket.bucket_arn,
                    f"{self.domain_stack.artifacts_bucket.bucket_arn}/*",
                    # CDK assets bucket (created by CDK bootstrap)
                    f"arn:aws:s3:::cdk-*-assets-{self.account}-{self.region}",
                    f"arn:aws:s3:::cdk-*-assets-{self.account}-{self.region}/*",
                ],
            )
        )

        # KMS permissions - for endpoint encryption
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="KMSKeyManagement",
                effect=iam.Effect.ALLOW,
                actions=[
                    "kms:CreateKey",
                    "kms:DescribeKey",
                    "kms:GetKeyPolicy",
                    "kms:PutKeyPolicy",
                    "kms:CreateAlias",
                    "kms:DeleteAlias",
                    "kms:UpdateAlias",
                    "kms:TagResource",
                    "kms:UntagResource",
                    "kms:EnableKeyRotation",
                    "kms:GetKeyRotationStatus",
                    "kms:ScheduleKeyDeletion",
                    "kms:CancelKeyDeletion",
                ],
                resources=[f"arn:aws:kms:{self.region}:{self.account}:key/*"],
                conditions={
                    "StringLike": {"kms:RequestAlias": ["alias/esg-*", "alias/ESG*"]}
                },
            )
        )

        # ECR permissions - for custom inference containers
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:DescribeRepositories",
                    "ecr:DescribeImages",
                    "ecr:ListImages",
                ],
                resources=["*"],  # GetAuthorizationToken requires *
            )
        )

        # CloudWatch Logs - for endpoint logs
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogsForEndpoint",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogGroups",
                    "logs:DescribeLogStreams",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*",
                ],
            )
        )

        # SSM Parameter Store - for configuration parameters
        github_deploy_role.add_to_policy(
            iam.PolicyStatement(
                sid="SSMParameterAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ssm:GetParameter",
                    "ssm:GetParameters",
                    "ssm:PutParameter",
                ],
                resources=[
                    f"arn:aws:ssm:{self.region}:{self.account}:parameter/esg/*",
                    f"arn:aws:ssm:{self.region}:{self.account}:parameter/cdk-bootstrap/*",
                ],
            )
        )

        return github_deploy_role

    def _create_lambda_functions(self) -> dict:
        """Create Lambda functions for repository management with scoped-down permissions"""

        functions = {}

        # Common Lambda environment variables
        common_env = {
            "GITHUB_TOKEN_SECRET_NAME": self.config.git.github_token_secret_name,
            "GITHUB_ORG": self.config.git.target_github_org,  # Used by create-deploy-repository and model-approval-trigger
            "TARGET_GITHUB_ORG": self.config.git.target_github_org,  # Used by sync-repositories
            "TEMPLATE_ORG": self.config.git.template_github_org,
            "TEMPLATE_REPO": self.config.git.template_github_repo,
            "TEMPLATE_FOLDER": self.config.git.template_code_folder,
            "SAGEMAKER_DOMAIN_ID": self.domain_stack.domain.attr_domain_id,
            "SAGEMAKER_EXECUTION_ROLE_ARN": self.domain_stack.execution_role.role_arn,
            "GITHUB_BUILD_ROLE_ARN": "",  # Will be set after role creation
            "GITHUB_DEPLOY_ROLE_ARN": "",  # Will be set after role creation
            "ARTIFACTS_BUCKET": self.domain_stack.artifacts_bucket.bucket_name,
        }

        # Create Lambda execution role with scoped-down permissions
        lambda_role = iam.Role(
            self,
            "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # Scoped permissions for Lambda functions
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="SecretsManagerAccess",
                effect=iam.Effect.ALLOW,
                actions=["secretsmanager:GetSecretValue"],
                resources=[
                    f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:{self.config.git.github_token_secret_name}-*"
                ],
            )
        )

        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="SageMakerReadAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:DescribeDomain",
                    "sagemaker:ListProjects",
                    "sagemaker:DescribeProject",
                    "sagemaker:ListTags",
                    "sagemaker:DescribeModelPackageGroup",
                    "sagemaker:ListModelPackageGroups",
                    "sagemaker:DescribeModelPackage",
                    "sagemaker:ListModelPackages",
                    "sagemaker:ListDomains",
                ],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:domain/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:project/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:model-package-group/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:model-package/*",
                ],
            )
        )

        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="EventBridgeAccess",
                effect=iam.Effect.ALLOW,
                actions=["events:PutEvents"],
                resources=[
                    f"arn:aws:events:{self.region}:{self.account}:event-bus/default"
                ],
            )
        )

        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="LambdaInvokeAccess",
                effect=iam.Effect.ALLOW,
                actions=["lambda:InvokeFunction"],
                resources=[f"arn:aws:lambda:{self.region}:{self.account}:function:*"],
            )
        )

        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="IAMReadAccess",
                effect=iam.Effect.ALLOW,
                actions=["iam:GetRole", "iam:ListRoles"],
                resources=[f"arn:aws:iam::{self.account}:role/*"],
            )
        )

        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ReadAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:ListBucket",
                    "s3:GetBucketLocation",
                    "s3:ListAllMyBuckets",
                ],
                resources=[
                    self.domain_stack.artifacts_bucket.bucket_arn,
                    "arn:aws:s3:::*",
                ],
            )
        )

        # Import the dependency layer construct
        from llmops_sm.constructs.depedency_layer import DependencyLayerConstruct

        dependency_layer_construct = DependencyLayerConstruct(
            self, "DependencyLayerConstruct"
        )
        python_layer = dependency_layer_construct.layer

        # Project status checker
        functions["check_project_status"] = lambda_.Function(
            self,
            "CheckProjectStatusFunction",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("lambda/check-project-status"),
            timeout=Duration.minutes(5),
            environment=common_env,
            role=lambda_role,
            layers=[python_layer],
        )

        # Repository creator
        functions["create_deploy_repo"] = lambda_.Function(
            self,
            "CreateDeployRepoFunction",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("lambda/create-deploy-repository"),
            timeout=Duration.minutes(10),
            environment=common_env,
            role=lambda_role,
            layers=[python_layer],
        )

        # Repository synchronizer
        functions["sync_repositories"] = lambda_.Function(
            self,
            "SyncRepositoriesFunction",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("lambda/sync-repositories"),
            timeout=Duration.minutes(10),
            environment=common_env,
            role=lambda_role,
            layers=[python_layer],
        )

        # Model approval trigger
        sync_function = functions["sync_repositories"]
        functions["model_approval_trigger"] = lambda_.Function(
            self,
            "ModelApprovalTriggerFunction",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("lambda/model-approval-trigger"),
            timeout=Duration.minutes(5),
            environment={
                **common_env,
                "SYNC_REPOSITORIES_FUNCTION_ARN": sync_function.function_arn,
            },
            role=lambda_role,
            layers=[python_layer],
        )

        # Grant secret access to all functions (scoped to specific secret)
        for func in functions.values():
            self.github_secret.grant_read(func)

        # Update environment variables with role ARNs after creation
        for func in functions.values():
            func.add_environment(
                "GITHUB_BUILD_ROLE_ARN", self.github_build_role.role_arn
            )
            func.add_environment(
                "GITHUB_DEPLOY_ROLE_ARN", self.github_deploy_role.role_arn
            )

        return functions

    def _create_step_function(self) -> sfn.StateMachine:
        """Create Step Functions workflow for project setup"""

        # Define failure states
        repo_creation_failed = sfn.Fail(
            self,
            "RepoCreationFailed",
            cause="Repository creation failed",
            error="REPOSITORY_CREATION_FAILED",
        )

        sync_failed = sfn.Fail(
            self,
            "SyncFailed",
            cause="Repository synchronization failed",
            error="REPOSITORY_SYNC_FAILED",
        )

        project_failed = sfn.Fail(
            self,
            "ProjectFailed",
            cause="Project creation failed",
            error="PROJECT_CREATION_FAILED",
        )

        # Define the workflow steps with error handling
        check_project = tasks.LambdaInvoke(
            self,
            "CheckProjectStatus",
            lambda_function=self.lambda_functions["check_project_status"],
            output_path="$.Payload",
        )

        create_build_repo = (
            tasks.LambdaInvoke(
                self,
                "CreateBuildRepository",
                lambda_function=self.lambda_functions["create_deploy_repo"],
                output_path="$.Payload",
            )
            .add_catch(
                repo_creation_failed, errors=["States.ALL"], result_path="$.error"
            )
            .next(
                sfn.Choice(self, "BuildRepoCreationSuccess?")
                .when(
                    sfn.Condition.string_equals("$.status", "FAILED"),
                    repo_creation_failed,
                )
                .otherwise(sfn.Succeed(self, "BuildRepoCreated"))
            )
        )

        # Create the workflow definition
        definition = check_project.next(
            sfn.Choice(self, "ProjectReady?")
            .when(
                sfn.Condition.string_equals("$.status", "COMPLETED"),
                create_build_repo,
            )
            .when(
                sfn.Condition.string_equals("$.status", "PENDING"),
                sfn.Wait(
                    self,
                    "WaitForProject",
                    time=sfn.WaitTime.duration(Duration.minutes(2)),
                ).next(check_project),
            )
            .otherwise(project_failed)
        )

        # Create the state machine
        step_function = sfn.StateMachine(
            self,
            "ProjectSetupStateMachine",
            state_machine_name=self.config.aws.step_function_name,
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.minutes(30),
        )

        return step_function

    def _create_sagemaker_project_template(
        self,
    ) -> tuple[servicecatalog.CloudFormationProduct, servicecatalog.Portfolio]:
        """Create SageMaker Project Template for ESG Benchmarking"""

        # Create IAM role for SageMaker Projects
        sagemaker_projects_role = iam.Role(
            self,
            "SageMakerProjectsRole",
            role_name=f"SageMakerProjectsRole-{self.config.environment}",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.ServicePrincipal("servicecatalog.amazonaws.com"),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AWSServiceCatalogEndUserFullAccess"
                ),
            ],
        )

        # Add scoped permissions for project template
        # S3 - scoped to specific buckets and project-created buckets
        sagemaker_projects_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ScopedAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:CreateBucket",
                    "s3:DeleteBucket",
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation",
                    "s3:GetBucketVersioning",
                    "s3:PutBucketVersioning",
                    "s3:PutBucketTagging",
                    "s3:GetBucketTagging",
                    "s3:PutBucketPublicAccessBlock",
                    "s3:GetBucketPublicAccessBlock",
                    "s3:PutBucketPolicy",
                    "s3:GetBucketPolicy",
                    "s3:DeleteBucketPolicy",
                    "s3:PutEncryptionConfiguration",
                    "s3:GetEncryptionConfiguration",
                    "s3:PutBucketLogging",
                    "s3:GetBucketLogging",
                    "s3:PutLifecycleConfiguration",
                    "s3:GetLifecycleConfiguration",
                ],
                resources=[
                    self.domain_stack.artifacts_bucket.bucket_arn,
                    f"{self.domain_stack.artifacts_bucket.bucket_arn}/*",
                    f"arn:aws:s3:::sagemaker-{self.region}-{self.account}",
                    f"arn:aws:s3:::sagemaker-{self.region}-{self.account}/*",
                    # Allow project-created buckets (they follow pattern: projectname-projectid-*)
                    "arn:aws:s3:::*-p-*-artifacts",
                    "arn:aws:s3:::*-p-*-artifacts/*",
                    "arn:aws:s3:::*-p-*-access-logs",
                    "arn:aws:s3:::*-p-*-access-logs/*",
                ],
            )
        )

        # IAM - scoped PassRole and GetRole
        sagemaker_projects_role.add_to_policy(
            iam.PolicyStatement(
                sid="IAMScopedAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "iam:PassRole",
                    "iam:GetRole",
                ],
                resources=[
                    self.domain_stack.execution_role.role_arn,
                    f"arn:aws:iam::{self.account}:role/service-role/AmazonSageMaker*",
                ],
                conditions={
                    "StringEquals": {
                        "iam:PassedToService": [
                            "sagemaker.amazonaws.com",
                            "events.amazonaws.com",
                        ]
                    }
                },
            )
        )

        # EventBridge - scoped to SageMaker rules
        sagemaker_projects_role.add_to_policy(
            iam.PolicyStatement(
                sid="EventBridgeScopedAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "events:PutRule",
                    "events:PutTargets",
                    "events:DescribeRule",
                    "events:DeleteRule",
                    "events:RemoveTargets",
                    "events:EnableRule",
                    "events:DisableRule",
                    "events:PutEvents",
                ],
                resources=[
                    f"arn:aws:events:{self.region}:{self.account}:rule/sagemaker-*",
                    f"arn:aws:events:{self.region}:{self.account}:event-bus/default",
                ],
            )
        )

        # Lambda - scoped to LLMOps functions
        sagemaker_projects_role.add_to_policy(
            iam.PolicyStatement(
                sid="LambdaScopedAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "lambda:InvokeFunction",
                    "lambda:GetFunction",
                    "lambda:GetFunctionConfiguration",
                ],
                resources=[
                    f"arn:aws:lambda:{self.region}:{self.account}:function:llmops-*",
                    f"arn:aws:lambda:{self.region}:{self.account}:function:*ProjectStatus*",
                    f"arn:aws:lambda:{self.region}:{self.account}:function:*DeployRepo*",
                    f"arn:aws:lambda:{self.region}:{self.account}:function:*SyncRepositories*",
                ],
            )
        )

        # Step Functions - scoped to LLMOps state machines
        sagemaker_projects_role.add_to_policy(
            iam.PolicyStatement(
                sid="StepFunctionsScopedAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "states:StartExecution",
                    "states:DescribeExecution",
                    "states:DescribeStateMachine",
                    "states:ListExecutions",
                ],
                resources=[
                    f"arn:aws:states:{self.region}:{self.account}:stateMachine:llmops-*",
                    f"arn:aws:states:{self.region}:{self.account}:stateMachine:{self.config.aws.step_function_name}",
                    f"arn:aws:states:{self.region}:{self.account}:execution:llmops-*:*",
                    f"arn:aws:states:{self.region}:{self.account}:execution:{self.config.aws.step_function_name}:*",
                ],
            )
        )

        # Secrets Manager - scoped to LLMOps secrets
        sagemaker_projects_role.add_to_policy(
            iam.PolicyStatement(
                sid="SecretsManagerScopedAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:DescribeSecret",
                ],
                resources=[
                    f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:llmops-*",
                    f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:{self.config.git.github_token_secret_name}-*",
                ],
            )
        )

        # Service Catalog - keep broad for project template functionality
        sagemaker_projects_role.add_to_policy(
            iam.PolicyStatement(
                sid="ServiceCatalogAccess",
                effect=iam.Effect.ALLOW,
                actions=["servicecatalog:*"],
                resources=["*"],
            )
        )

        # Create Service Catalog Portfolio
        portfolio = servicecatalog.Portfolio(
            self,
            "ESGBenchmarkingPortfolio",
            display_name="ESG Benchmarking MLOps Templates",
            description="SageMaker project templates for ESG sustainability report generation",
            provider_name="LLMOps Platform",
        )

        # Create the CloudFormation product using CloudFormationProduct
        project_template = servicecatalog.CloudFormationProduct(
            self,
            "ESGBenchmarkingTemplate",
            product_name="ESG Benchmarking MLOps Project",
            description="Complete MLOps pipeline for ESG sustainability report generation using SusGen approach",
            owner="LLMOps Platform",
            product_versions=[
                servicecatalog.CloudFormationProductVersion(
                    product_version_name="1.0",
                    cloud_formation_template=servicecatalog.CloudFormationTemplate.from_asset(
                        "templates/esg-project-template.yaml"
                    ),
                )
            ],
            replace_product_version_ids=True,
        )

        # Associate product with portfolio
        portfolio.add_product(project_template)

        # Add launch constraint - designates IAM role for Service Catalog to assume
        launch_constraint = servicecatalog.CfnLaunchRoleConstraint(
            self,
            "ESGBenchmarkingLaunchConstraint",
            portfolio_id=portfolio.portfolio_id,
            product_id=project_template.product_id,
            role_arn=sagemaker_projects_role.role_arn,
            description="Launch constraint for ESG Benchmarking template",
        )

        # Add regular tag (not TagOptions) for SageMaker visibility
        cdk.Tags.of(project_template).add("sagemaker:studio-visibility", "true")

        # Grant access to SageMaker execution role
        portfolio.give_access_to_role(self.domain_stack.execution_role)

        return project_template, portfolio

    def _create_eventbridge_rules(self):
        """Create EventBridge rules for automation"""

        # Rule for SageMaker project creation events
        project_creation_rule = events.Rule(
            self,
            "ProjectCreationRule",
            rule_name=f"{self.config.aws.eventbridge_rule_prefix}-project-creation",
            description="Trigger workflow when SageMaker project is created",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["AWS API Call via CloudTrail"],
                detail={
                    "eventSource": ["sagemaker.amazonaws.com"],
                    "eventName": ["CreateProject"],
                    "responseElements": {"projectArn": [{"exists": True}]},
                },
            ),
        )

        # Add Step Functions as target with complete event
        project_creation_rule.add_target(
            targets.SfnStateMachine(
                self.step_function,
                input=events.RuleTargetInput.from_object(
                    {
                        "detail": events.EventField.from_path("$.detail"),
                        "source": events.EventField.from_path("$.source"),
                        "detail-type": events.EventField.from_path("$.detail-type"),
                        "time": events.EventField.from_path("$.time"),
                        "region": events.EventField.from_path("$.region"),
                        "account": events.EventField.from_path("$.account"),
                    }
                ),
            )
        )

        # Alternative rule for Service Catalog provisioning (backup trigger)
        service_catalog_rule = events.Rule(
            self,
            "ServiceCatalogProvisioningRule",
            rule_name=f"{self.config.aws.eventbridge_rule_prefix}-service-catalog",
            description="Backup trigger for SageMaker project creation via Service Catalog",
            event_pattern=events.EventPattern(
                source=["aws.servicecatalog"],
                detail_type=["AWS API Call via CloudTrail"],
                detail={
                    "eventSource": ["servicecatalog.amazonaws.com"],
                    "eventName": ["ProvisionProduct"],
                    "responseElements": {"recordDetail": {"status": ["SUCCEEDED"]}},
                },
            ),
        )

        # Add Step Functions as target for Service Catalog rule too
        service_catalog_rule.add_target(
            targets.SfnStateMachine(
                self.step_function,
                input=events.RuleTargetInput.from_object(
                    {
                        "detail": events.EventField.from_path("$.detail"),
                        "source": events.EventField.from_path("$.source"),
                        "detail-type": events.EventField.from_path("$.detail-type"),
                        "time": events.EventField.from_path("$.time"),
                        "region": events.EventField.from_path("$.region"),
                        "account": events.EventField.from_path("$.account"),
                    }
                ),
            )
        )

        # Rule for model approval events
        model_approval_rule = events.Rule(
            self,
            "ModelApprovalRule",
            rule_name=f"{self.config.aws.eventbridge_rule_prefix}-model-approval",
            description="Trigger deployment when model is approved",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Package State Change"],
                detail={"ModelApprovalStatus": ["Approved"]},
            ),
        )

        # Add Lambda as target for model approval
        model_approval_rule.add_target(
            targets.LambdaFunction(self.lambda_functions["model_approval_trigger"])
        )

    def _create_outputs(self):
        """Create CloudFormation outputs"""

        CfnOutput(
            self,
            "GitHubBuildRoleArn",
            value=self.github_build_role.role_arn,
            description="GitHub Actions BUILD IAM Role ARN",
            export_name=f"{self.stack_name}-GitHubBuildRoleArn",
        )

        CfnOutput(
            self,
            "GitHubDeployRoleArn",
            value=self.github_deploy_role.role_arn,
            description="GitHub Actions DEPLOY IAM Role ARN",
            export_name=f"{self.stack_name}-GitHubDeployRoleArn",
        )

        CfnOutput(
            self,
            "GitHubTokenSecretName",
            value=self.github_secret.secret_name,
            description="GitHub Token Secret Name",
            export_name=f"{self.stack_name}-GitHubTokenSecretName",
        )

        CfnOutput(
            self,
            "StepFunctionArn",
            value=self.step_function.state_machine_arn,
            description="Step Functions State Machine ARN",
            export_name=f"{self.stack_name}-StepFunctionArn",
        )

        CfnOutput(
            self,
            "SageMakerDomainId",
            value=self.domain_stack.domain.attr_domain_id,
            description="SageMaker Domain ID (from Domain Stack)",
            export_name=f"{self.stack_name}-SageMakerDomainId",
        )

        CfnOutput(
            self,
            "ServiceCatalogPortfolioId",
            value=self.portfolio.portfolio_id,
            description="Service Catalog Portfolio ID for SageMaker Projects",
            export_name=f"{self.stack_name}-ServiceCatalogPortfolioId",
        )

        CfnOutput(
            self,
            "ServiceCatalogProductId",
            value=self.project_template.product_id,
            description="Service Catalog Product ID for ESG Benchmarking Template",
            export_name=f"{self.stack_name}-ServiceCatalogProductId",
        )
