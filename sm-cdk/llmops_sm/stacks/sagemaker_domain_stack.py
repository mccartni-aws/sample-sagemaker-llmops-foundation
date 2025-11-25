"""
SageMaker Domain Stack
Creates SageMaker Studio Domain and associated resources
"""

from typing import Dict, Any
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_sagemaker as sagemaker,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_s3 as s3,
    CfnOutput,
    Duration,
    RemovalPolicy,
)
from constructs import Construct

from llmops_sm.config import PlatformConfig


class SageMakerDomainStack(Stack):
    """Stack for creating SageMaker Studio Domain and related resources"""

    def __init__(
        self, scope: Construct, construct_id: str, config: PlatformConfig, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.config = config

        # Create VPC for SageMaker Domain (or use default)
        self.vpc = self._create_or_get_vpc()

        # Create S3 bucket for SageMaker artifacts
        self.artifacts_bucket = self._create_artifacts_bucket()

        # Create S3 bucket for ESG data
        self.esg_data_bucket = self._create_esg_data_bucket()

        # Create SageMaker execution role
        self.execution_role = self._create_sagemaker_execution_role()

        # Create SageMaker Domain
        self.domain = self._create_sagemaker_domain()

        # Create default user profile
        self.user_profile = self._create_user_profile()

        # Create project templates
        self._create_project_templates()

        # Create outputs
        self._create_outputs()

    def _create_or_get_vpc(self) -> ec2.IVpc:
        """Create VPC or use default VPC for SageMaker Domain"""

        if self.config.sagemaker.use_dedicated_vpc:
            # Create dedicated VPC with NAT Gateway for secure internet access
            vpc = ec2.Vpc(
                self,
                "SageMakerVPC",
                vpc_name=f"{self.config.sagemaker.domain_name}-vpc",
                max_azs=2,
                nat_gateways=1,
                subnet_configuration=[
                    ec2.SubnetConfiguration(
                        name="Public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                    ),
                    ec2.SubnetConfiguration(
                        name="Private",
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                        cidr_mask=24,
                    ),
                ],
            )
        else:
            # Use default VPC (simpler setup)
            vpc = ec2.Vpc.from_lookup(self, "DefaultVPC", is_default=True)

        return vpc

    def _create_artifacts_bucket(self) -> s3.Bucket:
        """Create S3 bucket for SageMaker artifacts"""

        bucket = s3.Bucket(
            self,
            "ArtifactsBucket",
            bucket_name=f"{self.config.sagemaker.artifacts_bucket_prefix}-{self.account}-{self.region}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=(
                RemovalPolicy.DESTROY
                if self.config.environment == "dev"
                else RemovalPolicy.RETAIN
            ),
            auto_delete_objects=True if self.config.environment == "dev" else False,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteOldVersions",
                    enabled=True,
                    noncurrent_version_expiration=Duration.days(30),
                ),
                s3.LifecycleRule(
                    id="ArchiveOldObjects",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30),
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90),
                        ),
                    ],
                ),
            ],
        )

        return bucket

    def _create_esg_data_bucket(self) -> s3.Bucket:
        """Create S3 bucket for ESG data storage"""

        bucket = s3.Bucket(
            self,
            "ESGDataBucket",
            bucket_name=f"esg-data-{self.account}-{self.region}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=(
                RemovalPolicy.DESTROY
                if self.config.environment == "dev"
                else RemovalPolicy.RETAIN
            ),
            auto_delete_objects=True if self.config.environment == "dev" else False,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteOldVersions",
                    enabled=True,
                    noncurrent_version_expiration=Duration.days(90),
                ),
                s3.LifecycleRule(
                    id="ArchiveOldData",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(90),
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(365),
                        ),
                    ],
                ),
            ],
        )

        return bucket

    def _create_sagemaker_execution_role(self) -> iam.Role:
        """Create IAM role for SageMaker execution with comprehensive permissions"""

        # Create the execution role with proper trust relationship
        role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name=self.config.sagemaker.execution_role_name,
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="SageMaker execution role for LLMOps platform",
        )

        # Update the trust policy to include sts:SetSourceIdentity
        role.assume_role_policy.add_statements(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                principals=[iam.ServicePrincipal("sagemaker.amazonaws.com")],
                actions=["sts:AssumeRole", "sts:SetSourceIdentity"],
            )
        )

        # Add AWS managed policies as specified
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonEC2ContainerRegistryReadOnly"
            )
        )
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AWSServiceCatalogEndUserFullAccess"
            )
        )
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchEventsFullAccess")
        )

        # Add MLflow-specific permissions for the tracking server
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sagemaker-mlflow:*"],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:mlflow-tracking-server/llmops-mlflow-{self.config.environment}"
                ],
            )
        )

        # Add general MLflow permissions
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker-mlflow:AccessUI",
                    "sagemaker-mlflow:CreateExperiment",
                    "sagemaker-mlflow:CreateRun",
                    "sagemaker-mlflow:UpdateRun",
                    "sagemaker-mlflow:DeleteRun",
                    "sagemaker-mlflow:GetExperiment",
                    "sagemaker-mlflow:GetRun",
                    "sagemaker-mlflow:SearchExperiments",
                    "sagemaker-mlflow:SearchRuns",
                    "sagemaker-mlflow:LogMetric",
                    "sagemaker-mlflow:LogParam",
                    "sagemaker-mlflow:LogArtifact",
                    "sagemaker-mlflow:GetArtifact",
                    "sagemaker-mlflow:ListArtifacts",
                ],
                resources=["*"],
            )
        )

        # Add CloudWatch Logs permissions - scoped to SageMaker logs
        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogsScopedAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*",
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*:log-stream:*",
                ],
            )
        )

        # ECR - GetAuthorizationToken requires wildcard (AWS API limitation)
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRAuthorizationToken",
                effect=iam.Effect.ALLOW,
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],  # AWS API requirement - cannot be scoped
            )
        )

        # ECR - scope other operations to SageMaker and LLMOps repositories
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRRepositoryAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                resources=[
                    f"arn:aws:ecr:{self.region}:{self.account}:repository/sagemaker-*",
                    f"arn:aws:ecr:{self.region}:{self.account}:repository/llmops-*",
                    f"arn:aws:ecr:{self.region}:{self.account}:repository/*",  # Allow any repo for flexibility
                ],
            )
        )

        # Bedrock - scope to specific model families used in project
        role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockModelAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    # Scope to common model families - add more as needed
                    f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-*",
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-*",
                    f"arn:aws:bedrock:{self.region}::foundation-model/mistral.*",
                    f"arn:aws:bedrock:{self.region}::foundation-model/meta.llama*",
                    f"arn:aws:bedrock:{self.region}::foundation-model/ai21.*",
                    f"arn:aws:bedrock:{self.region}::foundation-model/cohere.*",
                ],
            )
        )

        # Bedrock - list operations (requires wildcard but read-only)
        role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockListModels",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:GetFoundationModel",
                    "bedrock:ListFoundationModels",
                ],
                resources=["*"],  # Required for list operations
            )
        )

        # Add specific S3 permissions for artifacts bucket
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject*",
                    "s3:GetBucket*",
                    "s3:List*",
                    "s3:DeleteObject*",
                    "s3:PutObject",
                    "s3:PutObjectLegalHold",
                    "s3:PutObjectRetention",
                    "s3:PutObjectTagging",
                    "s3:PutObjectVersionTagging",
                    "s3:Abort*",
                ],
                resources=[
                    f"arn:aws:s3:::llmops-sm-artifacts-{self.account}-{self.region}",
                    f"arn:aws:s3:::llmops-sm-artifacts-{self.account}-{self.region}/*",
                ],
            )
        )

        # Add specific S3 permissions for ESG data bucket
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject*",
                    "s3:GetBucket*",
                    "s3:List*",
                    "s3:DeleteObject*",
                    "s3:PutObject",
                    "s3:PutObjectLegalHold",
                    "s3:PutObjectRetention",
                    "s3:PutObjectTagging",
                    "s3:PutObjectVersionTagging",
                    "s3:Abort*",
                ],
                resources=[
                    f"arn:aws:s3:::esg-data-{self.account}-{self.region}",
                    f"arn:aws:s3:::esg-data-{self.account}-{self.region}/*",
                ],
            )
        )

        # Add IAM PassRole permission with condition
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["iam:PassRole"],
                resources=[
                    f"arn:aws:iam::{self.account}:role/{self.config.sagemaker.execution_role_name}"
                ],
                conditions={
                    "StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}
                },
            )
        )

        # Add Service Catalog permissions for SageMaker Projects
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "servicecatalog:ListAcceptedPortfolioShares",
                    "servicecatalog:ListConstraintsForPortfolio",
                    "servicecatalog:ListLaunchPaths",
                    "servicecatalog:ListPortfolios",
                    "servicecatalog:ListPrincipalsForPortfolio",
                    "servicecatalog:ListProvisioningArtifacts",
                    "servicecatalog:ListTagOptionsForResource",
                    "servicecatalog:ProvisionProduct",
                    "servicecatalog:SearchProducts",
                    "servicecatalog:TerminateProvisionedProduct",
                    "servicecatalog:UpdateProvisionedProduct",
                ],
                resources=["*"],
            )
        )

        return role

    def _create_sagemaker_domain(self) -> sagemaker.CfnDomain:
        """Create SageMaker Studio Domain"""

        # Get subnet IDs for the domain
        subnet_ids = [subnet.subnet_id for subnet in self.vpc.private_subnets[:2]]
        if not subnet_ids:
            # Fallback to public subnets if no private subnets
            subnet_ids = [subnet.subnet_id for subnet in self.vpc.public_subnets[:2]]

        # Create security group for SageMaker Domain
        security_group = ec2.SecurityGroup(
            self,
            "SageMakerDomainSecurityGroup",
            vpc=self.vpc,
            description="Security group for SageMaker Studio Domain",
            allow_all_outbound=True,
        )

        # Allow HTTPS traffic within the security group
        security_group.add_ingress_rule(
            peer=security_group,
            connection=ec2.Port.tcp(443),
            description="HTTPS within security group",
        )

        # Create the SageMaker Domain
        domain = sagemaker.CfnDomain(
            self,
            "SageMakerDomain",
            auth_mode="IAM",
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=self.execution_role.role_arn,
                security_groups=[security_group.security_group_id],
                sharing_settings=sagemaker.CfnDomain.SharingSettingsProperty(
                    notebook_output_option="Allowed",
                    s3_output_path=f"s3://{self.artifacts_bucket.bucket_name}/shared-notebooks",
                ),
                jupyter_server_app_settings=sagemaker.CfnDomain.JupyterServerAppSettingsProperty(
                    default_resource_spec=sagemaker.CfnDomain.ResourceSpecProperty(
                        instance_type="system"  # Only 'system' is supported, no image ARN needed
                    )
                ),
                kernel_gateway_app_settings=sagemaker.CfnDomain.KernelGatewayAppSettingsProperty(
                    default_resource_spec=sagemaker.CfnDomain.ResourceSpecProperty(
                        instance_type=self.config.sagemaker.default_instance_type,  # ml.m5.large is OK for KernelGateway
                        # Use SageMaker's managed data science image
                        # Note: SageMaker managed images don't require explicit ARN specification
                        # The service will use the appropriate image for the region automatically
                    )
                ),
                studio_web_portal="ENABLED",
            ),
            domain_name=self.config.sagemaker.domain_name,
            subnet_ids=subnet_ids,
            vpc_id=self.vpc.vpc_id,
            # app_network_access_type="VpcOnly",  # More secure but blocks internet
            app_network_access_type="PublicInternetOnly",  # Allows internet access
            domain_settings=sagemaker.CfnDomain.DomainSettingsProperty(
                execution_role_identity_config="USER_PROFILE_NAME",
                security_group_ids=[security_group.security_group_id],
            ),
        )

        return domain

    def _create_user_profile(self) -> sagemaker.CfnUserProfile:
        """Create default user profile for the domain"""

        user_profile = sagemaker.CfnUserProfile(
            self,
            "DefaultUserProfile",
            domain_id=self.domain.attr_domain_id,
            user_profile_name="default-user",
            user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                execution_role=self.execution_role.role_arn,
                jupyter_server_app_settings=sagemaker.CfnUserProfile.JupyterServerAppSettingsProperty(
                    default_resource_spec=sagemaker.CfnUserProfile.ResourceSpecProperty(
                        instance_type="system"  # Only 'system' is supported, no image ARN needed
                    )
                ),
            ),
        )

        return user_profile

    def _create_project_templates(self):
        """Create SageMaker project templates for LLMOps"""

        # Create service catalog portfolio (required for project templates)
        # Note: This is a simplified version. In production, you might want to use
        # AWS Service Catalog constructs or create templates separately

        project_template_role = iam.Role(
            self,
            "ProjectTemplateRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AWSCodeCommitFullAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AWSCodeBuildDeveloperAccess"
                ),
            ],
        )

        # Add custom permissions for CodePipeline (since the managed policy doesn't exist)
        project_template_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "codepipeline:CreatePipeline",
                    "codepipeline:DeletePipeline",
                    "codepipeline:GetPipeline",
                    "codepipeline:GetPipelineExecution",
                    "codepipeline:GetPipelineState",
                    "codepipeline:ListPipelineExecutions",
                    "codepipeline:ListPipelines",
                    "codepipeline:StartPipelineExecution",
                    "codepipeline:StopPipelineExecution",
                    "codepipeline:UpdatePipeline",
                    "codepipeline:GetJobDetails",
                    "codepipeline:PutJobFailureResult",
                    "codepipeline:PutJobSuccessResult",
                    "codepipeline:PutApprovalResult",
                    "codepipeline:GetThirdPartyJobDetails",
                    "codepipeline:PutThirdPartyJobFailureResult",
                    "codepipeline:PutThirdPartyJobSuccessResult",
                ],
                resources=["*"],
            )
        )

        # Add custom permissions for GitHub integration
        project_template_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "events:PutEvents",
                    "events:PutRule",
                    "events:PutTargets",
                    "events:DeleteRule",
                    "events:RemoveTargets",
                    "lambda:InvokeFunction",
                    "stepfunctions:StartExecution",
                    "stepfunctions:DescribeExecution",
                    "stepfunctions:StopExecution",
                ],
                resources=["*"],
            )
        )

    def _create_outputs(self):
        """Create CloudFormation outputs"""

        CfnOutput(
            self,
            "DomainId",
            value=self.domain.attr_domain_id,
            description="SageMaker Studio Domain ID",
            export_name=f"{self.stack_name}-DomainId",
        )

        CfnOutput(
            self,
            "DomainUrl",
            value=self.domain.attr_url,
            description="SageMaker Studio Domain URL",
            export_name=f"{self.stack_name}-DomainUrl",
        )

        CfnOutput(
            self,
            "ExecutionRoleArn",
            value=self.execution_role.role_arn,
            description="SageMaker Execution Role ARN",
            export_name=f"{self.stack_name}-ExecutionRoleArn",
        )

        CfnOutput(
            self,
            "ArtifactsBucketName",
            value=self.artifacts_bucket.bucket_name,
            description="S3 Artifacts Bucket Name",
            export_name=f"{self.stack_name}-ArtifactsBucketName",
        )

        CfnOutput(
            self,
            "UserProfileName",
            value=self.user_profile.user_profile_name,
            description="Default User Profile Name",
            export_name=f"{self.stack_name}-UserProfileName",
        )

        CfnOutput(
            self,
            "ESGDataBucketName",
            value=self.esg_data_bucket.bucket_name,
            description="S3 ESG Data Bucket Name",
            export_name=f"{self.stack_name}-ESGDataBucketName",
        )
