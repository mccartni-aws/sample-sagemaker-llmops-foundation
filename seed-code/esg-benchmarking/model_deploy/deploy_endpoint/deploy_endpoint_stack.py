# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: MIT-0
#
# See license text in the original file.

import os
from datetime import datetime, timezone

from aws_cdk import (
    Aws,
    Stack,
    Tags,
    CfnTag,
    aws_iam as iam,
    aws_kms as kms,
    aws_s3_assets as s3_assets,
    aws_sagemaker as sagemaker,
    BundlingOptions,
    DockerImage,
)
import constructs

from config.constants import (
    PROJECT_NAME,
    PROJECT_ID,
    MODEL_PACKAGE_GROUP_NAME,
    DEPLOY_ACCOUNT,
    ECR_REPO_ARN,
    MODEL_BUCKET_ARN,  # kept for compatibility (not used directly here)
    SAGEMAKER_DOMAIN_ARN,
)

# DJL image for GPU
DJL_GPU = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.31.0-lmi13.0.0-cu124"
)

# Families that use NVMe instance storage (SageMaker doesn't allow kms_key_id on these)
NVME_FAMILIES = {"g5", "p4d", "p4de", "p3dn", "inf2", "trn1", "trn1n"}


def _is_nvme_instance(instance_type: str) -> bool:
    # e.g. "ml.g5.2xlarge" -> "g5"
    parts = instance_type.split(".")
    family = parts[1] if len(parts) >= 3 else ""
    return family in NVME_FAMILIES


class DeployEndpointStack(Stack):
    def __init__(
        self,
        scope: constructs,
        id: str,
        adapter_s3_uri: str | None = None,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)

        # Allow env overrides (so GitHub Actions `env.ADAPTER_S3_URI` work)
        adapter_s3_uri = adapter_s3_uri or os.environ.get("ADAPTER_S3_URI", "")

        # === Tags ===
        Tags.of(self).add("sagemaker:project-id", PROJECT_ID)
        Tags.of(self).add("sagemaker:project-name", PROJECT_NAME)
        Tags.of(self).add("sagemaker:deployment-stage", Stack.of(self).stack_name)
        Tags.of(self).add("sagemaker:domain-arn", SAGEMAKER_DOMAIN_ARN)
        Tags.of(self).add("UseCase", "ESG-Sustainability-Reports")
        Tags.of(self).add("ModelType", "Text-Generation")
        Tags.of(self).add("Domain", "Environmental-Social-Governance")

        # === Execution Role ===
        model_execution_policy = iam.ManagedPolicy(
            self,
            "ESGModelExecutionPolicy",
            description="Policy for ESG model endpoint execution",
            document=iam.PolicyDocument(
                statements=[
                    # Read/write to artifacts bucket (where adapters/checkpoints may live)
                    iam.PolicyStatement(
                        actions=[
                            "s3:GetObject",
                            "s3:PutObject",
                            "s3:DeleteObject",
                            "s3:GetObjectVersion",
                            "s3:ListBucket",
                            "s3:GetBucketLocation",
                        ],
                        resources=[
                            f"arn:aws:s3:::llmops-sm-artifacts-{DEPLOY_ACCOUNT}-{Aws.REGION}",
                            f"arn:aws:s3:::llmops-sm-artifacts-{DEPLOY_ACCOUNT}-{Aws.REGION}/*",
                        ],
                    ),
                    # KMS access for the region/account
                    iam.PolicyStatement(
                        actions=[
                            "kms:Encrypt",
                            "kms:ReEncrypt*",
                            "kms:GenerateDataKey*",
                            "kms:Decrypt",
                            "kms:DescribeKey",
                        ],
                        resources=[f"arn:aws:kms:{Aws.REGION}:{DEPLOY_ACCOUNT}:key/*"],
                    ),
                    # Read CDK (gitactions) assets bucket so the model container can fetch the artifact
                    iam.PolicyStatement(
                        actions=[
                            "s3:GetObject",
                            "s3:GetObjectVersion",
                            "s3:ListBucket",
                        ],
                        resources=[
                            f"arn:aws:s3:::cdk-gitactions-assets-{DEPLOY_ACCOUNT}-{Aws.REGION}",
                            f"arn:aws:s3:::cdk-gitactions-assets-{DEPLOY_ACCOUNT}-{Aws.REGION}/*",
                        ],
                    ),
                ]
            ),
        )

        if ECR_REPO_ARN:
            model_execution_policy.add_statements(
                iam.PolicyStatement(
                    actions=["ecr:Get*"],
                    effect=iam.Effect.ALLOW,
                    resources=[ECR_REPO_ARN],
                )
            )

        model_execution_role = iam.Role(
            self,
            "ESGModelExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                model_execution_policy,
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
            ],
        )

        # === Timestamped Names ===
        now = datetime.now().replace(tzinfo=timezone.utc)
        timestamp = now.strftime("%Y%m%d%H%M%S")

        model_name = f"{MODEL_PACKAGE_GROUP_NAME}-{timestamp}"
        endpoint_config_name = f"{MODEL_PACKAGE_GROUP_NAME}-ec-{timestamp}"
        endpoint_name = f"{MODEL_PACKAGE_GROUP_NAME}-{PROJECT_NAME}-endpoint"

        # === Instance and container ===
        instance_type = "ml.g5.2xlarge"
        inference_image_uri = DJL_GPU

        # Package ./src into a gzipped tarball as a CDK asset (uploaded to cdk-gitactions-assets-<acct>-<region>)
        djl_model_asset = s3_assets.Asset(
            self,
            "DJLModelArchive",
            path="src",
            bundling=BundlingOptions(
                image=DockerImage.from_registry("debian:bullseye-slim"),
                command=[
                    "bash",
                    "-lc",
                    "tar -czf /asset-output/model.tar.gz -C /asset-input .",
                ],
            ),
        )

        # === SageMaker Model (uses DJL runtime) ===
        model = sagemaker.CfnModel(
            self,
            "ESGModel",
            execution_role_arn=model_execution_role.role_arn,
            model_name=model_name,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=inference_image_uri,
                model_data_url=f"s3://{djl_model_asset.s3_bucket_name}/{djl_model_asset.s3_object_key}",
                environment={
                    "MODEL_LOADING_TIMEOUT": "600000",
                    "ADAPTER_S3_URI": adapter_s3_uri,
                    "BASE_MODEL": "mistralai/Mistral-7B-Instruct-v0.3",
                    "LOAD_IN_4BIT": "true",
                },
            ),
        )
        # allow the role to read the uploaded asset
        djl_model_asset.grant_read(model_execution_role)

        # === KMS key (only for non-NVMe/EBS-backed instance families) ===
        kms_key = kms.Key(
            self,
            "ESGEndpointKMSKey",
            description="KMS key for endpoint data (only for non-NVMe families)",
            enable_key_rotation=True,
        )

        prod_variant = sagemaker.CfnEndpointConfig.ProductionVariantProperty(
            model_name=model.model_name,
            variant_name="AllTraffic",
            initial_instance_count=1,
            initial_variant_weight=1.0,
            instance_type=instance_type,
        )

        endpoint_cfg_kwargs = {
            "endpoint_config_name": endpoint_config_name,
            "production_variants": [prod_variant],
            "tags": [
                CfnTag(key="OptimizedFor", value="Sustainability-Report-Generation")
            ],
        }
        if not _is_nvme_instance(instance_type):
            endpoint_cfg_kwargs["kms_key_id"] = kms_key.key_id

        endpoint_config = sagemaker.CfnEndpointConfig(
            self, "ESGEndpointConfig", **endpoint_cfg_kwargs
        )
        endpoint_config.add_depends_on(model)

        endpoint = sagemaker.CfnEndpoint(
            self,
            "ESGEndpoint",
            endpoint_config_name=endpoint_config.endpoint_config_name,
            endpoint_name=endpoint_name,
            tags=[
                CfnTag(key="Environment", value="Production"),
                CfnTag(key="Purpose", value="Sustainability-Report-Generation"),
            ],
        )
        endpoint.add_depends_on(endpoint_config)

        # === References for app.py/tests ===
        self.endpoint = endpoint
        self.endpoint_name = endpoint_name
        self.model_execution_role = model_execution_role
