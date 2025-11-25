"""
Dependency layer using stable CDK APIs with Docker-based bundling.
This construct uses aws_lambda.LayerVersion with BundlingOptions for dependency management.
"""

from constructs import Construct
from aws_cdk import (
    aws_lambda as _lambda,
    BundlingOptions,
    DockerImage,
)
import os


class DependencyLayerConstruct(Construct):
    """
    Creates a Lambda layer with Python dependencies using stable CDK APIs.

    This construct uses aws_lambda.LayerVersion with Docker bundling which:
    - Bundles dependencies in a Lambda-compatible Docker container
    - Handles platform-specific builds (x86_64/arm64)
    - Works with Docker alternatives like Finch
    - Uses only stable CDK APIs (no alpha packages required)
    """

    def __init__(self, scope: Construct, id: str) -> None:
        super().__init__(scope, id)

        # Get the path to the layer directory containing requirements.txt
        project_root = os.getcwd()
        layer_dir = os.path.join(project_root, "layers", "python-layer")

        # Create layer using stable LayerVersion with Docker bundling
        # This approach bundles dependencies during CDK synthesis
        self.layer = _lambda.LayerVersion(
            self,
            "DependencyLayer",
            code=_lambda.Code.from_asset(
                layer_dir,
                bundling=BundlingOptions(
                    image=DockerImage.from_registry(
                        "public.ecr.aws/sam/build-python3.11:latest"
                    ),
                    command=[
                        "bash",
                        "-c",
                        " && ".join(
                            [
                                "pip install -r requirements.txt -t /asset-output/python",
                                "find /asset-output -type d -name '__pycache__' -exec rm -rf {} + || true",
                                "find /asset-output -type f -name '*.pyc' -delete || true",
                            ]
                        ),
                    ],
                ),
            ),
            compatible_runtimes=[
                _lambda.Runtime.PYTHON_3_9,
                _lambda.Runtime.PYTHON_3_10,
                _lambda.Runtime.PYTHON_3_11,
            ],
            layer_version_name="ml-ops-smus-dependency-layer",
            description=(
                "Python dependencies for LLMOps Lambda functions "
                "(boto3, requests, cryptography, PyNaCl)"
            ),
        )
