"""
SageMaker Pipeline Runner for ESG Benchmarking

This script runs the ESG model training pipeline with the specified configuration.
"""

import argparse
import json
import logging
import sys
from importlib import import_module
import os
import boto3
from sagemaker.session import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_mlflow_tracking_server_info():
    """
    Print information about the MLFlow tracking server for visibility.
    """
    try:
        # Get current region
        session = boto3.Session()
        region = session.region_name or "us-east-1"

        # Try to get environment from environment variables or assume 'dev'
        environment = os.environ.get("ENVIRONMENT", "dev")

        # Create SageMaker client
        sagemaker_client = boto3.client("sagemaker", region_name=region)

        # Expected tracking server name from CDK stack
        tracking_server_name = f"llmops-mlflow-{environment}"

        logger.info("=" * 60)
        logger.info("MLFLOW TRACKING SERVER INFORMATION")
        logger.info("=" * 60)
        logger.info(f"Environment: {environment}")
        logger.info(f"Region: {region}")
        logger.info(f"Expected MLFlow Server Name: {tracking_server_name}")

        mlflow_tracking_arn = os.environ.get("MLFLOW_TRACKING_ARN", "")

        if mlflow_tracking_arn is None:
            # Try to describe the tracking server
            try:
                response = sagemaker_client.describe_mlflow_tracking_server(
                    TrackingServerName=tracking_server_name
                )
                tracking_server_url = response["TrackingServerUrl"]
                tracking_server_arn = response["TrackingServerArn"]
                server_status = response["TrackingServerStatus"]

                logger.info(f"✅ MLFlow Server Status: {server_status}")
                logger.info(f"✅ MLFlow Server URL: {tracking_server_url}")
                logger.info(f"✅ MLFlow Server ARN: {tracking_server_arn}")
                logger.info("✅ MLFlow tracking is available for this pipeline run!")

            except sagemaker_client.exceptions.ResourceNotFound:
                logger.warning(
                    f"⚠️  MLFlow tracking server '{tracking_server_name}' not found"
                )

                # Try to list all tracking servers as fallback
                try:
                    response = sagemaker_client.list_mlflow_tracking_servers()
                    servers = response.get("TrackingServerSummaries", [])

                    if servers:
                        logger.info("📋 Available MLFlow tracking servers:")
                        for server in servers:
                            server_name = server["TrackingServerName"]
                            server_status = server["TrackingServerStatus"]
                            logger.info(f"   - {server_name} (Status: {server_status})")
                    else:
                        logger.warning(
                            "❌ No MLFlow tracking servers found in this region"
                        )
                except Exception as e:
                    logger.error(f"❌ Error listing MLFlow tracking servers: {str(e)}")
        else:
            logger.info(
                f"Found MLFlow Server with name: {tracking_server_name} in GitHub Env Variables"
            )

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error getting MLFlow tracking server info: {str(e)}")


def get_pipeline(module_name, role, pipeline_name, **kwargs):
    """
    Gets a SageMaker Pipeline instance from the specified module.

    Args:
        module_name (str): Name of the module containing the pipeline
        role (str): SageMaker execution role ARN
        pipeline_name (str): Name of the pipeline
        **kwargs: Additional keyword arguments for pipeline configuration

    Returns:
        sagemaker.workflow.pipeline.Pipeline: The pipeline instance
    """
    # Security: Whitelist of allowed pipeline modules to prevent arbitrary code execution
    allowed_modules = {
        "training.pipeline",
        "ml_pipelines.training.pipeline",
        "esg_benchmarking.training.pipeline",
    }

    if module_name not in allowed_modules:
        raise ValueError(
            f"Module '{module_name}' is not in the allowed modules list: {allowed_modules}"
        )

    try:
        pipeline_module = import_module(module_name)

        return pipeline_module.get_pipeline(
            role=role, pipeline_name=pipeline_name, **kwargs
        )
    except Exception as e:
        logger.warning(f"Error importing pipeline module {module_name}: {str(e)}")
        raise


def run_pipeline(pipeline, role_arn, tags=None):
    """
    Executes the SageMaker pipeline.

    Args:
        pipeline: SageMaker Pipeline instance
        role_arn (str): SageMaker execution role ARN
        tags (list): List of tags to apply to the pipeline

    Returns:
        dict: Pipeline execution response
    """
    try:
        # Upsert the pipeline (create or update)
        logger.info(f"Upserting pipeline: {pipeline.name}")
        pipeline.upsert(role_arn=role_arn, tags=tags)

        # Start pipeline execution
        logger.info(f"Starting pipeline execution: {pipeline.name}")
        execution = pipeline.start()

        logger.info(f"Pipeline execution started successfully")
        logger.info(f"Execution ARN: {execution.arn}")

        return {
            "pipeline_name": pipeline.name,
            "execution_arn": execution.arn,
            "status": "Started",
        }

    except Exception as e:
        logger.warning(f"Error running pipeline: {str(e)}")
        raise


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Run SageMaker Pipeline")

    parser.add_argument(
        "--module-name",
        type=str,
        required=True,
        help="Python module name containing the pipeline definition",
    )

    parser.add_argument(
        "--role-arn", type=str, required=True, help="SageMaker execution role ARN"
    )

    parser.add_argument(
        "--tags",
        type=str,
        default="[]",
        help="JSON string of tags to apply to the pipeline",
    )

    parser.add_argument(
        "--kwargs",
        type=str,
        default="{}",
        help="JSON string of additional keyword arguments for pipeline configuration",
    )

    args = parser.parse_args()

    try:
        # Parse tags and kwargs
        tags = json.loads(args.tags)
        kwargs = json.loads(args.kwargs)

        logger.info(f"Running pipeline with module: {args.module_name}")
        logger.info(f"Role ARN: {args.role_arn}")
        logger.info(f"Tags: {tags}")
        logger.info(f"Additional kwargs: {kwargs}")

        # Print MLFlow tracking server information
        print_mlflow_tracking_server_info()

        # Get pipeline instance
        pipeline = get_pipeline(
            module_name=args.module_name, role=args.role_arn, **kwargs
        )

        # Run pipeline - pass role_arn explicitly
        result = run_pipeline(pipeline, role_arn=args.role_arn, tags=tags)

        logger.info("Pipeline execution completed successfully")
        logger.info(f"Result: {json.dumps(result, indent=2)}")

        return 0

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON arguments: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
