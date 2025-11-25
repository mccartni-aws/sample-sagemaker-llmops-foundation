# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: MIT-0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import boto3
from botocore.exceptions import ClientError
from logging import Logger
from config.constants import DEFAULT_DEPLOYMENT_REGION, MODEL_PACKAGE_GROUP_NAME

"""Initialize Logger class for ESG model deployment"""
logger = Logger(name="esg_deploy_stack")

"""Initialize boto3 SDK resources"""
sm_client = boto3.client("sagemaker", region_name=DEFAULT_DEPLOYMENT_REGION)


def get_approved_package():
    """Gets the latest approved ESG model package for a model package group.

    This function specifically looks for approved ESG sustainability models
    in the model registry and returns the most recently approved model.

    Returns:
        The SageMaker Model Package ARN for the latest approved ESG model.

    Raises:
        Exception: If no approved ESG models are found in the model package group.
    """
    try:
        logger.info(
            f"Searching for approved ESG models in group: {MODEL_PACKAGE_GROUP_NAME}"
        )

        # Get the latest approved ESG model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",  # Get most recent first
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug(
                f"Getting more ESG model packages for token: {response['NextToken']}"
            )
            response = sm_client.list_model_packages(
                ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no approved ESG models found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ESG ModelPackage found for ModelPackageGroup: {MODEL_PACKAGE_GROUP_NAME}. "
                f"Please ensure at least one ESG model has been approved in the SageMaker Model Registry."
            )
            logger.error(error_message)
            raise Exception(error_message)

        # Return the latest approved ESG model package ARN
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        model_package_version = approved_packages[0].get(
            "ModelPackageVersion", "Unknown"
        )

        logger.info(
            f"✅ Identified the latest approved ESG model package: {model_package_arn}"
        )
        logger.info(f"📊 ESG Model Package Version: {model_package_version}")
        logger.info(
            f"🌱 ESG model is ready for sustainability report generation deployment"
        )

        return model_package_arn

    except ClientError as e:
        error_message = f"AWS API Error while retrieving ESG model packages: {e.response['Error']['Message']}"
        logger.error(error_message)
        raise Exception(error_message)
    except Exception as e:
        error_message = (
            f"Unexpected error while retrieving approved ESG model packages: {str(e)}"
        )
        logger.error(error_message)
        raise Exception(error_message)
