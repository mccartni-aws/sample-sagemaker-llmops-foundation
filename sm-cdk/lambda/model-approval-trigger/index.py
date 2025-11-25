import json
import boto3
import requests
import os
from datetime import datetime

# Name of the GitHub Actions workflow file that handles model deployment
# When a deploy repository is created, it's populated with seed code from the template
# which includes 'deploy_model_pipeline.yml'. This workflow is triggered when a
# SageMaker model is approved, and it handles:
#   - Creating/Updating SageMaker endpoints
#   - Model deployment and testing
#   - A/B testing and validation
#   - Other deployment-related tasks
# IMPORTANT: If you modify the workflow filename in your template or deploy repositories,
#           update this value accordingly

WORKFLOW_FILENAME = "deploy_model_pipeline.yml"


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def extract_project_info_from_model_package_group(model_package_group_name):
    """Extract project information from model package group name and tags"""
    try:
        # For SageMaker Studio Domain, model package group names typically follow patterns like:
        # "esg-benchmarking-{project-id}-models" or "{template-name}-{project-id}-models"

        # Try to extract project ID from model package group name
        project_id = None

        # Common naming patterns for LLMOps projects
        if "-models" in model_package_group_name:
            # Split by hyphens and look for project ID pattern
            parts = model_package_group_name.replace("-models", "").split("-")

            # Project IDs are typically UUIDs or alphanumeric strings
            for part in reversed(parts):  # Start from the end
                if len(part) >= 8 and part.replace("-", "").isalnum():
                    project_id = part
                    break

            # If not found, try combining last few parts
            if not project_id and len(parts) >= 2:
                project_id = "-".join(parts[-2:])  # Take last 2 parts

        print(f"Extracted project_id from model package group name: {project_id}")

        return project_id

    except Exception as e:
        print(f"Error extracting project info from model package group name: {str(e)}")
        raise


def get_project_details_from_sagemaker(project_id):
    """Get project details from SageMaker using project ID - handles both project and direct model scenarios"""
    try:
        sagemaker_client = boto3.client("sagemaker")

        # List projects and find the one with matching project ID
        projects_response = sagemaker_client.list_projects()

        for project_summary in projects_response.get("ProjectSummaryList", []):
            project_name = project_summary.get("ProjectName")

            # Get project details
            try:
                project_details = sagemaker_client.describe_project(
                    ProjectName=project_name
                )

                if project_details.get("ProjectId") == project_id:
                    print(f"Found matching project: {project_name}")
                    return project_details
            except Exception as e:
                print(f"Error describing project {project_name}: {str(e)}")
                continue

        # If not found in current page, you might need to handle pagination
        # For now, return None
        print(
            f"Project with ID {project_id} not found - this is normal for direct model registration"
        )
        return None

    except Exception as e:
        print(f"Error getting project details from SageMaker: {str(e)}")
        return None


def create_deploy_repository(project_id, git_token):
    """Create deploy repository using sync-repositories lambda logic"""
    try:
        private_organization_name = os.environ.get("GITHUB_ORG")
        repo_name = f"{project_id}-deploy-repo"

        print(f"Creating deploy repository: {private_organization_name}/{repo_name}")

        headers = {"Accept": "application/json", "Authorization": f"token {git_token}"}

        payload = {
            "name": repo_name,
            "private": True,
            "auto_init": True,
            "description": f"Deploy repository for SageMaker LLMOps project {project_id}",
        }

        # Try as organization first
        response = requests.post(
            f"https://api.github.com/orgs/{private_organization_name}/repos",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        # If organization fails, try as user account
        if response.status_code == 404:
            response = requests.post(
                f"https://api.github.com/user/repos",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

        if response.status_code not in [201, 200]:
            # Repository might already exist
            if response.status_code == 422:
                print(f"Repository {repo_name} already exists, continuing...")
                return f"{private_organization_name}/{repo_name}"
            else:
                raise Exception(
                    f"Failed to create repository: {response.status_code} - {response.text}"
                )

        print(
            f"Successfully created deploy repository: {private_organization_name}/{repo_name}"
        )
        return f"{private_organization_name}/{repo_name}"

    except Exception as e:
        print(f"Error creating deploy repository: {str(e)}")
        raise


def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event, indent=2, cls=DateTimeEncoder))

        # Extract model package group name and model package ARN from the event
        detail = event.get("detail", {})
        model_package_group_name = detail.get("ModelPackageGroupName")
        model_package_arn = detail.get("ModelPackageArn")

        if not model_package_group_name:
            raise ValueError("ModelPackageGroupName not found in event detail")

        print(
            f"Processing model approval for package group: {model_package_group_name}"
        )
        print(f"Model package ARN: {model_package_arn}")

        # Extract model artifacts S3 URI from the approved model package
        sagemaker_client = boto3.client("sagemaker")
        model_artifacts_s3_uri = None
        if model_package_arn:
            try:
                # Get model package details to extract the S3 URI
                model_package_name = model_package_arn.split("/")[-2]
                print(f"🔍 Describing model package: {model_package_name}")
                model_package_response = sagemaker_client.describe_model_package(
                    # ModelPackageName=model_package_name
                    ModelPackageName=model_package_arn
                )

                print(
                    f"📋 Full model package response keys: {list(model_package_response.keys())}"
                )

                # Extract S3 URI from model package - try multiple locations
                model_artifacts_s3_uri = None

                # Method 1: Check InferenceSpecification.Containers[0].ModelDataUrl
                inference_specification = model_package_response.get(
                    "InferenceSpecification", {}
                )
                containers = inference_specification.get("Containers", [])

                if containers and len(containers) > 0:
                    model_artifacts_s3_uri = containers[0].get("ModelDataUrl")
                    if model_artifacts_s3_uri:
                        print(
                            f"✅ Found model artifacts S3 URI in InferenceSpecification: {model_artifacts_s3_uri}"
                        )
                    else:
                        print("⚠️ ModelDataUrl not found in first container")
                        print(f"Container keys: {list(containers[0].keys())}")
                else:
                    print("⚠️ No containers found in inference specification")

                # Method 2: Check ModelPackageDescription (alternative location)
                if not model_artifacts_s3_uri:
                    model_data_url = model_package_response.get("ModelDataUrl")
                    if model_data_url:
                        model_artifacts_s3_uri = model_data_url
                        print(
                            f"✅ Found model artifacts S3 URI in ModelDataUrl: {model_artifacts_s3_uri}"
                        )

                # Method 3: Check AdditionalInferenceSpecifications (if exists)
                if not model_artifacts_s3_uri:
                    additional_specs = model_package_response.get(
                        "AdditionalInferenceSpecifications", []
                    )
                    if additional_specs:
                        for spec in additional_specs:
                            spec_containers = spec.get("Containers", [])
                            if spec_containers and spec_containers[0].get(
                                "ModelDataUrl"
                            ):
                                model_artifacts_s3_uri = spec_containers[0].get(
                                    "ModelDataUrl"
                                )
                                print(
                                    f"✅ Found model artifacts S3 URI in AdditionalInferenceSpecifications: {model_artifacts_s3_uri}"
                                )
                                break

                if not model_artifacts_s3_uri:
                    print(
                        "❌ Could not find model artifacts S3 URI in any expected location"
                    )
                    print(
                        f"Full model package response: {json.dumps(model_package_response, indent=2, cls=DateTimeEncoder)}"
                    )

            except Exception as e:
                print(f"❌ Error extracting model artifacts S3 URI: {str(e)}")
                print(f"Model package ARN: {model_package_arn}")
                import traceback

                print(f"Full traceback: {traceback.format_exc()}")
        else:
            print("⚠️ No model package ARN provided in event")

        # Get tags for the model package group
        account_id = event["account"]
        region = event["region"]
        model_package_group_arn = f"arn:aws:sagemaker:{region}:{account_id}:model-package-group/{model_package_group_name}"

        # sagemaker_client already initialized above
        tags_response = sagemaker_client.list_tags(ResourceArn=model_package_group_arn)

        print(
            "SageMaker Tags Response:",
            json.dumps(tags_response, indent=2, cls=DateTimeEncoder),
        )

        # Extract project_id, project_name, and domain ID from tags or model package group name
        project_id = None
        project_name = None
        domain_id = None

        # First, try to get from tags
        for tag in tags_response.get("Tags", []):
            if tag["Key"] == "sagemaker:project-id":
                project_id = tag["Value"]
            elif tag["Key"] == "sagemaker:project-name":
                project_name = tag["Value"]
            elif tag["Key"] == "SageMakerDomain":  # Updated tag name for Studio Domain
                domain_id = tag["Value"]
            elif tag["Key"] == "sagemaker:domain-id":  # Alternative tag name
                domain_id = tag["Value"]

        # If project_id not found in tags, try to extract from model package group name
        if not project_id:
            project_id = extract_project_info_from_model_package_group(
                model_package_group_name, sagemaker_client
            )

        # If domain_id not found in tags, get from environment or project details
        if not domain_id:
            domain_id = os.environ.get("SAGEMAKER_DOMAIN_ID")

            # If still not found, try to get from project details
            if not domain_id and project_id:
                project_details = get_project_details_from_sagemaker(project_id)
                if project_details:
                    # Domain ID might be in project tags or other fields
                    project_tags = project_details.get("Tags", [])
                    for tag in project_tags:
                        if tag["Key"] in ["SageMakerDomain", "sagemaker:domain-id"]:
                            domain_id = tag["Value"]
                            break

        if not project_id:
            raise ValueError(
                f"Could not find or extract project ID from model package group {model_package_group_name}"
            )

        print(f"Using project_id: {project_id}")
        print(f"Using domain_id: {domain_id}")

        # Construct repository name for SageMaker Studio Domain
        private_organization_name = os.environ.get("GITHUB_ORG")
        if not private_organization_name:
            raise ValueError("GITHUB_ORG environment variable is not set")

        # Updated repository naming for SageMaker Studio Domain
        repo_name = f"{project_id}-deploy-repo"  # Simplified naming

        # Get GitHub token from Secrets Manager
        secret_name = os.environ["GITHUB_TOKEN_SECRET_NAME"]
        secrets_client = boto3.client("secretsmanager")
        secrets_response = secrets_client.get_secret_value(SecretId=secret_name)
        git_token = json.loads(secrets_response["SecretString"])["token"]

        print(f"Organization name: {private_organization_name}")
        print(f"Repository name: {repo_name}")
        print(f"Workflow filename: {WORKFLOW_FILENAME}")

        # Step 1: Create deploy repository if it doesn't exist
        deploy_repo_full_name = create_deploy_repository(project_id, git_token)
        print(f"Deploy repository ready: {deploy_repo_full_name}")

        # Step 2: Populate repository with deploy code (call sync-repositories lambda)
        try:
            lambda_client = boto3.client("lambda")

            # Use the actual project name from tags, or fallback to project_id if not found
            actual_project_name = project_name if project_name else project_id

            sync_payload = {
                "projectName": actual_project_name,  # Use actual project name from tags
                "projectId": project_id,
                "domainId": domain_id,
                "buildRepo": deploy_repo_full_name,  # Use deploy repo as target
                "modelArtifactsS3Uri": model_artifacts_s3_uri,  # Pass model S3 URI
            }

            print(
                f"Invoking sync-repositories lambda with payload: {json.dumps(sync_payload, indent=2)}"
            )

            # Invoke sync-repositories lambda to populate the deploy repo
            function_arn = os.environ.get("SYNC_REPOSITORIES_FUNCTION_ARN", "").strip()
            if not function_arn:
                raise ValueError(
                    "SYNC_REPOSITORIES_FUNCTION_ARN environment variable is not set or empty"
                )

            print(f"Invoking function: '{function_arn}'")

            sync_response = lambda_client.invoke(
                FunctionName=function_arn,
                InvocationType="RequestResponse",
                Payload=json.dumps(sync_payload),
            )

            sync_result = json.loads(sync_response["Payload"].read())
            print(f"Sync repositories result: {json.dumps(sync_result, indent=2)}")

            if sync_result.get("statusCode") != 200:
                print(f"Warning: Sync repositories failed: {sync_result}")
                # Continue anyway - repository might already be populated

        except Exception as e:
            print(f"Warning: Failed to populate deploy repository: {str(e)}")
            # Continue anyway - we'll try to trigger the workflow

        # Step 3: GitHub API endpoint for workflow trigger
        url = f"https://api.github.com/repos/{deploy_repo_full_name}/actions/workflows/{WORKFLOW_FILENAME}/dispatches"
        print(f"Complete GitHub API URL: {url}")

        # Headers for GitHub API
        headers = {
            "Authorization": f"token {git_token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        }

        # Enhanced payload for the workflow with model-specific information
        payload = {
            "ref": "main",
            "inputs": {
                "logLevel": "info",
            },
        }

        print(
            f"Triggering GitHub workflow with payload: {json.dumps(payload, indent=2)}"
        )

        # Trigger the workflow
        github_response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=30
        )
        github_response.raise_for_status()

        print(f"Response status code: {github_response.status_code}")
        print(f"Response body: {github_response.text}")

        if github_response.status_code == 204:
            print(
                f"Successfully triggered GitHub workflow for model {model_package_group_name}"
            )
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "message": "Workflow triggered successfully",
                        "project_id": project_id,
                        "domain_id": domain_id,
                        "repository": f"{private_organization_name}/{repo_name}",
                        "model_package_group": model_package_group_name,
                        "workflow_triggered": WORKFLOW_FILENAME,
                    }
                ),
            }
        else:
            error_message = f"Failed to trigger workflow. Status code: {github_response.status_code}, Response: {github_response.text}"
            print(error_message)
            raise Exception(error_message)

    except Exception as e:
        error_message = str(e)
        print(f"Error: {error_message}")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": error_message,
                    "project_id": project_id if "project_id" in locals() else None,
                    "domain_id": domain_id if "domain_id" in locals() else None,
                    "model_package_group": (
                        model_package_group_name
                        if "model_package_group_name" in locals()
                        else None
                    ),
                }
            ),
        }
