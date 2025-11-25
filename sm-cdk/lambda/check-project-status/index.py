import boto3
import json
from datetime import datetime


def datetime_handler(obj):
    """Handle datetime objects for JSON serialization"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def check_project_status(project_name):
    """Check if SageMaker project is ready"""
    try:
        sagemaker_client = boto3.client("sagemaker")
        servicecatalog_client = boto3.client("servicecatalog")

        # Describe the SageMaker project
        response = sagemaker_client.describe_project(ProjectName=project_name)

        project_status = response.get("ProjectStatus")
        project_id = response.get("ProjectId")
        creation_time = response.get("CreationTime")

        print(f"Project Name: {project_name}")
        print(f"Project ID: {project_id}")
        print(f"Project Status: {project_status}")
        print(f"Creation Time: {creation_time}")

        # Check Service Catalog provisioned product status
        service_catalog_details = response.get("ServiceCatalogProvisioningDetails", {})
        provisioned_product_details = response.get(
            "ServiceCatalogProvisionedProductDetails", {}
        )

        provisioning_artifact_id = service_catalog_details.get("ProvisioningArtifactId")
        product_id = service_catalog_details.get("ProductId")
        provisioned_product_id = provisioned_product_details.get("ProvisionedProductId")

        print(f"Service Catalog Product ID: {product_id}")
        print(f"Provisioning Artifact ID: {provisioning_artifact_id}")
        print(f"Provisioned Product ID: {provisioned_product_id}")

        # Check the provisioned product status if available
        provisioned_product_status = None
        if provisioned_product_id:
            try:
                pp_response = servicecatalog_client.describe_provisioned_product(
                    Id=provisioned_product_id
                )
                provisioned_product_status = pp_response.get(
                    "ProvisionedProductDetail", {}
                ).get("Status")
                print(f"Provisioned Product Status: {provisioned_product_status}")
            except Exception as e:
                print(f"Could not get provisioned product status: {str(e)}")

        # Determine the effective status
        effective_status = determine_effective_status(
            project_status, provisioned_product_status
        )
        print(f"Effective Status: {effective_status}")

        return effective_status, response

    except sagemaker_client.exceptions.ResourceNotFound:
        print(f"Project {project_name} not found")
        return "NOT_FOUND", {}
    except Exception as e:
        print(f"Error checking project status: {str(e)}")
        raise


def determine_effective_status(project_status, provisioned_product_status):
    """Determine the effective status based on both SageMaker project and Service Catalog status"""

    # If Service Catalog provisioned product is available, use it for more accurate status
    if provisioned_product_status:
        if provisioned_product_status in ["AVAILABLE"]:
            return "COMPLETED"  # Service Catalog product is ready
        elif provisioned_product_status in ["UNDER_CHANGE", "PLAN_IN_PROGRESS"]:
            return "PENDING"  # Still being provisioned
        elif provisioned_product_status in ["ERROR", "TAINTED"]:
            return "FAILED"  # Provisioning failed

    # Fall back to SageMaker project status
    if project_status in ["InService", "CreateCompleted"]:
        return "COMPLETED"
    elif project_status in ["CreateInProgress", "Pending"]:
        return "PENDING"
    elif project_status in ["CreateFailed", "DeleteInProgress", "DeleteFailed"]:
        return "FAILED"

    # Default to pending for unknown statuses
    return "PENDING"


def extract_project_details_from_event(event):
    """Extract project details from various event sources"""
    project_name = None
    project_id = None
    domain_id = None
    user_params = []

    if "detail" in event:
        # EventBridge event from SageMaker project creation
        detail = event["detail"]

        # Check for SageMaker project events
        if "responseElements" in detail:
            response_elements = detail["responseElements"]
            project_name = response_elements.get("projectName")
            project_id = response_elements.get("projectId")

        # Extract request parameters
        if "requestParameters" in detail:
            request_params = detail["requestParameters"]
            if not project_name:
                project_name = request_params.get("projectName")

            # For SageMaker Studio Domain projects, domain info might be in different places
            domain_id = request_params.get("domainId") or request_params.get(
                "domainIdentifier"
            )

        # Check for source-specific formats
        event_source = event.get("source", "")
        if event_source == "aws.sagemaker":
            # Direct SageMaker event
            if "ProjectName" in detail:
                project_name = detail["ProjectName"]
            if "ProjectId" in detail:
                project_id = detail["ProjectId"]

    else:
        # Direct invocation or Step Functions input
        project_name = event.get("projectName") or event.get("ProjectName")
        project_id = event.get("projectId") or event.get("ProjectId")
        domain_id = event.get("domainId") or event.get("DomainId")
        user_params = event.get("userParameters", [])

    return project_name, project_id, domain_id, user_params


def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event, indent=2, default=datetime_handler))

        # Extract project details from event
        project_name, project_id, domain_id, user_params = (
            extract_project_details_from_event(event)
        )

        print(
            f"Extracted - Project Name: {project_name}, Project ID: {project_id}, Domain ID: {domain_id}"
        )

        if not project_name:
            raise Exception("Missing required project name")

        # Check project status
        workflow_status, project_details = check_project_status(project_name)

        # Get the actual SageMaker status from project details
        sagemaker_status = (
            project_details.get("ProjectStatus", "UNKNOWN")
            if project_details
            else "NOT_FOUND"
        )

        # Extract additional details if project exists
        if workflow_status != "NOT_FOUND":
            if not project_id:
                project_id = project_details.get("ProjectId")
            if not domain_id:
                # Try to extract domain from project details or use environment variable
                import os

                domain_id = os.environ.get("SAGEMAKER_DOMAIN_ID")

        # Create response
        response = {
            "status": workflow_status,
            "sagemakerStatus": sagemaker_status,
            "projectName": project_name,
            "projectId": project_id,
            "domainId": domain_id,
            "userParameters": user_params,
            "projectDetails": project_details,
        }

        print(f"Response status: {workflow_status}")

        # Return JSON-serialized response using the custom handler
        return json.loads(json.dumps(response, default=datetime_handler))

    except Exception as e:
        print(f"Error: {str(e)}")
        # Return error status for Step Functions to handle
        return {
            "status": "FAILED",
            "error": str(e),
            "projectName": project_name if "project_name" in locals() else None,
            "projectId": project_id if "project_id" in locals() else None,
            "domainId": domain_id if "domain_id" in locals() else None,
        }


def map_sagemaker_status_to_workflow(sagemaker_status):
    """Map SageMaker project status to Step Functions workflow status"""
    status_mapping = {
        "Pending": "PENDING",
        "CreateInProgress": "PENDING",
        "InService": "COMPLETED",
        "CreateCompleted": "COMPLETED",
        "CreateFailed": "FAILED",
        "DeleteInProgress": "FAILED",
        "DeleteFailed": "FAILED",
        "NOT_FOUND": "PENDING",  # Might still be creating
    }

    return status_mapping.get(sagemaker_status, "PENDING")
