import boto3
import json
import os
import requests

# ----------------- GitHub helpers -----------------


def _gh_headers(token: str) -> dict:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _sanitize_var_name(name: str) -> str:
    """GitHub repo variable rules: A–Z, 0–9, _ ; cannot start with GITHUB_ ; <=100 chars."""
    import re

    n = re.sub(r"[^A-Z0-9_]", "_", name.upper())
    return n[:100]


# ----------------- autodiscovery (unchanged) -----------------


def auto_discover_mlflow_tracking_arn():
    try:
        # First try to get from CloudFormation stack outputs
        try:
            cloudformation_client = boto3.client("cloudformation")

            # Find stacks starting with LlmOpsSm-Observability
            paginator = cloudformation_client.get_paginator("list_stacks")
            for page in paginator.paginate(
                StackStatusFilter=["CREATE_COMPLETE", "UPDATE_COMPLETE"]
            ):
                for stack_summary in page["StackSummaries"]:
                    stack_name = stack_summary["StackName"]
                    if stack_name.startswith("LlmOpsSm-Observability"):
                        try:
                            response = cloudformation_client.describe_stacks(
                                StackName=stack_name
                            )
                            stacks = response.get("Stacks", [])
                            if stacks:
                                outputs = stacks[0].get("Outputs", [])
                                for output in outputs:
                                    output_key = output.get("OutputKey", "")
                                    if (
                                        "MLflowTrackingServerArn" in output_key
                                        or "MlflowTrackingArn" in output_key
                                        or "TrackingServerArn" in output_key
                                    ):
                                        arn = output.get("OutputValue")
                                        if arn and arn.startswith("arn:aws:sagemaker:"):
                                            print(
                                                f"Found MLflow tracking server ARN from CloudFormation stack '{stack_name}': {arn}"
                                            )
                                            return arn
                        except Exception:
                            continue
        except Exception as e:
            print(f"CloudFormation lookup for MLflow failed: {str(e)}")

        # Fallback to SageMaker API discovery
        print("CloudFormation lookup failed, trying SageMaker API...")
        sagemaker_client = boto3.client("sagemaker")
        response = sagemaker_client.list_mlflow_tracking_servers()
        tracking_servers = response.get("TrackingServerSummaries", [])
        if tracking_servers:
            for server in tracking_servers:
                if server.get("TrackingServerStatus") == "Created":
                    arn = server.get("TrackingServerArn")
                    print(f"Auto-discovered MLflow tracking server ARN: {arn}")
                    return arn

        # Final fallback: Try to construct ARN from common naming patterns
        print("No active MLflow tracking servers found via API, trying fallback...")
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        region = boto3.session.Session().region_name

        # Common MLflow server naming patterns
        common_names = [
            "llmops-mlflow-dev",
            "llmops-mlflow",
            "mlflow-tracking-server",
            "mlflow-server",
        ]

        for name in common_names:
            try:
                # Try to describe the specific tracking server
                response = sagemaker_client.describe_mlflow_tracking_server(
                    TrackingServerName=name
                )
                if response.get("TrackingServerStatus") == "Created":
                    arn = response.get("TrackingServerArn")
                    print(f"Found MLflow tracking server by name '{name}': {arn}")
                    return arn
            except Exception:
                continue

        print("No MLflow tracking servers found")
        return ""
    except Exception as e:
        print(f"Error auto-discovering MLflow tracking server: {str(e)}")
        return ""


def auto_discover_input_data_path(artifacts_bucket):
    try:
        s3_client = boto3.client("s3")
        common_paths = [
            f"s3://{artifacts_bucket}/datasets/esg/",
            f"s3://{artifacts_bucket}/data/esg/",
            f"s3://{artifacts_bucket}/training-data/",
            f"s3://{artifacts_bucket}/datasets/",
            f"s3://{artifacts_bucket}/data/",
        ]
        for path in common_paths:
            bucket_name = artifacts_bucket
            prefix = path.replace(f"s3://{bucket_name}/", "")
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name, Prefix=prefix, MaxKeys=1
                )
                if response.get("Contents"):
                    print(f"Auto-discovered input data path: {path}")
                    return path
            except Exception:
                continue
        default_path = f"s3://{artifacts_bucket}/datasets/esg/"
        print(f"Using default input data path: {default_path}")
        return default_path
    except Exception as e:
        print(f"Error auto-discovering input data path: {str(e)}")
        return f"s3://{artifacts_bucket}/datasets/esg/"


def auto_discover_sagemaker_domain_arn():
    try:
        # First try to get from CloudFormation stack outputs
        try:
            cloudformation_client = boto3.client("cloudformation")

            # Find stacks starting with LlmOpsSm-Infrastructure
            paginator = cloudformation_client.get_paginator("list_stacks")
            for page in paginator.paginate(
                StackStatusFilter=["CREATE_COMPLETE", "UPDATE_COMPLETE"]
            ):
                for stack_summary in page["StackSummaries"]:
                    stack_name = stack_summary["StackName"]
                    if stack_name.startswith("LlmOpsSm-Infrastructure"):
                        try:
                            response = cloudformation_client.describe_stacks(
                                StackName=stack_name
                            )
                            stacks = response.get("Stacks", [])
                            if stacks:
                                outputs = stacks[0].get("Outputs", [])
                                for output in outputs:
                                    output_key = output.get("OutputKey", "")
                                    if (
                                        "SageMakerDomainId" in output_key
                                        or "DomainId" in output_key
                                    ):
                                        domain_id = output.get("OutputValue")
                                        if domain_id:
                                            # Construct ARN from domain ID
                                            account_id = boto3.client(
                                                "sts"
                                            ).get_caller_identity()["Account"]
                                            region = boto3.session.Session().region_name
                                            domain_arn = f"arn:aws:sagemaker:{region}:{account_id}:domain/{domain_id}"
                                            print(
                                                f"Found SageMaker domain ID from CloudFormation stack '{stack_name}': {domain_id}"
                                            )
                                            print(
                                                f"Constructed domain ARN: {domain_arn}"
                                            )
                                            return domain_arn
                        except Exception:
                            continue
        except Exception as e:
            print(f"CloudFormation lookup failed: {str(e)}")

        # Fallback to SageMaker API discovery
        print("CloudFormation lookup failed, trying SageMaker API...")
        sagemaker_client = boto3.client("sagemaker")
        response = sagemaker_client.list_domains()
        domains = response.get("Domains", [])
        if domains:
            domain_arn = domains[0].get("DomainArn")
            print(f"Auto-discovered SageMaker domain ARN via API: {domain_arn}")
            return domain_arn
        print("No SageMaker domains found")
        return ""
    except Exception as e:
        print(f"Error auto-discovering SageMaker domain ARN: {str(e)}")
        return ""


def auto_discover_artifacts_bucket():
    try:
        s3_client = boto3.client("s3")
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        region = boto3.session.Session().region_name
        common_patterns = [
            f"llmops-sm-artifacts-{account_id}-{region}",
            f"sagemaker-{region}-{account_id}",
            f"mlops-artifacts-{account_id}",
            f"sm-artifacts-{account_id}",
            f"sagemaker-studio-{account_id}",
        ]
        response = s3_client.list_buckets()
        existing_buckets = [bucket["Name"] for bucket in response.get("Buckets", [])]
        for pattern in common_patterns:
            if pattern in existing_buckets:
                print(f"Auto-discovered artifacts bucket: {pattern}")
                return pattern
        for bucket_name in existing_buckets:
            if any(
                k in bucket_name.lower() for k in ["artifact", "mlops", "sagemaker"]
            ):
                print(f"Auto-discovered artifacts bucket by keyword: {bucket_name}")
                return bucket_name
        print("No artifacts bucket found, will use default naming")
        return ""
    except Exception as e:
        print(f"Error auto-discovering artifacts bucket: {str(e)}")
        return ""


def auto_discover_oidc_build_role_arn():
    """Auto-discover the GitHub OIDC Build role ARN."""
    try:
        iam_client = boto3.client("iam")
        common_patterns = [
            "GitHubAction-AssumeRoleWithAction-Build",
            "github-actions-build-role",
            "GitHubActionsBuildRole",
            "github-oidc-build-role",
            "GitHubOIDCBuildRole",
        ]
        for pattern in common_patterns:
            try:
                response = iam_client.get_role(RoleName=pattern)
                role_arn = response["Role"]["Arn"]
                print(f"Auto-discovered GitHub OIDC Build role ARN: {role_arn}")
                if not role_arn.startswith("arn:aws:iam::"):
                    print(f"Warning: Expected ARN format but got: {role_arn}")
                    continue
                return role_arn
            except iam_client.exceptions.NoSuchEntityException:
                continue

        # Search through all roles for GitHub/OIDC/Build related roles
        paginator = iam_client.get_paginator("list_roles")
        for page in paginator.paginate():
            for role in page["Roles"]:
                rn = role["RoleName"].lower()
                if any(k in rn for k in ["github", "oidc", "action"]) and "build" in rn:
                    role_arn = role["Arn"]
                    print(
                        f"Auto-discovered GitHub OIDC Build role ARN by keyword: {role_arn}"
                    )
                    if not role_arn.startswith("arn:aws:iam::"):
                        print(f"Warning: Expected ARN format but got: {role_arn}")
                        continue
                    return role_arn

        print("No GitHub OIDC Build role found")
        return ""
    except Exception as e:
        print(f"Error auto-discovering GitHub OIDC Build role ARN: {str(e)}")
        return ""


def auto_discover_oidc_deploy_role_arn():
    """Auto-discover the GitHub OIDC Deploy role ARN."""
    try:
        iam_client = boto3.client("iam")
        common_patterns = [
            "GitHubAction-AssumeRoleWithAction-Deploy",
            "github-actions-deploy-role",
            "GitHubActionsDeployRole",
            "github-oidc-deploy-role",
            "GitHubOIDCDeployRole",
        ]
        for pattern in common_patterns:
            try:
                response = iam_client.get_role(RoleName=pattern)
                role_arn = response["Role"]["Arn"]
                print(f"Auto-discovered GitHub OIDC Deploy role ARN: {role_arn}")
                if not role_arn.startswith("arn:aws:iam::"):
                    print(f"Warning: Expected ARN format but got: {role_arn}")
                    continue
                return role_arn
            except iam_client.exceptions.NoSuchEntityException:
                continue

        # Search through all roles for GitHub/OIDC/Deploy related roles
        paginator = iam_client.get_paginator("list_roles")
        for page in paginator.paginate():
            for role in page["Roles"]:
                rn = role["RoleName"].lower()
                if (
                    any(k in rn for k in ["github", "oidc", "action"])
                    and "deploy" in rn
                ):
                    role_arn = role["Arn"]
                    print(
                        f"Auto-discovered GitHub OIDC Deploy role ARN by keyword: {role_arn}"
                    )
                    if not role_arn.startswith("arn:aws:iam::"):
                        print(f"Warning: Expected ARN format but got: {role_arn}")
                        continue
                    return role_arn

        print("No GitHub OIDC Deploy role found")
        return ""
    except Exception as e:
        print(f"Error auto-discovering GitHub OIDC Deploy role ARN: {str(e)}")
        return ""


def get_environment_config(artifacts_bucket=None):
    if not artifacts_bucket:
        artifacts_bucket = (
            os.environ.get("ARTIFACTS_BUCKET") or auto_discover_artifacts_bucket()
        )
    config = {
        "MLFLOW_TRACKING_ARN": os.environ.get("MLFLOW_TRACKING_ARN")
        or auto_discover_mlflow_tracking_arn(),
        "INPUT_DATA_PATH": os.environ.get("INPUT_DATA_PATH")
        or auto_discover_input_data_path(artifacts_bucket),
        "SAGEMAKER_DOMAIN_ARN": os.environ.get("SAGEMAKER_DOMAIN_ARN")
        or auto_discover_sagemaker_domain_arn(),
        "ARTIFACTS_BUCKET": artifacts_bucket,
        "OIDC_ROLE_GITHUB_WORKFLOW_BUILD": os.environ.get("GITHUB_BUILD_ROLE_ARN")
        or auto_discover_oidc_build_role_arn(),
        "OIDC_ROLE_GITHUB_WORKFLOW_DEPLOY": os.environ.get("GITHUB_DEPLOY_ROLE_ARN")
        or auto_discover_oidc_deploy_role_arn(),
    }
    print("Environment configuration:")
    for key, value in config.items():
        print(f"  {key}: {'<empty>' if not value else str(value)[:50]}...")
    return config


# ----------------- GitHub actions -----------------


def create_github_repository(private_organization_name, repo_name, git_token):
    """Create repo in org or user; returns 'owner/repo'."""
    try:
        print(f"\nCreating GitHub repository: {private_organization_name}/{repo_name}")
        headers = _gh_headers(git_token)

        payload = {
            "name": repo_name,
            "private": True,
            "auto_init": True,
            "description": "Build repository for SageMaker LLMOps project",
        }

        # Try org first
        org_resp = requests.post(
            f"https://api.github.com/orgs/{private_organization_name}/repos",
            headers=headers,
            json=payload,
            timeout=30,
        )

        # Check status code BEFORE raising exception
        if org_resp.status_code == 201:
            print(
                f"Successfully created repository in org: {private_organization_name}/{repo_name}"
            )
            return f"{private_organization_name}/{repo_name}"

        if org_resp.status_code == 404:
            # Org doesn't exist or no access - fallback to user
            print(
                f"Organization '{private_organization_name}' not accessible (404), trying user account..."
            )
            user_resp = requests.post(
                "https://api.github.com/user/repos",
                headers=headers,
                json=payload,
                timeout=30,
            )
            user_resp.raise_for_status()
            if user_resp.status_code in (200, 201):
                owner = user_resp.json()["owner"]["login"]
                print(
                    f"Successfully created repository in user account: {owner}/{repo_name}"
                )
                return f"{owner}/{repo_name}"

        # If we get here, something else went wrong
        org_resp.raise_for_status()  # This will raise the appropriate error

        # Fallback error if raise_for_status didn't raise
        details = {
            "status_code": org_resp.status_code,
            "response_text": org_resp.text,
        }
        raise Exception(f"Failed to create repository: {json.dumps(details)}")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error creating GitHub repository: {str(e)}")
        raise
    except Exception as e:
        print(f"Error creating GitHub repository: {str(e)}")
        raise


def create_github_variables(repo_full_name, variables_data, git_token):
    """Create/Update repo variables (uses Actions Variables API)."""
    try:
        headers = _gh_headers(git_token)
        base_url = f"https://api.github.com/repos/{repo_full_name}/actions/variables"
        print(f"\nCreating GitHub repository variables in: {repo_full_name}")

        for raw_name, raw_value in variables_data.items():
            if raw_value is None:
                continue
            name = _sanitize_var_name(raw_name)
            if name.startswith("GITHUB_"):
                print(
                    f"Skipping variable '{raw_name}' -> '{name}': cannot start with GITHUB_."
                )
                continue

            value = str(raw_value)

            # Try create
            r = requests.post(
                base_url,
                headers=headers,
                json={"name": name, "value": value},
                timeout=30,
            )
            r.raise_for_status()
            if r.status_code == 201:
                print(f"✓ Variable created: {name}")
                continue

            if r.status_code == 409:
                # Exists → update
                u = requests.patch(
                    f"{base_url}/{name}",
                    headers=headers,
                    json={"value": value},
                    timeout=30,
                )
                u.raise_for_status()
                if u.status_code == 204:
                    print(f"✓ Variable updated: {name}")
                else:
                    print(
                        f"⚠ Failed to update variable {name}: {u.status_code} {u.text}"
                    )
                continue

            print(f"⚠ Failed to create variable {name}: {r.status_code} {r.text}")

        print(f"Completed GitHub variables setup for {repo_full_name}")
    except Exception as e:
        print(f"Error creating GitHub variables: {str(e)}")
        print("Skipping variable creation - variables will need to be created manually")


# ----------------- template repository ops -----------------


def find_template_repository(project_template_name):
    try:
        public_templates_org = os.environ["TEMPLATE_ORG"]
        public_templates_repo = os.environ["TEMPLATE_REPO"]
        template_code_folder = os.environ["TEMPLATE_FOLDER"]

        project_template_name = project_template_name.lower()

        print(f"\nLooking for template in:")
        print(f"Organization: {public_templates_org}")
        print(f"Repository: {public_templates_repo}")
        print(f"Code Folder: {template_code_folder}")
        print(f"Template Name: {project_template_name}")

        template_repo_url = (
            f"https://github.com/{public_templates_org}/{public_templates_repo}.git"
        )
        print(f"\nUsing template repository: {template_repo_url}")

        verify_url = f"https://api.github.com/repos/{public_templates_org}/{public_templates_repo}"

        git_token = os.environ.get("GITHUB_TOKEN")
        if not git_token:
            try:
                secret_name = os.environ.get("GITHUB_TOKEN_SECRET_NAME")
                if secret_name:
                    secrets_client = boto3.client("secretsmanager")
                    response = secrets_client.get_secret_value(SecretId=secret_name)
                    git_token = json.loads(response["SecretString"])["token"]
                    print(
                        f"Retrieved GitHub token from Secrets Manager (len={len(git_token) if git_token else 0})"
                    )
            except Exception as e:
                print(f"Could not get GitHub token: {str(e)}")

        headers = _gh_headers(git_token) if git_token else {}
        if headers:
            print("Testing repository access with token...")

        response = requests.get(verify_url, headers=headers, timeout=30)
        response.raise_for_status()
        print(f"Repository verification response: {response.status_code}")

        if response.status_code != 200:
            print(f"Repository verification failed: {response.text}")
            if response.status_code == 404:
                print("Repository not found - it may be private or doesn't exist")
            elif response.status_code == 401:
                print("Authentication failed - token may be invalid or expired")
            elif response.status_code == 403:
                print("Access forbidden - token may not have required permissions")
            raise Exception(
                f"Template repository not found: {template_repo_url} (Status: {response.status_code})"
            )

        print("Repository verification successful")
        return template_repo_url
    except Exception as e:
        print(f"Error finding template repository: {str(e)}")
        raise


def copy_template_content_via_api(build_repo_name, git_token, template_name):
    """Copy model_build folder contents using Git trees/blobs/commits APIs."""
    try:
        print("\nCopying template content via GitHub API...")

        template_org = os.environ["TEMPLATE_ORG"]
        template_repo = os.environ["TEMPLATE_REPO"]
        template_folder = os.environ["TEMPLATE_FOLDER"]

        headers = _gh_headers(git_token)

        template_path = f"{template_folder}/{template_name}/model_build"
        tree_url = f"https://api.github.com/repos/{template_org}/{template_repo}/git/trees/main?recursive=1"

        print(f"Fetching template repository tree from: {tree_url}")
        response = requests.get(tree_url, headers=headers, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to get template repository tree: {response.text}")

        tree_data = response.json()

        model_build_files = []
        for item in tree_data.get("tree", []):
            if item["type"] == "blob" and item["path"].startswith(template_path):
                rel = item["path"][len(template_path) :].lstrip("/")
                if rel:
                    model_build_files.append(
                        {"path": rel, "sha": item["sha"], "url": item["url"]}
                    )

        if not model_build_files:
            raise Exception(f"No files found in {template_path}")

        print(f"Found {len(model_build_files)} files to copy")

        # Build repo main ref
        branch_url = (
            f"https://api.github.com/repos/{build_repo_name}/git/refs/heads/main"
        )
        response = requests.get(branch_url, headers=headers, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to get build repository branch: {response.text}")
        main_branch_sha = response.json()["object"]["sha"]

        # Create blobs in target
        file_blobs = []
        for f in model_build_files:
            print(f"Processing file: {f['path']}")
            blob_response = requests.get(f["url"], headers=headers, timeout=30)
            if blob_response.status_code != 200:
                print(f"Warning: Failed to get content for {f['path']}")
                continue
            blob_data = blob_response.json()

            create_blob_url = (
                f"https://api.github.com/repos/{build_repo_name}/git/blobs"
            )
            create_response = requests.post(
                create_blob_url,
                headers=headers,
                json={
                    "content": blob_data["content"],
                    "encoding": blob_data["encoding"],
                },
                timeout=30,
            )
            if create_response.status_code != 201:
                print(f"Warning: Failed to create blob for {f['path']}")
                continue
            new_blob_sha = create_response.json()["sha"]
            file_blobs.append(
                {
                    "path": f["path"],
                    "mode": "100644",
                    "type": "blob",
                    "sha": new_blob_sha,
                }
            )

        if not file_blobs:
            raise Exception("No files were successfully processed")

        # Create tree
        create_tree_url = f"https://api.github.com/repos/{build_repo_name}/git/trees"
        tree_response = requests.post(
            create_tree_url,
            headers=headers,
            json={"base_tree": main_branch_sha, "tree": file_blobs},
            timeout=30,
        )
        if tree_response.status_code != 201:
            raise Exception(f"Failed to create tree: {tree_response.text}")
        new_tree_sha = tree_response.json()["sha"]

        # Create commit
        create_commit_url = (
            f"https://api.github.com/repos/{build_repo_name}/git/commits"
        )
        commit_response = requests.post(
            create_commit_url,
            headers=headers,
            json={
                "message": f"Initial setup - Copying model_build folder contents from {template_name}",
                "tree": new_tree_sha,
                "parents": [main_branch_sha],
            },
            timeout=60,
        )
        if commit_response.status_code != 201:
            raise Exception(f"Failed to create commit: {commit_response.text}")
        new_commit_sha = commit_response.json()["sha"]

        # Update ref
        update_ref_url = (
            f"https://api.github.com/repos/{build_repo_name}/git/refs/heads/main"
        )
        ref_response = requests.patch(
            update_ref_url,
            headers=headers,
            json={"sha": new_commit_sha},
            timeout=60,
        )
        if ref_response.status_code != 200:
            raise Exception(f"Failed to update branch reference: {ref_response.text}")

        print(f"Successfully copied {len(file_blobs)} files to build repository")
        return True
    except Exception as e:
        print(f"Error copying template content via API: {str(e)}")
        raise


# ----------------- project details -----------------


def get_sagemaker_project_details(project_name):
    try:
        sagemaker_client = boto3.client("sagemaker")
        project_response = sagemaker_client.describe_project(ProjectName=project_name)
        project_id = project_response.get("ProjectId")
        project_arn = project_response.get("ProjectArn")
        service_catalog_details = project_response.get(
            "ServiceCatalogProvisioningDetails", {}
        )
        domain_id = os.environ.get("SAGEMAKER_DOMAIN_ID")

        execution_role = None
        if "Tags" in project_response:
            for tag in project_response["Tags"]:
                if tag["Key"] == "ExecutionRole":
                    execution_role = tag["Value"]
                    break
        if not execution_role:
            execution_role = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN")

        artifacts_bucket = os.environ.get("ARTIFACTS_BUCKET")

        return {
            "projectId": project_id,
            "projectArn": project_arn,
            "domainId": domain_id,
            "executionRole": execution_role,
            "artifactsBucket": artifacts_bucket,
            "serviceCatalogDetails": service_catalog_details,
        }
    except Exception as e:
        print(f"Error getting SageMaker project details: {str(e)}")
        raise


# ----------------- handler -----------------


def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event, indent=2))

        event_data = event.get("body", event)
        if isinstance(event_data, str):
            event_data = json.loads(event_data)

        project_name = event_data.get("projectName")
        project_id = event_data.get("projectId")
        domain_id = event_data.get("domainId")
        project_details = event_data.get("projectDetails", {})

        template_name = "esg-benchmarking"
        if project_details:
            _ = project_details.get("ServiceCatalogProvisioningDetails", {})

        region = boto3.session.Session().region_name

        print(f"\nExtracted parameters:")
        print(f"project_name: {project_name}")
        print(f"project_id: {project_id}")
        print(f"domain_id: {domain_id}")
        print(f"template_name: {template_name}")

        if not all([project_name, project_id]):
            missing = []
            if not project_name:
                missing.append("project_name")
            if not project_id:
                missing.append("project_id")
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

        sagemaker_details = get_sagemaker_project_details(project_name)

        if not domain_id:
            domain_id = sagemaker_details.get("domainId")

        # GitHub token
        secret_name = os.environ.get("GITHUB_TOKEN_SECRET_NAME")
        secrets_client = boto3.client("secretsmanager")
        response = secrets_client.get_secret_value(SecretId=secret_name)
        git_token = json.loads(response["SecretString"])["token"]

        private_organization_name = os.environ.get("GITHUB_ORG") or os.environ.get(
            "TARGET_GITHUB_ORG"
        )

        # Create a unique, friendly repo name
        import random
        import string

        rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        base = (
            project_name.lower().replace(" ", "-").replace("_", "-")
            if project_name
            else template_name
        )
        repo_name = f"{base}-build-{rand}"

        build_repo = create_github_repository(
            private_organization_name, repo_name, git_token
        )
        print(f"Created build repository: {build_repo}")

        model_package_group_name = f"{template_name}-{project_id}-models"

        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity()["Account"]

        sagemaker_pipeline_role_arn = sagemaker_details.get("executionRole")
        if not sagemaker_pipeline_role_arn:
            sagemaker_pipeline_role_arn = f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMakerServiceCatalogProductsExecutionRole"

        env_config = get_environment_config(sagemaker_details.get("artifactsBucket"))

        # Write EVERYTHING as repo variables
        repo_vars = {
            "SAGEMAKER_PROJECT_NAME": project_name,
            "SAGEMAKER_PROJECT_ID": project_id,
            "SAGEMAKER_PIPELINE_ROLE_ARN": sagemaker_pipeline_role_arn,
            "REGION": region,
            "ARTIFACT_BUCKET": env_config.get("ARTIFACTS_BUCKET")
            or sagemaker_details.get("artifactsBucket"),
            "OIDC_ROLE_GITHUB_WORKFLOW_BUILD": env_config.get(
                "OIDC_ROLE_GITHUB_WORKFLOW_BUILD"
            ),
            "OIDC_ROLE_GITHUB_WORKFLOW_DEPLOY": env_config.get(
                "OIDC_ROLE_GITHUB_WORKFLOW_DEPLOY"
            ),
            "MODEL_PACKAGE_GROUP_NAME": model_package_group_name,
            "MLFLOW_TRACKING_ARN": env_config.get("MLFLOW_TRACKING_ARN", ""),
            "INPUT_DATA_PATH": env_config.get("INPUT_DATA_PATH", ""),
            "TRIGGER_PIPELINE_EXECUTION": "false",
            "SAGEMAKER_DOMAIN_ID": domain_id,
            "SAGEMAKER_DOMAIN_ARN": env_config.get("SAGEMAKER_DOMAIN_ARN", ""),
            "SAGEMAKER_PROJECT_ARN": sagemaker_details.get("projectArn"),
            "SAGEMAKER_EXECUTION_ROLE_ARN": sagemaker_details.get("executionRole"),
            "AWS_REGION": region,
            "ARTIFACTS_BUCKET": env_config.get("ARTIFACTS_BUCKET")
            or sagemaker_details.get("artifactsBucket"),
            "ESG_DATASET_S3_PATH": f"s3://{env_config.get('ARTIFACTS_BUCKET') or sagemaker_details.get('artifactsBucket')}/datasets/esg/",
            "ESG_BENCHMARK_S3_PATH": f"s3://{env_config.get('ARTIFACTS_BUCKET') or sagemaker_details.get('artifactsBucket')}/benchmarks/esg/",
            "MODEL_BASE_NAME": "mistralai/Mistral-7B-Instruct-v0.3",
        }

        print("\nVariables to be created/updated:")
        for k, v in repo_vars.items():
            print(f"{k}: {'<empty>' if not v else str(v)[:50]}...")

        create_github_variables(build_repo, repo_vars, git_token)

        # Verify template repo is reachable
        template_repo_url = find_template_repository(
            project_template_name=template_name
        )
        print(f"Found template repository: {template_repo_url}")

        # Seed model_build files
        copy_template_content_via_api(build_repo, git_token, template_name)

        return {
            "statusCode": 200,
            "status": "SUCCESSFUL",
            "projectName": project_name,
            "projectId": project_id,
            "domainId": domain_id,
            "buildRepo": build_repo,
            "sagemakerDetails": sagemaker_details,
            "additionalInfo": {
                "templateName": template_name,
                "secretsCreated": [],
                "variablesCreated": list(repo_vars.keys()),
                "templateRepo": template_repo_url,
                "modelPackageGroupName": model_package_group_name,
                "message": "Successfully created build repository, variables, and copied template content",
            },
        }

    except Exception as e:
        error_message = str(e)
        print(f"Error: {error_message}")
        return {
            "statusCode": 500,
            "status": "FAILED",
            "error": error_message,
            "projectName": (
                event_data.get("projectName") if "event_data" in locals() else None
            ),
            "projectId": (
                event_data.get("projectId") if "event_data" in locals() else None
            ),
        }
