import boto3
import json
import os
import requests
import shutil
import zipfile
import base64
import tempfile

# ---------- helpers ----------


def _gh_headers(token: str) -> dict:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _sanitize_var_name(name: str) -> str:
    # Uppercase, replace non [A-Z0-9_] with underscore, trim to 100 chars
    import re

    n = re.sub(r"[^A-Z0-9_]", "_", name.upper())
    return n[:100]


# ---------- autodiscovery (unchanged) ----------


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
        existing_buckets = [b["Name"] for b in response.get("Buckets", [])]
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


def auto_discover_oidc_role_arn():
    try:
        iam_client = boto3.client("iam")
        common_patterns = [
            "GitHubAction-AssumeRoleWithAction",
            "github-actions-role",
            "GitHubActionsRole",
            "github-oidc-role",
            "GitHubOIDCRole",
        ]
        for pattern in common_patterns:
            try:
                response = iam_client.get_role(RoleName=pattern)
                role_arn = response["Role"]["Arn"]
                print(f"Auto-discovered GitHub OIDC role ARN: {role_arn}")
                # Ensure we return the full ARN, not just the role name
                if not role_arn.startswith("arn:aws:iam::"):
                    print(f"Warning: Expected ARN format but got: {role_arn}")
                    continue
                return role_arn
            except iam_client.exceptions.NoSuchEntityException:
                continue

        # Search through all roles for GitHub/OIDC related roles
        paginator = iam_client.get_paginator("list_roles")
        for page in paginator.paginate():
            for role in page["Roles"]:
                rn = role["RoleName"].lower()
                if any(k in rn for k in ["github", "oidc", "action"]):
                    role_arn = role["Arn"]
                    print(
                        f"Auto-discovered GitHub OIDC role ARN by keyword: {role_arn}"
                    )
                    # Ensure we return the full ARN, not just the role name
                    if not role_arn.startswith("arn:aws:iam::"):
                        print(f"Warning: Expected ARN format but got: {role_arn}")
                        continue
                    return role_arn

        print("No GitHub OIDC role found")
        return ""
    except Exception as e:
        print(f"Error auto-discovering GitHub OIDC role ARN: {str(e)}")
        return ""


def get_environment_config(artifacts_bucket=None):
    if not artifacts_bucket:
        artifacts_bucket = (
            os.environ.get("ARTIFACTS_BUCKET") or auto_discover_artifacts_bucket()
        )
    config = {
        "MLFLOW_TRACKING_ARN": (
            os.environ.get("MLFLOW_TRACKING_ARN") or auto_discover_mlflow_tracking_arn()
        ),
        "INPUT_DATA_PATH": (
            os.environ.get("INPUT_DATA_PATH")
            or auto_discover_input_data_path(artifacts_bucket)
        ),
        "SAGEMAKER_DOMAIN_ARN": (
            os.environ.get("SAGEMAKER_DOMAIN_ARN")
            or auto_discover_sagemaker_domain_arn()
        ),
        "ARTIFACTS_BUCKET": artifacts_bucket,
        "OIDC_ROLE_GITHUB_WORKFLOW": (
            os.environ.get("OIDC_ROLE_GITHUB_WORKFLOW") or auto_discover_oidc_role_arn()
        ),
    }
    print("Environment configuration:")
    for k, v in config.items():
        print(f"  {k}: {'<empty>' if not v else str(v)[:50]}...")
    return config


# ---------- Git operations ----------


class GitOperations:
    def __init__(
        self, org_name, repo_name, template_name, private_repo, template_code_folder
    ):
        print("\n=== Initializing GitOperations ===")
        print(f"Organization Name: {org_name}")
        print(f"Repository Name: {repo_name}")
        print(f"Template Name: {template_name}")
        print(f"Private Repo: {private_repo}")
        print(f"Template Code Folder: {template_code_folder}")

        self.org_name = org_name
        self.repo_name = repo_name
        self.template_name = template_name.lower()
        self.private_repo = private_repo  # "owner/repo"
        self.template_code_folder = template_code_folder
        self.temp_dir = tempfile.gettempdir()

    def _get_git_credentials(self):
        try:
            secret_name = os.environ.get("GITHUB_TOKEN_SECRET_NAME")
            if not secret_name:
                raise Exception("GITHUB_TOKEN_SECRET_NAME environment variable not set")
            secrets_client = boto3.client("secretsmanager")
            response = secrets_client.get_secret_value(SecretId=secret_name)
            return json.loads(response["SecretString"])["token"]
        except Exception as e:
            print(f"Error getting git credentials: {str(e)}")
            raise

    def _download_github_repo_zip(self, org, repo, branch="main"):
        try:
            git_token = self._get_git_credentials()
            headers = _gh_headers(git_token)
            zip_url = f"https://api.github.com/repos/{org}/{repo}/zipball/{branch}"
            print(f"Downloading repository from: {zip_url}")
            response = requests.get(zip_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            if response.status_code != 200:
                raise Exception(
                    f"Failed to download repository: {response.status_code} - {response.text}"
                )
            zip_path = os.path.join(self.temp_dir, f"{repo}.zip")
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return zip_path
        except Exception as e:
            print(f"Error downloading repository: {str(e)}")
            raise

    def _extract_zip_repo(self, zip_path, extract_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            extracted_folders = [
                f
                for f in os.listdir(extract_path)
                if os.path.isdir(os.path.join(extract_path, f))
            ]
            if not extracted_folders:
                raise Exception("No folders found in extracted ZIP")
            return os.path.join(extract_path, extracted_folders[0])
        except Exception as e:
            print(f"Error extracting ZIP: {str(e)}")
            raise

    def _create_or_update_file_via_api(
        self, org, repo, file_path, content, message, branch="main"
    ):
        try:
            git_token = self._get_git_credentials()
            headers = _gh_headers(git_token)

            get_url = f"https://api.github.com/repos/{org}/{repo}/contents/{file_path}"
            get_response = requests.get(get_url, headers=headers, timeout=30)
            # Don't raise for 404 as it's expected when file doesn't exist yet
            if get_response.status_code not in (200, 404):
                get_response.raise_for_status()

            if isinstance(content, str):
                content_encoded = base64.b64encode(content.encode("utf-8")).decode(
                    "utf-8"
                )
            else:
                content_encoded = base64.b64encode(content).decode("utf-8")

            payload = {"message": message, "content": content_encoded, "branch": branch}

            if get_response.status_code == 200:
                existing_file = get_response.json()
                payload["sha"] = existing_file["sha"]
                print(f"Updating existing file: {file_path}")
            else:
                print(f"Creating new file: {file_path}")

            put_url = f"https://api.github.com/repos/{org}/{repo}/contents/{file_path}"
            response = requests.put(put_url, headers=headers, json=payload, timeout=30)
            if response.status_code not in (200, 201):
                raise Exception(
                    f"Failed to create/update file {file_path}: {response.status_code} - {response.text}"
                )
            return response.json()
        except Exception as e:
            print(f"Error creating/updating file {file_path}: {str(e)}")
            raise

    # -------- variables only (secrets removed) --------

    def create_github_variables(self, variables_data: dict):
        """Create/Update GitHub repository variables (create on 201, update on 409 -> 204)."""
        try:
            git_token = self._get_git_credentials()
            headers = _gh_headers(git_token)
            base_url = (
                f"https://api.github.com/repos/{self.private_repo}/actions/variables"
            )

            print(f"\nCreating GitHub repository variables in: {self.private_repo}")

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

                resp = requests.post(
                    base_url,
                    headers=headers,
                    json={"name": name, "value": value},
                    timeout=60,
                )
                resp.raise_for_status()
                if resp.status_code == 201:
                    print(f"✓ Variable created: {name}")
                    continue

                if resp.status_code == 409:
                    upd = requests.patch(
                        f"{base_url}/{name}",
                        headers=headers,
                        json={"value": value},
                        timeout=60,
                    )
                    upd.raise_for_status()
                    if upd.status_code == 204:
                        print(f"✓ Variable updated: {name}")
                    else:
                        print(
                            f"⚠ Failed to update variable {name}: {upd.status_code} {upd.text}"
                        )
                    continue

                print(
                    f"⚠ Failed to create variable {name}: {resp.status_code} {resp.text}"
                )

            print(f"Completed GitHub variables setup for {self.private_repo}")
        except Exception as e:
            print(f"Error creating GitHub variables: {e}")
            print(
                "Skipping variable creation - variables will need to be created manually"
            )

    # -------- syncing content (unchanged) --------

    def sync_template_folder(self, folder_type="model_build"):
        try:
            print(
                f"\nStep 1: Downloading source repository for template {self.template_name}"
            )
            source_zip = self._download_github_repo_zip(self.org_name, self.repo_name)

            source_extract_path = os.path.join(self.temp_dir, "source_extract")
            if os.path.exists(source_extract_path):
                shutil.rmtree(source_extract_path)
            os.makedirs(source_extract_path)

            source_repo_path = self._extract_zip_repo(source_zip, source_extract_path)

            print(f"\nStep 2: Locating {folder_type} folder in source repository")
            folder_found = False
            files_to_sync = []

            template_code_path = os.path.join(
                source_repo_path, self.template_code_folder
            )
            if os.path.exists(template_code_path):
                template_path = os.path.join(template_code_path, self.template_name)
                if os.path.exists(template_path):
                    target_folder_path = os.path.join(template_path, folder_type)
                    if os.path.exists(target_folder_path):
                        print(f"Found {folder_type} folder at: {target_folder_path}")
                        for root, dirs, files in os.walk(target_folder_path):
                            dirs[:] = [d for d in dirs]  # keep hidden dirs too
                            for file in files:
                                src_file = os.path.join(root, file)
                                rel_path = os.path.relpath(
                                    src_file, target_folder_path
                                ).replace(os.sep, "/")
                                files_to_sync.append((src_file, rel_path))
                                print(f"Found file to sync: {rel_path}")
                        folder_found = True

            if not folder_found:
                raise Exception(
                    f"{folder_type} folder not found in expected path: {self.template_code_folder}/{self.template_name}/"
                )

            print(f"\nStep 3: Syncing {len(files_to_sync)} files to private repo")
            private_org, private_repo_name = self.private_repo.split("/")

            for src_file, rel_path in files_to_sync:
                try:
                    with open(src_file, "rb") as f:
                        file_content = f.read()
                    print(f"Syncing file: {rel_path}")
                    self._create_or_update_file_via_api(
                        private_org,
                        private_repo_name,
                        rel_path,
                        file_content,
                        f"Sync {rel_path} from {self.template_name} template",
                    )
                except Exception as e:
                    print(f"Warning: Failed to sync {rel_path}: {str(e)}")

            if os.path.exists(source_zip):
                os.unlink(source_zip)
            if os.path.exists(source_extract_path):
                shutil.rmtree(source_extract_path)

            print(f"\nStep 4: Repository sync completed successfully for {folder_type}")
            return "files_synced"
        except Exception as e:
            print(f"Error in sync_template_folder: {str(e)}")
            raise

    def sync_model_build_folder(self):
        return self.sync_template_folder("model_build")

    def sync_model_deploy_folder(self):
        return self.sync_template_folder("model_deploy")


# ---------- sagemaker helpers (unchanged) ----------


def get_sagemaker_project_details(project_name):
    try:
        sagemaker_client = boto3.client("sagemaker")
        try:
            project_response = sagemaker_client.describe_project(
                ProjectName=project_name
            )
            project_id = project_response.get("ProjectId")
            project_arn = project_response.get("ProjectArn")
            template_name = "esg-benchmarking"
            if "Tags" in project_response:
                for tag in project_response["Tags"]:
                    if tag["Key"] in ["ProjectTemplate", "TemplateName"]:
                        template_name = tag["Value"]
                        break
            print(f"Using template name from SageMaker project: {template_name}")
            return {
                "projectId": project_id,
                "projectArn": project_arn,
                "templateName": template_name,
            }
        except Exception as project_error:
            print(
                f"SageMaker project not found (normal for direct model registration): {str(project_error)}"
            )
            project_id = (
                project_name
                if not project_name.startswith("project-")
                else project_name.replace("project-", "")
            )
            template_name = "esg-benchmarking"
            account_id = boto3.client("sts").get_caller_identity()["Account"]
            region = boto3.session.Session().region_name
            project_arn = (
                f"arn:aws:sagemaker:{region}:{account_id}:project/{project_name}"
            )
            print("Using synthetic project details for direct model registration:")
            print(f"  project_id: {project_id}")
            print(f"  template_name: {template_name}")
            return {
                "projectId": project_id,
                "projectArn": project_arn,
                "templateName": template_name,
            }
    except Exception as e:
        print(f"Error getting SageMaker project details: {str(e)}")
        raise


def get_sagemaker_domain_details(domain_id):
    try:
        sagemaker_client = boto3.client("sagemaker")
        domain_response = sagemaker_client.describe_domain(DomainId=domain_id)
        default_user_settings = domain_response.get("DefaultUserSettings", {})
        execution_role = default_user_settings.get("ExecutionRole")
        artifacts_bucket = os.environ.get("ARTIFACTS_BUCKET")
        if not artifacts_bucket:
            account_id = boto3.client("sts").get_caller_identity()["Account"]
            region = boto3.session.Session().region_name
            artifacts_bucket = f"llmops-sm-artifacts-{account_id}-{region}"
        return {
            "domainArn": domain_response["DomainArn"],
            "executionRole": execution_role,
            "artifactsBucket": artifacts_bucket,
        }
    except Exception as e:
        print(f"Error getting SageMaker domain details: {str(e)}")
        raise


# ---------- lambda handler ----------


def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event, indent=2))
        event_data = event.get("body", event)
        if isinstance(event_data, str):
            event_data = json.loads(event_data)

        project_name = event_data.get("projectName")
        project_id = event_data.get("projectId")
        domain_id = event_data.get("domainId")
        model_artifacts_s3_uri = event_data.get("modelArtifactsS3Uri")
        project_details = event_data.get("projectDetails", {})
        build_repo = event_data.get("buildRepo")
        if not build_repo:
            private_org = os.environ.get("TARGET_GITHUB_ORG")
            build_repo = f"{private_org}/{project_name}-build-repo"

        region = boto3.session.Session().region_name

        print(f"Extracted parameters:")
        print(f"project_name: {project_name}")
        print(f"project_id: {project_id}")
        print(f"domain_id: {domain_id}")
        print(f"build_repo: {build_repo}")
        print(f"region: {region}")

        if not all([project_name, project_id]):
            raise ValueError("Missing required parameters: project_name and project_id")

        sagemaker_project = get_sagemaker_project_details(project_name)
        template_name = sagemaker_project["templateName"]
        if not project_id:
            project_id = sagemaker_project["projectId"]

        if domain_id:
            domain_details = get_sagemaker_domain_details(domain_id)
        else:
            domain_id = os.environ.get("SAGEMAKER_DOMAIN_ID")
            domain_details = (
                get_sagemaker_domain_details(domain_id) if domain_id else {}
            )

        model_package_group_name = f"{template_name}-{project_id}-models"
        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity()["Account"]

        repo_type = "model_deploy" if "deploy" in build_repo.lower() else "model_build"
        env_config = get_environment_config(domain_details.get("artifactsBucket"))

        # Build one dict with EVERYTHING we want available to workflows as variables
        if repo_type == "model_deploy":
            repo_vars = {
                "ARTIFACT_BUCKET": env_config.get("ARTIFACTS_BUCKET")
                or domain_details.get("artifactsBucket"),
                "DEPLOY_ACCOUNT": account_id,
                "MODEL_PACKAGE_GROUP_NAME": model_package_group_name,
                "OIDC_ROLE_GITHUB_WORKFLOW": env_config.get(
                    "OIDC_ROLE_GITHUB_WORKFLOW"
                ),
                "REGION": region,
                "SAGEMAKER_DOMAIN_ARN": env_config.get("SAGEMAKER_DOMAIN_ARN")
                or domain_details.get("domainArn"),
                "SAGEMAKER_PROJECT_ID": project_id,
                "SAGEMAKER_PROJECT_NAME": project_name,
                "MLFLOW_TRACKING_ARN": env_config.get("MLFLOW_TRACKING_ARN", ""),
                "ADAPTER_S3_URI": model_artifacts_s3_uri or "",
                "SAGEMAKER_DOMAIN_ID": domain_id,
                "SAGEMAKER_PROJECT_ARN": sagemaker_project.get("projectArn"),
                "SAGEMAKER_EXECUTION_ROLE_ARN": domain_details.get("executionRole"),
                "AWS_REGION": region,
                "ARTIFACTS_BUCKET": env_config.get("ARTIFACTS_BUCKET")
                or domain_details.get("artifactsBucket"),
                "MODEL_BUCKET_ARN": f"arn:aws:s3:::{env_config.get('ARTIFACTS_BUCKET') or domain_details.get('artifactsBucket')}",
                "TRIGGER_PIPELINE_EXECUTION": "false",
            }
        else:
            sagemaker_pipeline_role_arn = (
                domain_details.get("executionRole")
                or f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMakerServiceCatalogProductsExecutionRole"
            )
            repo_vars = {
                "SAGEMAKER_PROJECT_NAME": project_name,
                "SAGEMAKER_PROJECT_ID": project_id,
                "SAGEMAKER_PIPELINE_ROLE_ARN": sagemaker_pipeline_role_arn,
                "REGION": region,
                "ARTIFACT_BUCKET": env_config.get("ARTIFACTS_BUCKET")
                or domain_details.get("artifactsBucket"),
                "OIDC_ROLE_GITHUB_WORKFLOW": env_config.get(
                    "OIDC_ROLE_GITHUB_WORKFLOW"
                ),
                "MODEL_PACKAGE_GROUP_NAME": model_package_group_name,
                "MLFLOW_TRACKING_ARN": env_config.get("MLFLOW_TRACKING_ARN", ""),
                "INPUT_DATA_PATH": env_config.get("INPUT_DATA_PATH", ""),
                "TRIGGER_PIPELINE_EXECUTION": "false",
                "SAGEMAKER_DOMAIN_ID": domain_id,
                "SAGEMAKER_DOMAIN_ARN": env_config.get("SAGEMAKER_DOMAIN_ARN", ""),
                "SAGEMAKER_PROJECT_ARN": sagemaker_project.get("projectArn"),
                "SAGEMAKER_EXECUTION_ROLE_ARN": domain_details.get("executionRole"),
                "AWS_REGION": region,
                "ARTIFACTS_BUCKET": env_config.get("ARTIFACTS_BUCKET")
                or domain_details.get("artifactsBucket"),
                "ESG_DATASET_S3_PATH": f"s3://{env_config.get('ARTIFACTS_BUCKET') or domain_details.get('artifactsBucket', 'default-bucket')}/datasets/esg/",
                "ESG_BENCHMARK_S3_PATH": f"s3://{env_config.get('ARTIFACTS_BUCKET') or domain_details.get('artifactsBucket', 'default-bucket')}/benchmarks/esg/",
                "MODEL_BASE_NAME": "mistralai/Mistral-7B-Instruct-v0.3",
            }

        print("\nVariables to be created/updated:")
        for k, v in repo_vars.items():
            print(f"{k}: {'<empty>' if not v else str(v)[:50]}...")

        git_ops = GitOperations(
            org_name=os.environ["TEMPLATE_ORG"],
            repo_name=os.environ["TEMPLATE_REPO"],
            template_name=template_name,
            private_repo=build_repo,
            template_code_folder=os.environ["TEMPLATE_FOLDER"],
        )

        # Upsert ALL variables
        git_ops.create_github_variables(repo_vars)

        # Sync files
        print(f"Detected {repo_type} repository, syncing folder")
        changes = (
            git_ops.sync_model_deploy_folder()
            if repo_type == "model_deploy"
            else git_ops.sync_model_build_folder()
        )
        commit_message = (
            "Files were synced to repository" if changes else "No changes to sync"
        )

        return {
            "statusCode": 200,
            "projectName": project_name,
            "projectId": project_id,
            "domainId": domain_id,
            "status": "SUCCESSFUL",
            "buildRepo": build_repo,
            "sagemakerDetails": {
                "projectArn": sagemaker_project.get("projectArn"),
                "domainArn": domain_details.get("domainArn"),
                "executionRole": domain_details.get("executionRole"),
                "artifactsBucket": domain_details.get("artifactsBucket"),
            },
            "additionalInfo": {
                "templateName": template_name,
                "sourceRepo": f"https://github.com/{os.environ['TEMPLATE_ORG']}/{os.environ['TEMPLATE_REPO']}",
                "message": f"Successfully created GitHub variables and {commit_message.lower()}",
                "secretsCreated": [],  # secrets no longer used
                "variablesCreated": list(repo_vars.keys()),
                "modelPackageGroupName": model_package_group_name,
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
