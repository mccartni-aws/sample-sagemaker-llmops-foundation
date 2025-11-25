#!/usr/bin/env python3
"""
SageMaker LLMOps Platform CDK Application
Main entry point for AWS CDK infrastructure deployment
"""

import os
import sys
import aws_cdk as cdk
from aws_cdk import Environment

from llmops_sm.stacks.sagemaker_domain_stack import SageMakerDomainStack
from llmops_sm.stacks.main_stack import LlmOpsSmStack
from llmops_sm.stacks.observability_stack import ObservabilityStack
from llmops_sm.config import get_config, validate_config


def validate_github_token():
    """Validate that GitHub token is provided as environment variable"""
    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        print("\n" + "=" * 80)
        print("❌ ERROR: GitHub Personal Access Token (PAT) is required!")
        print("=" * 80)
        print()
        print(
            "🔑 You must set the GITHUB_TOKEN environment variable with your GitHub PAT."
        )
        print()
        print("📋 Steps to fix this:")
        print("   1. Generate a GitHub Personal Access Token:")
        print("      • Go to: https://github.com/settings/tokens")
        print("      • Click 'Generate new token (classic)'")
        print("      • Select these scopes:")
        print("        ✓ repo (Full control of private repositories)")
        print("        ✓ workflow (Update GitHub Action workflows)")
        print("        ✓ admin:org (Full control of orgs) - if using organization")
        print()
        print("   2. Set the environment variable:")
        print("      export GITHUB_TOKEN='ghp_your_token_here'")
        print()
        print("   3. Re-run the deployment:")
        print("      make deploy")
        print()
        print("🔒 Security Note:")
        print(
            "   The token will be securely stored in AWS Secrets Manager during deployment."
        )
        print("   It will NOT be stored in your code or CDK outputs.")
        print()
        print("=" * 80)
        sys.exit(1)

    # Validate token format (GitHub PATs start with 'ghp_', 'gho_', 'ghu_', or 'ghs_')
    if not (
        github_token.startswith(("ghp_", "gho_", "ghu_", "ghs_"))
        and len(github_token) >= 40
    ):
        print("\n" + "=" * 80)
        print("❌ ERROR: Invalid GitHub token format!")
        print("=" * 80)
        print()
        print("🔍 The provided GITHUB_TOKEN does not appear to be a valid GitHub PAT.")
        print()
        print("✅ Valid GitHub tokens should:")
        print("   • Start with 'ghp_', 'gho_', 'ghu_', or 'ghs_'")
        print("   • Be at least 40 characters long")
        print()
        print("🔄 Please generate a new token at: https://github.com/settings/tokens")
        print()
        print("=" * 80)
        sys.exit(1)

    print("✅ GitHub token validation passed")
    return github_token


def main():
    """Main function to create and deploy CDK app"""

    # Initialize CDK app
    app = cdk.App()

    # Get environment from context or environment variable
    environment = app.node.try_get_context("environment") or os.getenv(
        "ENVIRONMENT", "dev"
    )

    # Validate GitHub token before proceeding with deployment
    github_token = validate_github_token()

    # Load and validate configuration
    config = get_config(environment)

    # Override config with environment variables if provided
    target_org = os.getenv("TARGET_GITHUB_ORG")
    if target_org:
        config.git.target_github_org = target_org
        print(f"✅ Using TARGET_GITHUB_ORG from environment: {target_org}")

    # Check for TEMPLATE_ORG first (used by Lambda functions), then TEMPLATE_GITHUB_ORG for backwards compatibility
    template_org = os.getenv("TEMPLATE_ORG") or os.getenv("TEMPLATE_GITHUB_ORG")
    if template_org:
        config.git.template_github_org = template_org
        env_var_used = (
            "TEMPLATE_ORG" if os.getenv("TEMPLATE_ORG") else "TEMPLATE_GITHUB_ORG"
        )
        print(f"✅ Using {env_var_used} from environment: {template_org}")

    # Validate configuration (will fail if target_github_org is not set)
    validate_config(config)

    # Get AWS account and region from environment or CDK context
    account = os.getenv("CDK_DEFAULT_ACCOUNT") or app.node.try_get_context("account")
    region = (
        os.getenv("CDK_DEFAULT_REGION")
        or app.node.try_get_context("region")
        or config.aws.default_region
    )

    # Create CDK environment
    env = Environment(account=account, region=region)

    # Create SageMaker Domain Stack (must be created first)
    domain_stack_name = f"LlmOpsSm-Domain-{environment}"
    domain_stack = SageMakerDomainStack(
        app,
        domain_stack_name,
        config=config,
        env=env,
        description=f"SageMaker Domain for LLMOps Platform - {environment.upper()} environment",
    )

    # Create main LLMOps infrastructure stack (depends on domain stack)
    main_stack_name = f"LlmOpsSm-Infrastructure-{environment}"
    main_stack = LlmOpsSmStack(
        app,
        main_stack_name,
        config=config,
        domain_stack=domain_stack,  # Pass domain stack for dependencies
        env=env,
        description=f"LLMOps Infrastructure Platform - {environment.upper()} environment",
    )

    # Create observability stack (depends on domain stack)
    observability_stack_name = f"LlmOpsSm-Observability-{environment}"
    observability_stack = ObservabilityStack(
        app,
        observability_stack_name,
        config=config,
        domain_stack=domain_stack,  # Pass domain stack for dependencies
        env=env,
        description=f"LLMOps Observability Stack - {environment.upper()} environment",
    )

    # Create dependency relationships
    main_stack.add_dependency(domain_stack)
    observability_stack.add_dependency(domain_stack)
    # Note: main_stack and observability_stack can be deployed in parallel

    # Add tags to all resources
    for stack in [domain_stack, main_stack, observability_stack]:
        cdk.Tags.of(stack).add("Project", "SageMaker-LLMOps")
        cdk.Tags.of(stack).add("Environment", environment)
        cdk.Tags.of(stack).add("ManagedBy", "CDK")
        cdk.Tags.of(stack).add("UseCase", "ESG-Benchmarking")

    # Synthesize the app
    app.synth()


if __name__ == "__main__":
    main()
