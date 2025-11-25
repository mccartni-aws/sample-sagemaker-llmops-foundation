"""
Observability Stack for LLMOps Platform
Creates MLflow tracking server, CloudWatch dashboards, and notification systems
"""

from typing import Dict, Any
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_cloudwatch_actions as cw_actions,
    aws_sns_subscriptions as sns_subscriptions,
    aws_sagemaker as sagemaker,
    aws_rds as rds,
    aws_s3 as s3,
    aws_iam as iam,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_sns as sns,
    aws_ses as ses,
    aws_lambda as lambda_,
    aws_events as events,
    aws_events_targets as targets,
    aws_secretsmanager as secretsmanager,
    CfnOutput,
    Duration,
    RemovalPolicy,
)
from constructs import Construct

from llmops_sm.config import PlatformConfig
from llmops_sm.stacks.sagemaker_domain_stack import SageMakerDomainStack


class ObservabilityStack(Stack):
    """Stack for MLflow tracking server and observability features"""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        config: PlatformConfig,
        domain_stack: SageMakerDomainStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.config = config
        self.domain_stack = domain_stack

        # Get VPC from domain stack
        self.vpc = domain_stack.vpc

        # Create MLflow components
        # self.mlflow_database = self._create_mlflow_database()
        self.mlflow_server = self._create_mlflow_tracking_server()

        # Create notification systems
        self.notification_topic = self._create_sns_topic()
        # self.ses_configuration = self._create_ses_configuration()
        self.slack_webhook_secret = self._create_slack_webhook_secret()

        # Create notification Lambda
        self.notification_lambda = self._create_notification_lambda()

        # Create CloudWatch dashboard
        self.dashboard = self._create_cloudwatch_dashboard()

        # Create custom metrics and alarms
        self._create_custom_alarms()

        # Create EventBridge rules for notifications
        self._create_notification_rules()

        # Create outputs
        self._create_outputs()

    def _create_mlflow_database(self) -> rds.DatabaseInstance:
        """Create RDS MySQL database for MLflow metadata"""

        # Create DB subnet group
        db_subnet_group = rds.SubnetGroup(
            self,
            "MLflowDbSubnetGroup",
            description="Subnet group for MLflow database",
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
        )

        # Create security group for database
        db_security_group = ec2.SecurityGroup(
            self,
            "MLflowDbSecurityGroup",
            vpc=self.vpc,
            description="Security group for MLflow database",
            allow_all_outbound=False,
        )

        # Create database instance
        database = rds.DatabaseInstance(
            self,
            "MLflowDatabase",
            engine=rds.DatabaseInstanceEngine.mysql(
                version=rds.MysqlEngineVersion.VER_8_0
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE3, ec2.InstanceSize.MICRO
            ),
            database_name="mlflow",
            credentials=rds.Credentials.from_generated_secret(
                "mlflow-admin", secret_name="mlflow-db-credentials"
            ),
            vpc=self.vpc,
            subnet_group=db_subnet_group,
            security_groups=[db_security_group],
            allocated_storage=20,
            storage_encrypted=True,
            backup_retention=Duration.days(7),
            deletion_protection=False if self.config.environment == "dev" else True,
            removal_policy=(
                RemovalPolicy.DESTROY
                if self.config.environment == "dev"
                else RemovalPolicy.RETAIN
            ),
        )

        return database

    def _create_mlflow_tracking_server(self) -> sagemaker.CfnMlflowTrackingServer:
        """Create managed MLflow tracking server"""

        # Create managed MLflow tracking server
        mlflow_server = sagemaker.CfnMlflowTrackingServer(
            self,
            "ManagedMLflowServer",
            tracking_server_name=f"llmops-mlflow-{self.config.environment}",
            artifact_store_uri=f"s3://{self.domain_stack.artifacts_bucket.bucket_name}/mlflow-artifacts",
            role_arn=self.domain_stack.execution_role.role_arn,
            tracking_server_size="Small",  # Small, Medium, Large
            weekly_maintenance_window_start="Sat:03:00",  # Optional maintenance window
        )

        return mlflow_server

    def _create_sns_topic(self) -> sns.Topic:
        """Create SNS topic for notifications"""

        topic = sns.Topic(
            self,
            "LLMOpsNotificationTopic",
            topic_name=f"llmops-notifications-{self.config.environment}",
            display_name="LLMOps Platform Notifications",
        )

        # Add email subscription if configured
        if self.config.notification_email:
            topic.add_subscription(
                sns_subscriptions.EmailSubscription(self.config.notification_email)
            )

        return topic

    def _create_ses_configuration(self) -> Dict[str, Any]:
        """Create SES configuration for email notifications"""

        # Create SES identity (domain or email)
        # Note: You'll need to verify this manually in SES console
        ses_identity = ses.EmailIdentity(
            self,
            "LLMOpsSESIdentity",
            identity=ses.Identity.email(self.config.notification_email),
        )

        return {
            "identity": ses_identity,
            "from_email": self.config.notification_email,
        }

    def _create_slack_webhook_secret(self) -> secretsmanager.Secret:
        """Create secret for Slack webhook URL"""

        secret = secretsmanager.Secret(
            self,
            "SlackWebhookSecret",
            secret_name=f"llmops-slack-webhook-{self.config.environment}",
            description="Slack webhook URL for LLMOps notifications",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"webhook_url": ""}',
                generate_string_key="webhook_url",
                exclude_characters='"',
            ),
        )

        return secret

    def _create_notification_lambda(self) -> lambda_.Function:
        """Create Lambda function for sending notifications"""

        # Create Lambda execution role
        lambda_role = iam.Role(
            self,
            "NotificationLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # Add scoped permissions for SES
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="SESScopedAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ses:SendEmail",
                    "ses:SendRawEmail",
                ],
                resources=[
                    # Scope to specific verified identities
                    f"arn:aws:ses:{self.region}:{self.account}:identity/*",
                ],
            )
        )

        # Add scoped permissions for SNS
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="SNSScopedAccess",
                effect=iam.Effect.ALLOW,
                actions=["sns:Publish"],
                resources=[
                    # Scope to the notification topic
                    self.notification_topic.topic_arn,
                ],
            )
        )

        # Add scoped permissions for Secrets Manager
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="SecretsManagerScopedAccess",
                effect=iam.Effect.ALLOW,
                actions=["secretsmanager:GetSecretValue"],
                resources=[
                    # Scope to the Slack webhook secret
                    self.slack_webhook_secret.secret_arn,
                ],
            )
        )

        # Create Lambda function
        notification_lambda = lambda_.Function(
            self,
            "NotificationLambda",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="lambda_function.lambda_handler",
            code=lambda_.Code.from_inline(
                """
import json
import os
import boto3
import requests
from datetime import datetime

def lambda_handler(event, context):
    try:
        print(f"Received event: {json.dumps(event, indent=2)}")

        # Extract notification details
        source = event.get('source', 'llmops-platform')
        detail_type = event.get('detail-type', 'Notification')
        detail = event.get('detail', {})

        # Format message
        message = format_message(source, detail_type, detail)

        # Send notifications
        send_email_notification(message, detail_type)
        send_slack_notification(message, detail_type)
        send_sns_notification(message, detail_type)

        return {
            'statusCode': 200,
            'body': json.dumps('Notifications sent successfully')
        }

    except Exception as e:
        print(f"Error sending notifications: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

def format_message(source, detail_type, detail):
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

    message = {
        'timestamp': timestamp,
        'source': source,
        'type': detail_type,
        'details': detail
    }

    # Create formatted text
    if 'model' in detail_type.lower():
        message['text'] = f"🤖 Model Event: {detail_type}\\n" + \\
                         f"Project: {detail.get('projectName', 'Unknown')}\\n" + \\
                         f"Model: {detail.get('modelName', 'Unknown')}\\n" + \\
                         f"Status: {detail.get('status', 'Unknown')}\\n" + \\
                         f"Time: {timestamp}"
    elif 'pipeline' in detail_type.lower():
        message['text'] = f"⚙️ Pipeline Event: {detail_type}\\n" + \\
                         f"Pipeline: {detail.get('pipelineName', 'Unknown')}\\n" + \\
                         f"Status: {detail.get('status', 'Unknown')}\\n" + \\
                         f"Time: {timestamp}"
    else:
        message['text'] = f"📢 {detail_type}\\n" + \\
                         f"Details: {json.dumps(detail, indent=2)}\\n" + \\
                         f"Time: {timestamp}"

    return message

def send_email_notification(message, subject):
    try:
        ses_client = boto3.client('ses')

        ses_client.send_email(
            Source='llmops-notifications@your-domain.com',  # Update this
            Destination={
                'ToAddresses': ['your-email@example.com']  # Update this
            },
            Message={
                'Subject': {'Data': f"LLMOps Alert: {subject}"},
                'Body': {
                    'Text': {'Data': message['text']},
                    'Html': {'Data': f"<pre>{message['text']}</pre>"}
                }
            }
        )
        print("Email notification sent")

    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def send_slack_notification(message, title):
    try:
        secrets_client = boto3.client('secretsmanager')

        # Get Slack webhook URL
        secret_response = secrets_client.get_secret_value(
            SecretId='llmops-slack-webhook-dev'  # Update for your environment
        )
        webhook_url = json.loads(secret_response['SecretString'])['webhook_url']

        if not webhook_url:
            print("Slack webhook URL not configured")
            return

        # Send to Slack
        slack_payload = {
            'text': f"LLMOps Platform Alert: {title}",
            'attachments': [
                {
                    'color': 'good' if 'success' in title.lower() else 'warning',
                    'fields': [
                        {
                            'title': 'Details',
                            'value': message['text'],
                            'short': False
                        }
                    ],
                    'ts': int(datetime.utcnow().timestamp())
                }
            ]
        }

        response = requests.post(webhook_url, json=slack_payload)
        if response.status_code == 200:
            print("Slack notification sent")
        else:
            print(f"Failed to send Slack notification: {response.status_code}")

    except Exception as e:
        print(f"Failed to send Slack notification: {str(e)}")

def send_sns_notification(message, subject):
    try:
        sns_client = boto3.client('sns')

        sns_client.publish(
            TopicArn=os.environ['SNS_TOPIC_ARN'],
            Subject=f"LLMOps: {subject}",
            Message=message['text']
        )
        print("SNS notification sent")

    except Exception as e:
        print(f"Failed to send SNS notification: {str(e)}")
"""
            ),
            timeout=Duration.minutes(5),
            environment={
                "SNS_TOPIC_ARN": self.notification_topic.topic_arn,
                "SLACK_SECRET_NAME": self.slack_webhook_secret.secret_name,
                "ENVIRONMENT": self.config.environment,
            },
            role=lambda_role,
        )

        # Grant permissions
        self.notification_topic.grant_publish(notification_lambda)
        self.slack_webhook_secret.grant_read(notification_lambda)

        return notification_lambda

    def _create_cloudwatch_dashboard(self) -> cloudwatch.Dashboard:
        """Create CloudWatch dashboard for LLMOps monitoring"""

        dashboard = cloudwatch.Dashboard(
            self,
            "LLMOpsDashboard",
            dashboard_name=f"LLMOps-Platform-{self.config.environment}",
            period_override=cloudwatch.PeriodOverride.AUTO,
        )

        # Add widgets for different metrics
        dashboard.add_widgets(
            # Lambda function metrics
            cloudwatch.GraphWidget(
                title="Lambda Function Performance",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/Lambda",
                        metric_name="Duration",
                        dimensions_map={"FunctionName": "CheckProjectStatusFunction"},
                        statistic="Average",
                    ),
                    cloudwatch.Metric(
                        namespace="AWS/Lambda",
                        metric_name="Errors",
                        dimensions_map={"FunctionName": "CheckProjectStatusFunction"},
                        statistic="Sum",
                    ),
                ],
                width=12,
                height=6,
            )
        )

        dashboard.add_widgets(
            # Step Functions metrics
            cloudwatch.GraphWidget(
                title="Step Functions Executions",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/States",
                        metric_name="ExecutionsSucceeded",
                        dimensions_map={"StateMachineArn": "*"},
                        statistic="Sum",
                    ),
                    cloudwatch.Metric(
                        namespace="AWS/States",
                        metric_name="ExecutionsFailed",
                        dimensions_map={"StateMachineArn": "*"},
                        statistic="Sum",
                    ),
                ],
                width=12,
                height=6,
            )
        )

        dashboard.add_widgets(
            # SageMaker metrics
            cloudwatch.GraphWidget(
                title="SageMaker Training Jobs",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/SageMaker",
                        metric_name="TrainingJobSucceeded",
                        statistic="Sum",
                    ),
                    cloudwatch.Metric(
                        namespace="AWS/SageMaker",
                        metric_name="TrainingJobFailed",
                        statistic="Sum",
                    ),
                ],
                width=12,
                height=6,
            )
        )

        return dashboard

    def _create_custom_alarms(self):
        """Create CloudWatch alarms for key metrics"""

        # Lambda function error alarm
        lambda_error_alarm = cloudwatch.Alarm(
            self,
            "LambdaErrorAlarm",
            metric=cloudwatch.Metric(
                namespace="AWS/Lambda", metric_name="Errors", statistic="Sum"
            ),
            threshold=5,
            evaluation_periods=2,
            alarm_description="Lambda function errors exceeded threshold",
        )

        lambda_error_alarm.add_alarm_action(
            cw_actions.SnsAction(self.notification_topic)
        )

        # Step Functions failure alarm
        stepfunctions_alarm = cloudwatch.Alarm(
            self,
            "StepFunctionsFailureAlarm",
            metric=cloudwatch.Metric(
                namespace="AWS/States", metric_name="ExecutionsFailed", statistic="Sum"
            ),
            threshold=3,
            evaluation_periods=1,
            alarm_description="Step Functions executions failing",
        )

        stepfunctions_alarm.add_alarm_action(
            cw_actions.SnsAction(self.notification_topic)
        )

    def _create_notification_rules(self):
        """Create EventBridge rules for notification triggers"""

        # Model approval notifications
        model_approval_rule = events.Rule(
            self,
            "ModelApprovalNotificationRule",
            rule_name=f"llmops-model-approval-notifications-{self.config.environment}",
            description="Send notifications when models are approved",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Package State Change"],
                detail={"ModelApprovalStatus": ["Approved"]},
            ),
        )

        model_approval_rule.add_target(targets.LambdaFunction(self.notification_lambda))

        # Training job completion notifications
        training_job_rule = events.Rule(
            self,
            "TrainingJobNotificationRule",
            rule_name=f"llmops-training-job-notifications-{self.config.environment}",
            description="Send notifications when training jobs complete",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Training Job State Change"],
                detail={"TrainingJobStatus": ["Completed", "Failed", "Stopped"]},
            ),
        )

        training_job_rule.add_target(targets.LambdaFunction(self.notification_lambda))

    def _create_outputs(self):
        """Create CloudFormation outputs"""

        CfnOutput(
            self,
            "MLflowTrackingServerName",
            value=self.mlflow_server.tracking_server_name,
            description="MLflow Tracking Server Name",
            export_name=f"{self.stack_name}-MLflowName",
        )

        CfnOutput(
            self,
            "MLflowTrackingServerArn",
            value=self.mlflow_server.attr_tracking_server_arn,
            description="MLflow Tracking Server ARN",
            export_name=f"{self.stack_name}-MLflowArn",
        )

        CfnOutput(
            self,
            "NotificationTopicArn",
            value=self.notification_topic.topic_arn,
            description="SNS Topic for Notifications",
            export_name=f"{self.stack_name}-NotificationTopic",
        )

        CfnOutput(
            self,
            "CloudWatchDashboardUrl",
            value=f"https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={self.dashboard.dashboard_name}",
            description="CloudWatch Dashboard URL",
            export_name=f"{self.stack_name}-DashboardUrl",
        )

        CfnOutput(
            self,
            "SlackWebhookSecretName",
            value=self.slack_webhook_secret.secret_name,
            description="Slack Webhook Secret Name (update this in Secrets Manager)",
            export_name=f"{self.stack_name}-SlackSecret",
        )
