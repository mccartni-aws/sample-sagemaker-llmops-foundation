"""
SusGen ESG Benchmarking SageMaker Pipeline

This module defines the SageMaker pipeline for training ESG models based on the SusGen approach.
The pipeline includes data download, preprocessing, LoRA-based model training, evaluation, and registration.
"""


def create_pipeline_definition():
    """
    Create a pipeline definition with default parameters.

    Returns:
        dict: Pipeline definition
    """
    # Default configuration
    config = {
        "region": "us-east-1",
        "default_bucket": "sagemaker-susgen-pipeline",
        "model_package_group_name": "susgen-esg-models",
        "pipeline_name": "susgen-esg-benchmarking-pipeline",
        "base_job_prefix": "susgen-esg",
    }

    pipeline = get_pipeline(**config)

    return {
        "pipeline": pipeline,
        "config": config,
        "description": "SusGen ESG model training pipeline with LoRA fine-tuning and comprehensive evaluation",
    }


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    esg_data_bucket=None,
    model_package_group_name="susgen-esg-models",
    pipeline_name="susgen-esg-pipeline",
    base_job_prefix="susgen-esg",
    # Direct parameter values instead of ParameterString objects
    preprocessing_instance_type="ml.m5.4xlarge",
    preprocessing_instance_count=1,
    training_instance_type="ml.g5.2xlarge",
    training_instance_count=1,
    evaluate_instance_type="ml.c7i.48xlarge",  # "ml.g5.2xlarge",
    evaluate_instance_count=1,
    model_approval_status="PendingManualApproval",
    base_model_name="unsloth/mistral-7b-v0.3",
    prompt_template="mistral_formal",
    max_length=512,
    num_train_epochs=1,
    learning_rate=2e-4,
    lora_r=16,
    lora_alpha=16,
    hf_token="",
    mlflow_tracking_arn=None,
    input_data_path=None,
):
    """
    Gets a SageMaker ML Pipeline instance for SusGen ESG model training.

    Args:
        region: AWS region
        role: SageMaker Execution Role ARN
        default_bucket: S3 bucket for storing artifacts
        esg_data_bucket: S3 bucket for ESG data storage (if None, uses default_bucket)
        model_package_group_name: Model package group name for registry
        pipeline_name: Name of the pipeline
        base_job_prefix: Prefix for SageMaker jobs
        input_data_path: S3 path to input data (optional)
        mlflow_tracking_arn: MLflow tracking server ARN (if None, will try to discover or use environment variable)

    Returns:
        sagemaker.workflow.pipeline.Pipeline: SageMaker pipeline instance
    """
    import boto3
    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.inputs import TrainingInput
    from sagemaker.model_metrics import MetricsSource, ModelMetrics
    from sagemaker.processing import (
        ProcessingInput,
        ProcessingOutput,
        FrameworkProcessor,
    )

    from sagemaker.pytorch import PyTorch
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.functions import JsonGet
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep
    from sagemaker.workflow.step_collections import RegisterModel
    from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo

    if role is None:
        role = get_execution_role()

    # Create SageMaker session with the provided bucket to prevent auto-creation
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = PipelineSession(
        boto_session=boto_session, default_bucket=default_bucket
    )

    # Use ESG data bucket if provided, otherwise fall back to default bucket
    data_bucket = esg_data_bucket if esg_data_bucket else default_bucket

    # Determine MLflow tracking ARN
    # Priority: 1. Parameter passed to function, 2. Environment variable, 3. Auto-discovery, 4. None
    import os

    if mlflow_tracking_arn is None:
        # Try to get from environment variable (set by GitHub Actions)
        mlflow_tracking_arn = os.environ.get("MLFLOW_TRACKING_ARN")

        if mlflow_tracking_arn is None:
            # Try to auto-discover MLflow tracking server
            try:
                environment = os.environ.get("ENVIRONMENT", "dev")
                tracking_server_name = f"llmops-mlflow-{environment}"

                sagemaker_client = boto3.client("sagemaker", region_name=region)
                response = sagemaker_client.describe_mlflow_tracking_server(
                    TrackingServerName=tracking_server_name
                )
                mlflow_tracking_arn = response["TrackingServerArn"]
                print(
                    f"✅ Auto-discovered MLflow tracking server: {mlflow_tracking_arn}"
                )

            except Exception as e:
                print(f"⚠️ Could not auto-discover MLflow tracking server: {str(e)}")
                print("⚠️ MLflow tracking will be disabled for this pipeline run")
                mlflow_tracking_arn = None

    print(f"🔧 MLflow Tracking ARN: {mlflow_tracking_arn or 'Not configured'}")

    pytorch_version = "2.8"
    pytorch_python_version = "py312"

    pytorch_image_uri_preprocessing = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        image_scope="training",
        version=pytorch_version,
        py_version=pytorch_python_version,
        instance_type=preprocessing_instance_type,
    )

    # # Step 1: Combined Data Download and Preprocessing
    pytorch_data_processor = FrameworkProcessor(
        estimator_cls=PyTorch,
        framework_version=pytorch_version,
        image_uri=pytorch_image_uri_preprocessing,
        instance_type=preprocessing_instance_type,
        instance_count=preprocessing_instance_count,
        base_job_name=f"{base_job_prefix}-data-preprocessing",
        command=["python3"],
        sagemaker_session=sagemaker_session,
        role=role,
    )

    preprocessing_step_args = pytorch_data_processor.run(
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"s3://{data_bucket}/susgen-train",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
                destination=f"s3://{data_bucket}/susgen-validation",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=f"s3://{data_bucket}/susgen-test",
            ),
        ],
        source_dir="source_scripts/preprocess/",
        code="preprocess.py",
        arguments=[
            "--output-dir",
            "/opt/ml/processing",
            "--dataset-name",
            "WHATX/SusGen-30k",
            "--max-length",
            str(max_length),
        ],
    )

    step_data_prep = ProcessingStep(
        name="SusGenDataPreprocessing", step_args=preprocessing_step_args
    )

    # Step 3: Model Training with LoRA
    model_path = f"s3://{default_bucket}/susgen-training-jobs"

    hyperparameters = {
        "model-name": base_model_name,
        "num-train-epochs": num_train_epochs,
        "max-steps": 100,
        "per-device-train-batch-size": 4,
        "per-device-eval-batch-size": 4,
        "gradient-accumulation-steps": 2,
        "learning-rate": learning_rate,
        "warmup-steps": 5,
        "logging-steps": 10,
        "max-seq-length": max_length,
        "lora-r": lora_r,
        "lora-alpha": lora_alpha,
        "lora-dropout": 0,
        "load-in-4bit": True,
        "optim": "adamw_8bit",
        "seed": 42,
        "save-strategy": "no",
    }

    pytorch_image_uri_training = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        image_scope="training",
        version=pytorch_version,
        py_version=pytorch_python_version,
        instance_type=training_instance_type,
    )

    pytorch_estimator = PyTorch(
        entry_point="train.py",
        source_dir="source_scripts/training",
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        role=role,
        image_uri=pytorch_image_uri_training,
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}-susgen-training",
        disable_profiler=True,
        debugger_hook_config=False,
        environment=(
            {"MLFLOW_TRACKING_ARN": mlflow_tracking_arn} if mlflow_tracking_arn else {}
        ),
    )

    train_step_args = pytorch_estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_data_prep.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="application/json",
            ),
            "validation": TrainingInput(
                s3_data=step_data_prep.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="application/json",
            ),
        },
    )

    step_train = TrainingStep(name="SusGenModelTraining", step_args=train_step_args)

    pytorch_image_uri_evaluate = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        image_scope="training",
        version=pytorch_version,
        py_version=pytorch_python_version,
        instance_type=evaluate_instance_type,
    )

    # Step 4: Model Evaluation
    pytorch_eval_processor = FrameworkProcessor(
        estimator_cls=PyTorch,
        framework_version=pytorch_version,
        image_uri=pytorch_image_uri_evaluate,
        instance_type=evaluate_instance_type,
        instance_count=evaluate_instance_count,
        base_job_name=f"{base_job_prefix}-susgen-eval",
        command=["python3"],
        sagemaker_session=sagemaker_session,
        role=role,
    )

    evaluation_report = PropertyFile(
        name="SusGenEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluate_step_args = pytorch_eval_processor.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_data_prep.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source="/opt/ml/processing/evaluation"
            ),
        ],
        source_dir="source_scripts/evaluate/",
        code="evaluate.py",
        arguments=[
            "--model-path",
            "/opt/ml/processing/model",
            "--output-path",
            "/opt/ml/processing/evaluation",
            "--num-batches",
            "10",
            "--batch-size",
            "4",
            "--max-seq-length",
            str(max_length),
            "--max-new-tokens",
            "16",
            "--load-in-4bit",
            "true",
        ],
    )

    step_eval = ProcessingStep(
        name="SusGenModelEvaluation",
        step_args=evaluate_step_args,
        property_files=[evaluation_report],
    )

    # Model metrics for registration
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                    "S3Uri"
                ]
            ),
            content_type="application/json",
        )
    )

    # Step 5: Model Registration
    step_register = RegisterModel(
        name="SusGenModelRegistration",
        estimator=pytorch_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv", "application/json"],
        response_types=["text/csv", "application/json"],
        inference_instances=["ml.g5.2xlarge", "ml.m5.large", "ml.g4dn.xlarge"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        description=(
            "SusGen ESG model trained with LoRA fine-tuning - "
            "automatically registered from pipeline"
        ),
        model_card=None,
        # Add additional metadata to help identify this as a pipeline-registered model
        customer_metadata_properties={
            "TrainingJobName": step_train.properties.TrainingJobName,
            "PipelineName": f"{pipeline_name}-esg-pipeline",
            "ModelSource": "SageMaker-Pipeline-Automatic",
            "BaseModel": base_model_name,
            "PromptTemplate": prompt_template,
        },
    )

    # Condition step for model quality (based on accuracy score)
    # Using >= 0.0 so it always passes (even with 0% accuracy)
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="rouge_metrics.rougeL.value",
        ),
        right=0.0,  # Accept any accuracy >= 0% (always passes)
    )

    step_cond = ConditionStep(
        name="SusGenModelQualityCheck",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    # Pipeline instance
    pipeline = Pipeline(
        name=f"{pipeline_name}-esg-pipeline",
        parameters=[],  # Empty parameters list since we use direct values
        steps=[step_data_prep, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )

    return pipeline
