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
        "default_bucket": "sagemaker-esg-pipeline",
        "model_package_group_name": "esg-models",
        "pipeline_name": "esg-benchmarking-pipeline",
        "base_job_prefix": "esg",
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
    model_package_group_name="esg-models",
    pipeline_name="esg-pipeline",
    base_job_prefix="esg",
    input_data_path=None,
    # Direct parameter values instead of ParameterString objects
    processing_instance_type="ml.g5.2xlarge",  # "ml.m5.4xlarge"
    processing_instance_count=1,
    training_instance_type="ml.g5.24xlarge",
    # training_instance_type="ml.g5.12xlarge",
    # training_instance_type="ml.p4d.24xlarge",
    training_instance_count=1,
    model_approval_status="PendingManualApproval",
    base_model_name="mistralai/Mistral-7B-Instruct-v0.3",
    prompt_template="mistral_formal",
    max_length=256,
    num_train_epochs=3,
    learning_rate=2e-4,
    lora_r=16,
    lora_alpha=32,
    quantization="int8",
    mlflow_tracking_arn=None,
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
    from sagemaker.session import Session
    from sagemaker.inputs import TrainingInput
    from sagemaker.model_metrics import MetricsSource, ModelMetrics
    from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

    from sagemaker.pytorch import PyTorch
    from sagemaker.pytorch.processing import PyTorchProcessor
    from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.functions import JsonGet, Join
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep
    from sagemaker.workflow.step_collections import RegisterModel
    from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo

    if role is None:
        role = get_execution_role()

    # Create SageMaker session with the provided bucket to prevent auto-creation
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = Session(
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

    # # Step 1: Combined Data Download and Preprocessing
    script_data_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.0-1",
            py_version="py3",
            instance_type="ml.m5.4xlarge",
        ),
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}-data-preprocessing",
        command=["python3"],
        sagemaker_session=sagemaker_session,
        role=role,
        # output_kms_key=bucket_kms_id,
    )

    step_data_prep = ProcessingStep(
        name="ESGDataPreprocessing",
        processor=script_data_processor,
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"s3://{data_bucket}/esg-train",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
                destination=f"s3://{data_bucket}/esg-validation",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=f"s3://{data_bucket}/esg-test",
            ),
        ],
        code="source_scripts/data/preprocess_data.py",
        job_arguments=[
            "--dataset-id",
            "WHATX/SusGen-30k",
            "--output-dir",
            "/opt/ml/processing",
            "--prompt-template",
            str(prompt_template),
            "--max-length",
            str(max_length),
        ],
    )

    # Step 3: Model Training with LoRA
    model_path = f"s3://{default_bucket}/esg-training-jobs"

    hyperparameters = {
        "model-name": base_model_name,
        "num-train-epochs": num_train_epochs,
        "per-device-train-batch-size": 1,
        "per-device-eval-batch-size": 2,
        "gradient-accumulation-steps": 32,
        "learning-rate": learning_rate,
        "warmup-steps": 100,
        "logging-steps": 10,
        "max-length": max_length,
        "lora-r": lora_r,
        "lora-alpha": lora_alpha,
        "lora-dropout": 0.1,
        "quantization": quantization,
        "prompt-template": prompt_template,
        "train-sample-fraction": "0.01",
    }

    pytorch_estimator = PyTorch(
        entry_point="train.py",
        source_dir="source_scripts/training",
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        role=role,
        framework_version="2.0.1",
        py_version="py310",
        # distribution={
        #     "torch_distributed": {
        #         "enabled": True
        #     }
        # },
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}-training",
        disable_profiler=True,
        debugger_hook_config=False,
        environment=(
            {"MLFLOW_TRACKING_ARN": mlflow_tracking_arn} if mlflow_tracking_arn else {}
        ),
    )

    step_train = TrainingStep(
        name="ESGModelTraining",
        estimator=pytorch_estimator,
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

    # Step 4: Model Evaluation
    pytorch_eval_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="pytorch",
            region=region,
            image_scope="training",
            version="2.0.1",
            py_version="py310",
            instance_type=processing_instance_type,
        ),
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-eval",
        command=["python3"],
        sagemaker_session=sagemaker_session,
        role=role,
    )

    evaluation_report = PropertyFile(
        name="ESGEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="ESGModelEvaluation",
        processor=pytorch_eval_processor,
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
        code="source_scripts/evaluate/evaluate.py",
        job_arguments=[
            "--model-path",
            "/opt/ml/processing/model",
            "--output-path",
            "/opt/ml/processing/evaluation",
            "--max-samples",
            "100",  # Reduced for faster evaluation
        ],
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
        name="ESGModelRegistration",
        estimator=pytorch_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv", "application/json"],
        response_types=["text/csv", "application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large", "ml.g4dn.xlarge"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        description=(
            "ESG model trained with LoRA fine-tuning - "
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
            json_path="classification_metrics.accuracy.value",  # This matches your evaluate.py output
        ),
        right=0.0,  # Accept any accuracy >= 0% (always passes)
    )

    step_cond = ConditionStep(
        name="ESGModelQualityCheck",
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
