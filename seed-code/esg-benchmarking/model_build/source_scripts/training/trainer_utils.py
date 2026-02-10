import logging
import mlflow
import os
from dataclasses import dataclass
from unsloth import is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    model_dir: str
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 5
    num_train_epochs: int = 1
    max_steps: int = 100
    learning_rate: float = 2e-4
    logging_steps: int = 10
    optim: str = "adamw_8bit"
    seed: int = 3407
    save_strategy: str = "no"


def setup_mlflow_tracking(experiment_name: str = "susgen-esg-training") -> bool:
    """
    Configure MLflow to use the SageMaker MLflow Tracking Server via its ARN.
    Requires:
      - mlflow==3.0.0
      - sagemaker-mlflow==0.1.0
    Pass the ARN via env: MLFLOW_TRACKING_ARN
    """
    try:
        tracking_arn = os.environ.get("MLFLOW_TRACKING_ARN")
        if not tracking_arn:
            logger.warning("MLFLOW_TRACKING_ARN not set; MLflow disabled.")
            return False

        mlflow.set_tracking_uri(tracking_arn)

        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment set to: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set experiment '{experiment_name}': {e}")

        # Prefer transformers autolog if available
        try:
            import mlflow.transformers as mlflow_transformers  # noqa: F401

            mlflow_transformers.autolog()
            logger.info("Enabled mlflow.transformers.autolog()")
        except Exception:
            mlflow.autolog()
            logger.info("Enabled mlflow.autolog()")

        logger.info(f"MLflow tracking set to ARN: {tracking_arn}")
        return True

    except Exception as e:
        logger.warning(f"MLflow setup skipped: {e}")
        return False


def create_trainer(model, tokenizer, train_dataset, eval_dataset, args):
    """Create and configure SFTTrainer."""
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.logging_steps,
            optim=args.optim,
            seed=args.seed,
            output_dir=args.model_dir,
            report_to="mlflow",
            save_strategy=args.save_strategy,
        ),
    )
