"""
SusGen Model Training Script
This script trains a transformer-based model for ESG sustainability report generation
using the SusGen approach with LoRA fine-tuning, adapted for SageMaker.
Based on the original SusGen repository implementation.
"""

import argparse
import logging
import os
import warnings

from model_setup import setup_model, ModelConfig
from data_utils import load_datasets
from trainer_utils import create_trainer, setup_mlflow_tracking, TrainConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SusGen ESG model")
    parser.add_argument("--model-name", type=str, default="unsloth/mistral-7b-v0.3")
    parser.add_argument(
        "--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"
        ),
    )
    parser.add_argument("--train-sample-fraction", type=float, default=0.05)
    parser.add_argument("--eval-sample-fraction", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0)
    parser.add_argument(
        "--load-in-4bit", type=lambda x: str(x).lower() == "true", default=True
    )
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save-strategy", type=str, default="no")
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    if args.train_sample_fraction:
        logger.info(
            f"⚡ FAST TRAIN ENABLED: training on {args.train_sample_fraction * 100:.2f}% of data"
        )
    if args.eval_sample_fraction:
        logger.info(
            f"⚡ FAST EVAL ENABLED: evaluating on {args.eval_sample_fraction * 100:.2f}% of data"
        )

    setup_mlflow_tracking("susgen-esg-training")

    model_config = ModelConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model, tokenizer = setup_model(model_config)

    train_dataset, eval_dataset = load_datasets(
        args.train_data,
        args.val_data,
        args.train_sample_fraction,
        args.eval_sample_fraction,
        args.seed,
    )

    train_config = TrainConfig(
        model_dir=args.model_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim=args.optim,
        seed=args.seed,
        save_strategy=args.save_strategy,
    )
    trainer = create_trainer(
        model, tokenizer, train_dataset, eval_dataset, train_config
    )

    trainer.train()

    logger.info("Saving model...")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
