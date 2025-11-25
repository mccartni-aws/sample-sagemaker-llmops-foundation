"""
ESG Model Training Script
This script trains a transformer-based model for ESG sustainability report generation
using LoRA fine-tuning, adapted for SageMaker.
"""

import argparse
import json
import logging
import os
import warnings
from datetime import datetime

import boto3
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from datasets import Dataset
from huggingface_hub import login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def setup_distributed():
    """Setup distributed training environment."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        logger.info(
            f"🚀 Distributed training - World size: {world_size}, Rank: {rank}, Local rank: {local_rank}"
        )

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        torch.cuda.set_device(local_rank)
        return True, world_size, rank, local_rank
    else:
        logger.info("Single GPU training")
        return False, 1, 0, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------------------------------------------------------
# MLflow (SageMaker plugin / ARN URI + autolog)
# -------------------------------------------------------------------------
def setup_mlflow_tracking(experiment_name: str = "esg-training") -> bool:
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


def setup_model(model_config):
    """Set up the model with optional quantization and LoRA."""
    logger.info(f"Setting up model: {model_config['model_path']}")

    device_map = None

    # Load base model with optional quantization
    if not model_config.get("quantization"):
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_path"],
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    elif model_config["quantization"] == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_path"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    else:
        if model_config["quantization"] == "int4":
            bnb_config = BitsAndBytesConfig(**model_config["int4_config"])
        elif model_config["quantization"] == "int8":
            bnb_config = BitsAndBytesConfig(**model_config["int8_config"])
        else:
            raise ValueError(f"Unknown quantization: {model_config['quantization']}")

        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_path"],
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

    # Load existing LoRA adapters if specified
    if model_config.get("lora_path"):
        logger.info(f"Loading LoRA adapters from: {model_config['lora_path']}")
        peft_config = PeftConfig.from_pretrained(model_config["lora_path"])
        base_with_adapters_model = PeftModel.from_pretrained(
            model, model_config["lora_path"]
        )
        model = base_with_adapters_model.merge_and_unload()

    # Apply LoRA configuration for training
    if model_config.get("lora"):
        logger.info("Applying LoRA configuration")
        peft_config = LoraConfig(**model_config["lora"])
        # Enable gradient checkpointing and prepare for k-bit training
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        # Get the PEFT model
        model = get_peft_model(model, peft_config)

    # Set model configuration
    if "window" in model_config:
        if hasattr(model.config, "sliding_window"):
            model.config.sliding_window = model_config["window"]
        else:
            logger.warning("Model does not support sliding_window attribute")

    # Print trainable parameters
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        print_trainable_params(model)

    return model


def print_trainable_params(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.info(f"Total parameters: {all_params}")
    logger.info(f"Trainable parameters ratio: {100 * trainable_params/all_params:.2f}%")


def load_and_tokenize_data(
    data_path, tokenizer, tokenizer_config, max_samples=None, sample_fraction=None
):
    """
    Load and tokenize the training data with proper label masking.
    Optionally sample a fraction of rows first (e.g. 0.01 for 1%).
    """
    logger.info(f"Loading data from: {data_path}")

    # Load data
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".json"):
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    logger.info(f"Loaded {len(df)} samples")

    # Optional fractional sampling first (e.g., 0.01 = 1%)
    if sample_fraction is not None:
        try:
            frac = float(sample_fraction)
        except ValueError:
            frac = None
        if frac is not None and 0 < frac < 1.0:
            n_before = len(df)
            n_keep = max(1, int(n_before * frac))
            logger.info(
                f"⚡ FAST TRAIN: sampling {n_keep} rows ({frac*100:.2f}% of {n_before})"
            )
            df = df.sample(n=n_keep, random_state=42).reset_index(drop=True)
            logger.info(f"After FAST TRAIN sampling: {len(df)} samples")

    # Apply max_samples cap (after fraction, if provided)
    if max_samples is not None and len(df) > max_samples:
        logger.info(f"Capping to max_samples={max_samples} (from {len(df)})")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"After capping: {len(df)} samples")

    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        """Tokenize with accurate label masking and smart fallback."""

        batch_size = len(examples["instruction"])
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        # Get the token ID for [/INST]
        inst_end_token = "[/INST]"
        inst_end_ids = tokenizer.encode(inst_end_token, add_special_tokens=False)

        # Statistics for logging
        fallback_count = 0

        for i in range(batch_size):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i] if examples["input"][i] else ""
            output_text = examples["output"][i]

            # Create full text
            if input_text:
                instruction_text = f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
            else:
                instruction_text = f"<s>[INST] {instruction} [/INST]"

            full_text = f"{instruction_text} {output_text}</s>"

            # Tokenize
            full_tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=tokenizer_config.get("max_length", 512),
                padding="max_length",
                add_special_tokens=False,
            )

            # Find [/INST] token position
            input_ids = full_tokens["input_ids"]
            instruction_end = 0

            # Search for the [/INST] token(s)
            for j in range(len(input_ids) - len(inst_end_ids) + 1):
                if input_ids[j : j + len(inst_end_ids)] == inst_end_ids:
                    instruction_end = j + len(inst_end_ids)
                    break

            if instruction_end == 0:
                # SMART FALLBACK: Estimate based on instruction/output ratio
                instruction_only_tokens = tokenizer(
                    instruction_text,
                    truncation=False,
                    add_special_tokens=False,
                )
                instruction_true_length = len(instruction_only_tokens["input_ids"])

                if instruction_true_length >= tokenizer_config.get("max_length", 512):
                    # Instruction is too long - it got truncated
                    instruction_end = int(len(input_ids) * 0.7)
                    fallback_count += 1
                    if i < 5:  # Log first few
                        logger.warning(
                            f"Sample {i}: Instruction truncated "
                            f"(true len: {instruction_true_length}, "
                            f"using 70% split at position {instruction_end})"
                        )
                else:
                    # Instruction fits but [/INST] wasn't found
                    instruction_end = min(instruction_true_length, len(input_ids) - 10)
                    fallback_count += 1
                    if i < 5:
                        logger.warning(
                            f"Sample {i}: [/INST] not found but instruction fits "
                            f"(len: {instruction_true_length}, using position {instruction_end})"
                        )

            # Create labels
            labels = input_ids.copy()

            # Mask everything up to instruction_end
            for j in range(min(instruction_end, len(labels))):
                labels[j] = -100

            # Mask padding
            for j in range(len(labels)):
                if input_ids[j] == tokenizer.pad_token_id:
                    labels[j] = -100

            input_ids_list.append(input_ids)
            attention_mask_list.append(full_tokens["attention_mask"])
            labels_list.append(labels)

        # Log fallback statistics
        if fallback_count > 0:
            logger.info(
                f"Used fallback for {fallback_count}/{batch_size} samples in this batch"
            )

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")

    # Log a sample to verify masking
    if len(tokenized_dataset) > 0:
        sample = tokenized_dataset[0]
        logger.info("Sample tokenization check:")
        logger.info(f"  Input IDs length: {len(sample['input_ids'])}")
        logger.info(f"  Labels length: {len(sample['labels'])}")
        masked_count = sum(1 for label in sample["labels"] if label == -100)
        logger.info(f"  Masked tokens (instruction): {masked_count}")
        logger.info(
            f"  Training tokens (output): {len(sample['labels']) - masked_count}"
        )

    return tokenized_dataset


def create_trainer(
    model, tokenizer, train_dataset, eval_dataset, training_config, output_dir
):
    """
    Create and configure the trainer.
    """
    logger.info("Creating trainer")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get(
            "per_device_train_batch_size", 4
        ),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_config.get(
            "gradient_accumulation_steps", 1
        ),
        warmup_steps=training_config.get("warmup_steps", 500),
        learning_rate=training_config.get("learning_rate", 5e-5),
        logging_steps=training_config.get("logging_steps", 10),
        logging_dir=f"{output_dir}/logs",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=training_config.get("eval_steps", 500),
        save_steps=training_config.get("save_steps", 1000),
        save_strategy=training_config.get("save_strategy", "steps"),
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        optim=training_config.get("optim", "paged_adamw_32bit"),
        bf16=training_config.get("bf16", torch.cuda.is_available()),
        fp16=training_config.get("fp16", False),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=training_config.get("report_to", "none"),
        run_name=f"esg-training-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    return trainer


def save_training_info_simple(args, model_config, training_config, output_dir):
    """
    Save training configuration and metadata (simple fix).
    """
    # Create a copy and convert torch.dtype to string
    safe_model_config = model_config.copy()
    safe_training_config = training_config.copy()

    # Handle the specific dtype issue in quantization config
    if "int4_config" in safe_model_config:
        if "bnb_4bit_compute_dtype" in safe_model_config["int4_config"]:
            safe_model_config["int4_config"]["bnb_4bit_compute_dtype"] = str(
                safe_model_config["int4_config"]["bnb_4bit_compute_dtype"]
            )

    if "int8_config" in safe_model_config:
        # Handle any dtypes in int8_config if they exist
        for key, value in safe_model_config["int8_config"].items():
            if hasattr(value, "dtype") or str(type(value)).find("torch") != -1:
                safe_model_config["int8_config"][key] = str(value)

    training_info = {
        "model_name": safe_model_config["model_path"],
        "training_config": safe_training_config,
        "model_config": safe_model_config,
        "timestamp": datetime.now().isoformat(),
        "task_type": "esg_generation",
    }

    # Save training info
    info_file = os.path.join(output_dir, "training_info.json")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2)
    logger.info(f"Training info saved to: {info_file}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ESG model")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)

    # NEW: fractional sampling flags
    parser.add_argument(
        "--train-sample-fraction",
        type=float,
        default=None,
        help="Fraction of training rows to sample (e.g., 0.01 for 1%).",
    )
    parser.add_argument(
        "--eval-sample-fraction",
        type=float,
        default=None,
        help="Fraction of evaluation rows to sample (e.g., 0.01 for 1%).",
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model name or path",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Model output directory",
    )

    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        help="Training data path",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"
        ),
        help="Validation data path",
    )

    # Training arguments
    parser.add_argument(
        "--num-train-epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=4,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--warmup-steps", type=int, default=200, help="Warmup steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps")
    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )

    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")

    # Quantization arguments
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["int4", "int8", "bf16", "none"],
        default="bf16",
        help="Quantization type",
    )

    # Other arguments
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="mistral_formal",
        help="Prompt template used in preprocessing",
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Env fallbacks for sampling so you can flip this on without changing args
    # FAST_TRAIN_FRACTION / TRAIN_SAMPLE_FRACTION for train
    # FAST_EVAL_FRACTION / EVAL_SAMPLE_FRACTION for eval
    # ----------------------------------------------------------------------
    if args.train_sample_fraction is None:
        env_val = os.environ.get("FAST_TRAIN_FRACTION") or os.environ.get(
            "TRAIN_SAMPLE_FRACTION"
        )
        if env_val is not None:
            try:
                args.train_sample_fraction = float(env_val)
            except ValueError:
                logger.warning(f"Invalid FAST_TRAIN_FRACTION '{env_val}', ignoring.")
                args.train_sample_fraction = None

    if args.eval_sample_fraction is None:
        env_val = os.environ.get("FAST_EVAL_FRACTION") or os.environ.get(
            "EVAL_SAMPLE_FRACTION"
        )
        if env_val is not None:
            try:
                args.eval_sample_fraction = float(env_val)
            except ValueError:
                logger.warning(f"Invalid FAST_EVAL_FRACTION '{env_val}', ignoring.")
                args.eval_sample_fraction = None

    if args.train_sample_fraction:
        logger.info(
            f"⚡ FAST TRAIN ENABLED: training on {args.train_sample_fraction*100:.2f}% of data"
        )
    if args.eval_sample_fraction:
        logger.info(
            f"⚡ FAST EVAL ENABLED: evaluating on {args.eval_sample_fraction*100:.2f}% of data"
        )

    try:
        # =====================================================================
        # DISTRIBUTED TRAINING SETUP
        # =====================================================================
        is_distributed, world_size, rank, local_rank = setup_distributed()

        # Suppress logs on non-zero ranks to avoid clutter
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)

        logger.info(f"Starting training on rank {rank}/{world_size}")

        # =====================================================================
        # MODEL LOADING NOTE
        # =====================================================================
        # Models will be loaded using tokens from environment if available
        if is_distributed:
            dist.barrier()

        # =====================================================================
        # MLFLOW SETUP (RANK 0 ONLY)
        # =====================================================================
        mlflow_enabled = False
        if rank == 0:
            mlflow_enabled = setup_mlflow_tracking("esg-training")

        # =====================================================================
        # MODEL CONFIGURATION
        # =====================================================================
        model_config = {
            "model_path": args.model_name,
            "quantization": args.quantization if args.quantization != "none" else None,
            "window": 256,
            "lora_path": None,
        }

        # Configure quantization
        if args.quantization == "int4":
            model_config["int4_config"] = {
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": torch.float16,  # int4 needs fp16
            }
        elif args.quantization == "int8":
            model_config["int8_config"] = {
                "load_in_4bit": False,
                "load_in_8bit": True,
            }

        # Configure LoRA - Optimized (no MLP layers for speed/memory)
        model_config["lora"] = {
            "task_type": "CAUSAL_LM",
            "inference_mode": False,
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                # Removed MLP layers for 30% speed boost and 20% memory savings:
                # "gate_proj", "up_proj", "down_proj", "lm_head"
            ],
        }

        # =====================================================================
        # TRAINING CONFIGURATION
        # =====================================================================
        training_config = {
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "logging_steps": args.logging_steps,
            "lr_scheduler_type": "cosine",
            # "optim": "paged_adamw_32bit",
            "optim": "paged_adamw_8bit",
            "bf16": args.quantization != "int4",  # bf16 for everything except int4
            "fp16": args.quantization == "int4",  # fp16 only for int4
            "save_strategy": "steps",
            "eval_steps": 250,
            "save_steps": 500,
            "report_to": "none",
            "gradient_checkpointing": False,  # Save memory
            "ddp_find_unused_parameters": True,  # For distributed training
            "max_grad_norm": 1.0,  # Gradient clipping
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            "torch_compile": False,  # Disable compilation for stability
        }

        # =====================================================================
        # TOKENIZER CONFIGURATION
        # =====================================================================
        tokenizer_config = {
            "pretrained_model_name_or_path": args.model_name,
            "use_fast": True,
            "padding_side": "left",
            "truncation_side": "right",
            "add_bos_token": True,
            "add_eos_token": False,
            "model_max_length": args.max_length,
            "trust_remote_code": True,
        }

        # =====================================================================
        # LOAD MODEL AND TOKENIZER (ALL RANKS)
        # =====================================================================
        if rank == 0:
            logger.info("Rank 0: Pre-downloading model...")
            try:
                from transformers import AutoConfig

                _ = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Pre-download warning: {str(e)}")

        if is_distributed:
            dist.barrier()

        # Then load model normally
        model = setup_model(model_config)

        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
        tokenizer.pad_token = tokenizer.eos_token

        if rank == 0:
            logger.info("Model and tokenizer loaded successfully")

        # =====================================================================
        # LOAD AND PREPARE DATA (ALL RANKS)
        # =====================================================================
        train_file = os.path.join(args.train_data, "train.csv")

        if rank == 0:
            logger.info(f"Loading training data from: {train_file}")

        train_dataset = load_and_tokenize_data(
            train_file,
            tokenizer,
            {"max_length": args.max_length},
            max_samples=args.max_train_samples,
            sample_fraction=args.train_sample_fraction,  # NEW
        )

        eval_dataset = None
        if os.path.exists(args.val_data):
            val_file = os.path.join(args.val_data, "validation.csv")
            if os.path.exists(val_file):
                if rank == 0:
                    logger.info(f"Loading validation data from: {val_file}")
                eval_dataset = load_and_tokenize_data(
                    val_file,
                    tokenizer,
                    {"max_length": args.max_length},
                    max_samples=args.max_eval_samples,
                    sample_fraction=args.eval_sample_fraction,  # NEW
                )

        # =====================================================================
        # CREATE TRAINER (ALL RANKS)
        # =====================================================================
        trainer = create_trainer(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            training_config,
            args.model_dir,
        )

        # Disable cache for training (required for gradient checkpointing)
        model.config.use_cache = False

        # =====================================================================
        # TRAINING WITH OPTIONAL MLFLOW (RANK 0 LOGS)
        # =====================================================================
        if mlflow_enabled and rank == 0:
            with mlflow.start_run() as run:
                logger.info(f"Started MLFlow run: {run.info.run_id}")

                # Log parameters (rank 0 only)
                mlflow.log_params(
                    {
                        "model_name": args.model_name,
                        "num_train_epochs": args.num_train_epochs,
                        "per_device_train_batch_size": args.per_device_train_batch_size,
                        "per_device_eval_batch_size": args.per_device_eval_batch_size,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "learning_rate": args.learning_rate,
                        "warmup_steps": args.warmup_steps,
                        "max_length": args.max_length,
                        "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "lora_dropout": args.lora_dropout,
                        "quantization": args.quantization,
                        "prompt_template": args.prompt_template,
                        "train_dataset_size": len(train_dataset),
                        "eval_dataset_size": len(eval_dataset) if eval_dataset else 0,
                        "world_size": world_size,
                        "distributed": is_distributed,
                        "train_sample_fraction": args.train_sample_fraction,
                        "eval_sample_fraction": args.eval_sample_fraction,
                    }
                )

                # Log configs
                mlflow.log_dict(model_config, "model_config.json")
                mlflow.log_dict(training_config, "training_config.json")

                # Train the model (ALL RANKS participate)
                logger.info("Starting training with MLFlow tracking")
                train_result = trainer.train()

                # Log training metrics (rank 0 only)
                if hasattr(train_result, "metrics"):
                    for key, value in train_result.metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)

                # Log training history
                if hasattr(trainer.state, "log_history"):
                    for log_entry in trainer.state.log_history:
                        if "train_loss" in log_entry:
                            mlflow.log_metric(
                                "train_loss",
                                log_entry["train_loss"],
                                step=log_entry.get("step", 0),
                            )
                        if "eval_loss" in log_entry:
                            mlflow.log_metric(
                                "eval_loss",
                                log_entry["eval_loss"],
                                step=log_entry.get("step", 0),
                            )

                # Save model (rank 0 only)
                if rank == 0:
                    logger.info("Saving model...")
                    trainer.save_model(args.model_dir)
                    tokenizer.save_pretrained(args.model_dir)

                # Log artifacts (rank 0 only)
                try:
                    mlflow.log_artifacts(args.model_dir, "model")
                    logger.info("Model artifacts logged to MLFlow")
                except Exception as e:
                    logger.warning(f"Failed to log model artifacts to MLFlow: {str(e)}")

                logger.info("Training completed successfully with MLFlow tracking")
        else:
            # Train without MLFlow (ALL RANKS participate)
            if rank == 0:
                logger.info("Starting training (MLFlow tracking disabled)")

            trainer.train()

            # Save model (rank 0 only)
            if rank == 0:
                logger.info("Saving model...")
                trainer.save_model(args.model_dir)
                tokenizer.save_pretrained(args.model_dir)
                logger.info("Training completed successfully")

        # =====================================================================
        # CLEANUP
        # =====================================================================
        if is_distributed:
            dist.barrier()  # Wait for all ranks to finish
            cleanup_distributed()
            if rank == 0:
                logger.info("Distributed training cleanup completed")

    except Exception as e:
        logger.warning(f"Training failed: {str(e)}")
        # Cleanup on error
        if dist.is_initialized():
            cleanup_distributed()
        raise


if __name__ == "__main__":
    main()
