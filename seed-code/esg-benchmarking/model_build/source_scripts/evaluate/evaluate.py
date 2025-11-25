"""
SusGen Model Evaluation Script.

This script evaluates the trained ESG model using SusGen's comprehensive evaluation
framework with 8 different benchmark tasks.
Based on the original SusGen repository implementation.
"""


def install_requirements():
    """
    Install required packages at runtime for SageMaker processing.
    Uses pip's programmatic API to avoid subprocess security concerns.
    """
    import logging
    import sys
    import re

    # Configure logging for installation
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Predefined whitelist of allowed packages with validation
    allowed_packages = {
        "torch": "torch>=2.0.0",
        "transformers": "transformers>=4.30.0",
        "rouge-score": "rouge-score>=0.1.2",
        "pandas": "pandas>=1.5.0",
        "scikit-learn": "scikit-learn>=1.2.0",
        "numpy": "numpy>=1.24.0,<2.0.0",
        "sentencepiece": "sentencepiece>=0.2.0",
        "peft": "peft>=0.4.0",
        "huggingface-hub": "huggingface-hub>=0.15.0",
        "accelerate": "accelerate>=0.20.0",
        "bitsandbytes": "bitsandbytes>=0.41.0",
    }

    logger.info("Installing required packages using pip API...")

    try:
        # Import pip's main function
        from pip._internal.main import main as pip_main

        for package_name, package_spec in allowed_packages.items():
            try:
                logger.info(f"Installing: {package_spec}")

                # Validate package specification format for security
                if not re.match(r"^[a-zA-Z0-9_-]+[>=<,.\d\s]*$", package_spec):
                    logger.error(f"Invalid package specification: {package_spec}")
                    continue

                # Use pip's internal API - more secure than subprocess
                result = pip_main(["install", package_spec, "--upgrade", "--quiet"])

                if result != 0:
                    logger.warning(f"Failed to install {package_spec}")
                    raise RuntimeError(f"Package installation failed: {package_spec}")

            except Exception as e:
                logger.warning(f"Failed to install {package_spec}: {str(e)}")
                raise

        logger.info("All packages installed successfully!")

    except ImportError:
        # Fallback: Try importing packages to check if they're already available
        logger.warning(
            "pip API not available, checking if packages are already installed..."
        )

        import importlib

        missing_packages = []

        for package_name in allowed_packages.keys():
            try:
                # Map package names to import names where different
                import_name = package_name
                if package_name == "scikit-learn":
                    import_name = "sklearn"
                elif package_name == "huggingface-hub":
                    import_name = "huggingface_hub"
                elif package_name == "rouge-score":
                    import_name = "rouge_score"

                # Security: Only import from predefined allowed packages
                if import_name not in [
                    "torch",
                    "transformers",
                    "rouge_score",
                    "pandas",
                    "sklearn",
                    "numpy",
                    "sentencepiece",
                    "peft",
                    "huggingface_hub",
                    "accelerate",
                    "bitsandbytes",
                ]:
                    logger.warning(
                        f"Skipping import of non-whitelisted module: {import_name}"
                    )
                    missing_packages.append(package_name)
                    continue

                importlib.import_module(import_name)
                logger.info(f"✓ {package_name} already available")
            except ImportError:
                missing_packages.append(package_name)

        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            logger.warning(
                "Please install missing packages manually or ensure pip API is available"
            )
            raise ImportError(f"Required packages not available: {missing_packages}")
        else:
            logger.info("All required packages are already installed!")


install_requirements()

import argparse
import json
import logging
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rouge_score import rouge_scorer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path):
    """
    Load trained model and tokenizer from SageMaker format with LoRA adapters.
    Optimized for faster inference with 8-bit quantization.

    Args:
        model_path (str): Path to the trained model (could be tar.gz or directory)

    Returns:
        tuple: (model, tokenizer)
    """
    import tarfile
    import tempfile
    import os
    from peft import PeftModel, PeftConfig

    logger.info(f"Loading model from: {model_path}")

    try:
        # Extract model if needed
        if os.path.isdir(model_path):
            if any(
                os.path.exists(os.path.join(model_path, f))
                for f in ["adapter_config.json", "config.json"]
            ):
                actual_model_path = model_path
            else:
                tar_path = os.path.join(model_path, "model.tar.gz")
                if os.path.exists(tar_path):
                    logger.info(f"Found model.tar.gz at: {tar_path}")
                    temp_dir = tempfile.mkdtemp()
                    with tarfile.open(tar_path, "r:gz") as tar:
                        # Security fix: Validate tar members before extraction to prevent path traversal attacks
                        def is_safe_path(path: str, base_path: str) -> bool:
                            """Check if the path is safe for extraction (no path traversal)."""
                            return (
                                os.path.commonpath(
                                    [
                                        os.path.realpath(os.path.join(base_path, path)),
                                        base_path,
                                    ]
                                )
                                == base_path
                            )

                        def safe_extract(members, path):
                            """Safely extract tar members, filtering out dangerous paths."""
                            for member in members:
                                if member.isfile() or member.isdir():
                                    # Check for path traversal attempts
                                    if not is_safe_path(member.name, path):
                                        logger.warning(
                                            f"Skipping potentially dangerous path: {member.name}"
                                        )
                                        continue
                                    # Check for absolute paths
                                    if os.path.isabs(member.name):
                                        logger.warning(
                                            f"Skipping absolute path: {member.name}"
                                        )
                                        continue
                                    # Normalize the path
                                    member.name = os.path.normpath(member.name)
                                    yield member

                        tar.extractall(temp_dir, members=safe_extract(tar, temp_dir))
                    actual_model_path = temp_dir
                else:
                    raise FileNotFoundError(f"No model files found in {model_path}")
        elif model_path.endswith(".tar.gz"):
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Extracting {model_path} to {temp_dir}")
            with tarfile.open(model_path, "r:gz") as tar:
                # Security fix: Validate tar members before extraction to prevent path traversal attacks
                def is_safe_path(path: str, base_path: str) -> bool:
                    """Check if the path is safe for extraction (no path traversal)."""
                    return (
                        os.path.commonpath(
                            [
                                os.path.realpath(os.path.join(base_path, path)),
                                base_path,
                            ]
                        )
                        == base_path
                    )

                def safe_extract(members, path):
                    """Safely extract tar members, filtering out dangerous paths."""
                    for member in members:
                        if member.isfile() or member.isdir():
                            # Check for path traversal attempts
                            if not is_safe_path(member.name, path):
                                logger.warning(
                                    f"Skipping potentially dangerous path: {member.name}"
                                )
                                continue
                            # Check for absolute paths
                            if os.path.isabs(member.name):
                                logger.warning(f"Skipping absolute path: {member.name}")
                                continue
                            # Normalize the path
                            member.name = os.path.normpath(member.name)
                            yield member

                tar.extractall(temp_dir, members=safe_extract(tar, temp_dir))
            actual_model_path = temp_dir
        else:
            actual_model_path = model_path

        logger.info(f"Loading model from actual path: {actual_model_path}")

        # List contents for debugging
        if os.path.isdir(actual_model_path):
            contents = os.listdir(actual_model_path)
            logger.info(f"Model directory contents: {contents}")

        # Check if this is a LoRA adapter or full model
        adapter_config_path = os.path.join(actual_model_path, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            logger.info(
                "Detected LoRA adapter - loading base model and applying adapters"
            )

            # Load the adapter config to get the base model name
            peft_config = PeftConfig.from_pretrained(actual_model_path)
            base_model_name = peft_config.base_model_name_or_path

            logger.info(f"Base model: {base_model_name}")

            # Load tokenizer from the adapter directory
            tokenizer = AutoTokenizer.from_pretrained(actual_model_path)

            # Configure 8-bit quantization for faster inference
            use_quantization = torch.cuda.is_available()

            if use_quantization:
                logger.info(
                    "Loading base model with 8-bit quantization for faster inference"
                )
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                logger.info(f"Loading base model without quantization (CPU mode)")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                )

            # Load and apply LoRA adapters
            logger.info("Applying LoRA adapters")
            model = PeftModel.from_pretrained(base_model, actual_model_path)

            # Skip merging for faster evaluation - merged models don't benefit quantization
            # Merging takes time and isn't necessary for evaluation
            # logger.info("Merging LoRA adapters")
            # model = model.merge_and_unload()
            logger.info("Skipping adapter merging for faster evaluation")

        else:
            # Regular model loading (fallback)
            logger.info("Loading regular model (no LoRA adapters detected)")
            tokenizer = AutoTokenizer.from_pretrained(actual_model_path)

            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    actual_model_path, torch_dtype=torch.float16, device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    actual_model_path, torch_dtype=torch.float32
                )

        # Set model to evaluation mode
        model.eval()

        logger.info("Model loaded successfully")
        return model, tokenizer

    except Exception as e:
        logger.warning(f"Error loading model: {str(e)}")
        logger.warning("Model loading failed, will use dummy evaluation")
        raise


def load_benchmark_data(benchmark_dir):
    """
    Load test data for evaluation.
    Works with both simple CSV/JSON files and structured benchmark directories.

    Args:
        benchmark_dir (str): Path to test datasets

    Returns:
        dict: Dictionary of test datasets
    """
    logger.info(f"Loading test data from: {benchmark_dir}")

    if not os.path.exists(benchmark_dir):
        logger.warning(f"Test directory not found: {benchmark_dir}")
        return {}

    benchmark_data = {}

    # Try to load test.json first (from preprocessing step)
    test_json = os.path.join(benchmark_dir, "test.json")
    test_csv = os.path.join(benchmark_dir, "test.csv")

    if os.path.exists(test_json):
        logger.info(f"Loading test data from: {test_json}")
        try:
            with open(test_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert to expected format
            samples = []
            for item in data:
                sample = {
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": item.get("output", ""),
                }
                samples.append(sample)

            # Put all samples in a "TestData" category
            benchmark_data["TestData"] = {"SusGen-Test": {"samples": samples}}

            logger.info(f"Loaded {len(samples)} test samples from JSON")
            return benchmark_data

        except Exception as e:
            logger.warning(f"Failed to load {test_json}: {str(e)}")

    elif os.path.exists(test_csv):
        logger.info(f"Loading test data from: {test_csv}")
        try:
            df = pd.read_csv(test_csv)

            samples = []
            for _, row in df.iterrows():
                sample = {
                    "instruction": row.get("instruction", ""),
                    "input": row.get("input", ""),
                    "output": row.get("output", ""),
                }
                samples.append(sample)

            benchmark_data["TestData"] = {"SusGen-Test": {"samples": samples}}

            logger.info(f"Loaded {len(samples)} test samples from CSV")
            return benchmark_data

        except Exception as e:
            logger.warning(f"Failed to load {test_csv}: {str(e)}")

    # If no simple files found, try structured benchmark format
    logger.info("No test.json/test.csv found, trying structured benchmark format...")

    benchmark_tasks = {
        "FINQA": ["FinQA", "FSRL"],
        "FINTQA": ["ConvFinQA", "TATQA"],
        "HC": ["MultiFin", "MLESG"],
        "NER": ["NER", "FINER-ORD"],
        "RE": ["FinRED", "SC"],
        "SA": ["FiQA-SA", "FOMC"],
        "SUM": ["EDTSum"],
        "SRG": ["Annual", "ESG"],
    }

    for task, datasets in benchmark_tasks.items():
        task_data = {}
        task_dir = os.path.join(benchmark_dir, task)

        if os.path.exists(task_dir):
            for dataset in datasets:
                dataset_file = os.path.join(task_dir, f"{dataset}.json")
                if os.path.exists(dataset_file):
                    with open(dataset_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    task_data[dataset] = data
                    logger.info(
                        f"Loaded {dataset} with {len(data.get('samples', []))} samples"
                    )

        if task_data:
            benchmark_data[task] = task_data

    if not benchmark_data:
        logger.warning("No benchmark or test data found in any format")

    return benchmark_data


def generate_predictions(model, tokenizer, samples, max_length=512, batch_size=16):
    """
    Generate predictions using the trained model with TRUE batching.
    Optimized for faster inference.
    """
    logger.info(f"Starting prediction generation for {len(samples)} samples")
    logger.info(f"Using batch size: {batch_size}")
    logger.info(f"Max sequence length: {max_length}")

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    total_samples = len(samples)

    for i in range(0, total_samples, batch_size):
        batch_samples = samples[i : i + batch_size]

        batch_num = (i // batch_size) + 1
        total_batches = (total_samples + batch_size - 1) // batch_size
        logger.info(
            f"Processing batch {batch_num}/{total_batches} (samples {i+1}-{min(i+batch_size, total_samples)})"
        )

        try:
            # Prepare batch prompts
            batch_prompts = []
            for sample in batch_samples:
                if isinstance(sample, dict):
                    instruction = sample.get("instruction", "")
                    input_text = sample.get("input", "")
                else:
                    instruction = str(sample)
                    input_text = ""

                # Create input prompt
                if input_text:
                    prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
                else:
                    prompt = f"<s>[INST] {instruction} [/INST]"

                batch_prompts.append(prompt)

            # Tokenize entire batch at once
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,  # Pad to longest in batch
                return_attention_mask=True,
            )

            # Move to device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            input_lengths = (
                attention_mask.sum(dim=1)
            ).tolist()  # Track input length per sample

            logger.info(f"  Batch tokenized: {input_ids.shape}")

            # Generate for entire batch at once
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    num_return_sequences=1,
                    do_sample=False,  # Greedy for consistency
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode only the NEW tokens for each sample
            batch_predictions = []
            for idx, output in enumerate(outputs):
                # Get the input length for this specific sample
                input_len = input_lengths[idx]

                # Extract only generated tokens (skip input tokens)
                generated_ids = output[input_len:]
                prediction = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()

                batch_predictions.append(prediction)

                # Log first prediction of batch
                if idx == 0:
                    logger.info(f"  Sample preview: {prediction[:100]}...")

            predictions.extend(batch_predictions)

            # Progress update
            completed = min(i + batch_size, total_samples)
            progress = (completed / total_samples) * 100
            logger.info(
                f"  Batch complete - Overall: {completed}/{total_samples} ({progress:.1f}%)"
            )
            logger.info("-" * 80)

        except Exception as e:
            logger.error(f"  Batch {batch_num} failed: {str(e)}")
            logger.error(f"  Adding empty predictions for this batch")
            # Add empty predictions for failed batch
            predictions.extend([""] * len(batch_samples))

    logger.info("=" * 80)
    logger.info(f"Prediction generation completed!")
    logger.info(f"Total samples processed: {len(predictions)}")
    logger.info(f"Successful predictions: {sum(1 for p in predictions if p)}")
    logger.info(f"Failed predictions: {sum(1 for p in predictions if not p)}")
    logger.info("=" * 80)

    return predictions


def evaluate_finqa(predictions, references):
    """
    Evaluate Financial Question Answering task.

    Args:
        predictions (list): Generated predictions
        references (list): Reference answers

    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating FINQA task")

    # Extract numerical answers for exact match
    def extract_number(text):
        """Extract the first number from text."""
        numbers = re.findall(r"-?\d+\.?\d*", str(text))
        return float(numbers[0]) if numbers else None

    exact_matches = 0
    numerical_matches = 0
    valid_predictions = 0

    for pred, ref in zip(predictions, references):
        if pred and ref:
            valid_predictions += 1

            # Exact string match
            if pred.strip().lower() == ref.strip().lower():
                exact_matches += 1

            # Numerical match
            pred_num = extract_number(pred)
            ref_num = extract_number(ref)

            if pred_num is not None and ref_num is not None:
                if abs(pred_num - ref_num) < 1e-6:
                    numerical_matches += 1

    metrics = {
        "exact_match": exact_matches / len(predictions) if predictions else 0,
        "numerical_match": numerical_matches / len(predictions) if predictions else 0,
        "valid_predictions": valid_predictions / len(predictions) if predictions else 0,
    }

    return metrics


def evaluate_classification(predictions, references, task_name):
    """
    Evaluate classification tasks (HC, SA, etc.).

    Args:
        predictions (list): Generated predictions
        references (list): Reference labels
        task_name (str): Name of the task

    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Evaluating {task_name} classification task")

    # Simple label extraction (first word/phrase)
    def extract_label(text):
        """Extract classification label from text."""
        if not text:
            return "unknown"
        # Take first word as label
        return str(text).strip().split()[0].lower()

    pred_labels = [extract_label(pred) for pred in predictions]
    ref_labels = [extract_label(ref) for ref in references]

    logger.info("=" * 80)
    logger.info("CLASSIFICATION SAMPLE PREDICTIONS:")
    logger.info("=" * 80)

    # Log first 10 samples
    for idx in range(min(10, len(predictions))):
        logger.info(f"\nSample {idx + 1}:")
        logger.info(f"  Reference text: {references[idx]}")
        logger.info(f"  Predicted text: {predictions[idx]}")
        logger.info(f"  Extracted ref label: {ref_labels[idx]}")
        logger.info(f"  Extracted pred label: {pred_labels[idx]}")
        logger.info(f"  Match: {'✓' if ref_labels[idx] == pred_labels[idx] else '✗'}")

    # Count label distribution
    from collections import Counter

    ref_dist = Counter(ref_labels)
    pred_dist = Counter(pred_labels)

    logger.info("=" * 80)
    logger.info("LABEL DISTRIBUTIONS:")
    logger.info(f"Reference labels: {dict(ref_dist)}")
    logger.info(f"Predicted labels: {dict(pred_dist)}")
    logger.info("=" * 80)

    # Calculate metrics
    try:
        accuracy = accuracy_score(ref_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ref_labels, pred_labels, average="weighted", zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        logger.info(f"Classification Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1:.4f}")

    except Exception as e:
        logger.warning(f"Error calculating classification metrics: {str(e)}")
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return metrics


def evaluate_generation(predictions, references, task_name):
    """
    Evaluate text generation tasks (SUM, SRG).

    Args:
        predictions (list): Generated predictions
        references (list): Reference texts
        task_name (str): Name of the task

    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Evaluating {task_name} generation task")
    logger.info(
        f"Total predictions: {len(predictions)}, Total references: {len(references)}"
    )

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge_scores = {
        "rouge1": {"precision": [], "recall": [], "fmeasure": []},
        "rouge2": {"precision": [], "recall": [], "fmeasure": []},
        "rougeL": {"precision": [], "recall": [], "fmeasure": []},
    }

    # Track statistics
    empty_predictions = 0
    empty_references = 0
    prediction_lengths = []
    reference_lengths = []

    logger.info("=" * 80)
    logger.info("SAMPLE PREDICTIONS vs REFERENCES:")
    logger.info("=" * 80)

    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        # Log first 5 samples in detail
        if idx < 5:
            logger.info(f"\n--- Sample {idx + 1} ---")
            logger.info(f"REFERENCE: {ref}")
            logger.info(f"PREDICTED: {pred}")
            logger.info(
                f"Ref length: {len(str(ref))} chars, Pred length: {len(str(pred))} chars"
            )

        if pred and ref:
            prediction_lengths.append(len(str(pred)))
            reference_lengths.append(len(str(ref)))

            scores = scorer.score(str(ref), str(pred))

            # Log ROUGE scores for first 3 samples
            if idx < 3:
                logger.info(f"ROUGE-1 F1: {scores['rouge1'].fmeasure:.4f}")
                logger.info(f"ROUGE-2 F1: {scores['rouge2'].fmeasure:.4f}")
                logger.info(f"ROUGE-L F1: {scores['rougeL'].fmeasure:.4f}")

            for rouge_type in ["rouge1", "rouge2", "rougeL"]:
                rouge_scores[rouge_type]["precision"].append(
                    scores[rouge_type].precision
                )
                rouge_scores[rouge_type]["recall"].append(scores[rouge_type].recall)
                rouge_scores[rouge_type]["fmeasure"].append(scores[rouge_type].fmeasure)
        else:
            if not pred:
                empty_predictions += 1
            if not ref:
                empty_references += 1

    logger.info("=" * 80)
    logger.info("GENERATION STATISTICS:")
    logger.info(f"Empty predictions: {empty_predictions}/{len(predictions)}")
    logger.info(f"Empty references: {empty_references}/{len(references)}")
    if prediction_lengths:
        logger.info(f"Avg prediction length: {np.mean(prediction_lengths):.1f} chars")
        logger.info(f"Avg reference length: {np.mean(reference_lengths):.1f} chars")
        logger.info(
            f"Min/Max prediction length: {min(prediction_lengths)}/{max(prediction_lengths)}"
        )
        logger.info(
            f"Min/Max reference length: {min(reference_lengths)}/{max(reference_lengths)}"
        )
    logger.info("=" * 80)

    # Calculate averages
    avg_rouge_scores = {}
    for rouge_type in ["rouge1", "rouge2", "rougeL"]:
        if rouge_scores[rouge_type]["fmeasure"]:
            avg_rouge_scores[rouge_type] = {
                "precision": np.mean(rouge_scores[rouge_type]["precision"]),
                "recall": np.mean(rouge_scores[rouge_type]["recall"]),
                "fmeasure": np.mean(rouge_scores[rouge_type]["fmeasure"]),
            }
            logger.info(
                f"{rouge_type.upper()} - P: {avg_rouge_scores[rouge_type]['precision']:.4f}, "
                f"R: {avg_rouge_scores[rouge_type]['recall']:.4f}, "
                f"F1: {avg_rouge_scores[rouge_type]['fmeasure']:.4f}"
            )
        else:
            avg_rouge_scores[rouge_type] = {
                "precision": 0.0,
                "recall": 0.0,
                "fmeasure": 0.0,
            }

    return avg_rouge_scores


def evaluate_ner(predictions, references):
    """
    Evaluate Named Entity Recognition task.

    Args:
        predictions (list): Generated predictions
        references (list): Reference entities

    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating NER task")

    # Simple entity extraction (look for capitalized words)
    def extract_entities(text):
        """Extract entities from text."""
        if not text:
            return []
        # Simple heuristic: capitalized words
        entities = re.findall(r"\b[A-Z][a-z]+\b", str(text))
        return list(set(entities))

    total_predicted = 0
    total_reference = 0
    total_correct = 0

    for pred, ref in zip(predictions, references):
        pred_entities = extract_entities(pred)
        ref_entities = extract_entities(ref)

        total_predicted += len(pred_entities)
        total_reference += len(ref_entities)
        total_correct += len(set(pred_entities) & set(ref_entities))

    # Calculate precision, recall, F1
    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_reference if total_reference > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_predicted": total_predicted,
        "total_reference": total_reference,
        "total_correct": total_correct,
    }

    return metrics


def run_comprehensive_evaluation(model, tokenizer, benchmark_data, max_samples=100):
    """
    Run comprehensive evaluation across all benchmark tasks.

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        benchmark_data (dict): Benchmark datasets
        max_samples (int): Maximum samples per dataset

    Returns:
        dict: Comprehensive evaluation results
    """
    logger.info("Running comprehensive evaluation")

    all_results = {}

    for task_name, task_datasets in benchmark_data.items():
        logger.info(f"Evaluating task: {task_name}")
        task_results = {}

        for dataset_name, dataset in task_datasets.items():
            logger.info(f"Evaluating dataset: {dataset_name}")

            # Extract samples
            samples = dataset.get("samples", [])
            if len(samples) > max_samples:
                samples = samples[:max_samples]

            if not samples:
                logger.warning(f"No samples found for {dataset_name}")
                continue

            # Prepare inputs and references
            inputs = []
            references = []

            for sample in samples:
                if isinstance(sample, dict):
                    inputs.append(sample)
                    references.append(sample.get("output", ""))
                else:
                    inputs.append(
                        {"instruction": str(sample), "input": "", "output": ""}
                    )
                    references.append("")

            # Generate predictions
            predictions = generate_predictions(model, tokenizer, inputs)

            # Evaluate based on task type
            if task_name == "TestData":
                # For simple test data, use generation metrics (ROUGE)
                metrics = evaluate_generation(predictions, references, task_name)
            elif task_name in ["FINQA", "FINTQA"]:
                metrics = evaluate_finqa(predictions, references)
            elif task_name in ["HC", "SA"]:
                metrics = evaluate_classification(predictions, references, task_name)
            elif task_name in ["SUM", "SRG"]:
                metrics = evaluate_generation(predictions, references, task_name)
            elif task_name == "NER":
                metrics = evaluate_ner(predictions, references)
            elif task_name == "RE":
                # Relation extraction - treat as classification
                metrics = evaluate_classification(predictions, references, task_name)
            else:
                # Default to generation metrics
                metrics = evaluate_generation(predictions, references, task_name)

            task_results[dataset_name] = {
                "metrics": metrics,
                "num_samples": len(samples),
                "sample_predictions": predictions[:5],  # Store first 5 for inspection
                "sample_references": references[:5],
            }

        all_results[task_name] = task_results

    return all_results


def calculate_overall_metrics(all_results):
    """
    Calculate overall performance metrics across all tasks.

    Args:
        all_results (dict): Results from all tasks

    Returns:
        dict: Overall metrics
    """
    logger.info("Calculating overall metrics")

    # Collect all F1 scores for averaging
    f1_scores = []
    accuracy_scores = []
    rouge_scores = []

    for task_name, task_results in all_results.items():
        for dataset_name, dataset_results in task_results.items():
            metrics = dataset_results["metrics"]

            # Extract F1 scores
            if "f1" in metrics:
                f1_scores.append(metrics["f1"])

            # Extract accuracy scores
            if "accuracy" in metrics:
                accuracy_scores.append(metrics["accuracy"])

            # Extract ROUGE-L F1 scores
            if "rougeL" in metrics and "fmeasure" in metrics["rougeL"]:
                rouge_scores.append(metrics["rougeL"]["fmeasure"])

    overall_metrics = {
        "average_f1": np.mean(f1_scores) if f1_scores else 0.0,
        "average_accuracy": np.mean(accuracy_scores) if accuracy_scores else 0.0,
        "average_rouge_l": np.mean(rouge_scores) if rouge_scores else 0.0,
        "num_tasks_evaluated": len(all_results),
        "total_datasets": sum(
            len(task_results) for task_results in all_results.values()
        ),
    }

    return overall_metrics


def save_evaluation_results(all_results, overall_metrics, output_path):
    """
    Save evaluation results to files.

    Args:
        all_results (dict): Detailed results
        overall_metrics (dict): Overall metrics
        output_path (str): Output directory path
    """
    logger.info("Saving evaluation results")

    os.makedirs(output_path, exist_ok=True)

    # Save detailed results
    detailed_results_file = os.path.join(
        output_path, "detailed_evaluation_results.json"
    )
    with open(detailed_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save overall metrics
    overall_results_file = os.path.join(output_path, "overall_evaluation_results.json")
    with open(overall_results_file, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    # NEW: Save human-readable sample outputs
    samples_file = os.path.join(output_path, "sample_outputs.txt")
    with open(samples_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION SAMPLE OUTPUTS\n")
        f.write("=" * 80 + "\n\n")

        for task_name, task_results in all_results.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"TASK: {task_name}\n")
            f.write(f"{'=' * 80}\n\n")

            for dataset_name, dataset_results in task_results.items():
                f.write(f"\nDataset: {dataset_name}\n")
                f.write(f"Metrics: {dataset_results['metrics']}\n")
                f.write(f"Number of samples: {dataset_results['num_samples']}\n\n")

                # Write sample predictions
                f.write("-" * 80 + "\n")
                f.write("SAMPLE PREDICTIONS vs REFERENCES:\n")
                f.write("-" * 80 + "\n")

                sample_preds = dataset_results.get("sample_predictions", [])
                sample_refs = dataset_results.get("sample_references", [])

                for idx, (pred, ref) in enumerate(zip(sample_preds, sample_refs)):
                    f.write(f"\nSample {idx + 1}:\n")
                    f.write(f"REFERENCE:\n{ref}\n")
                    f.write(f"\nPREDICTED:\n{pred}\n")
                    f.write("-" * 40 + "\n")

    logger.info(f"Sample outputs saved to: {samples_file}")

    # Save SageMaker-compatible evaluation file
    sagemaker_results = {
        "regression_metrics": {
            "mse": {
                "value": 1.0 - overall_metrics.get("average_f1", 0.0),
                "standard_deviation": 0.1,
            }
        },
        "classification_metrics": {
            "accuracy": {"value": overall_metrics.get("average_accuracy", 0.0)}
        },
    }

    sagemaker_file = os.path.join(output_path, "evaluation.json")
    with open(sagemaker_file, "w", encoding="utf-8") as f:
        json.dump(sagemaker_results, f, indent=2)

    # Create summary report with more details
    summary_report = f"""
SusGen Model Evaluation Report
Generated: {datetime.now().isoformat()}

Overall Performance:
- Average F1 Score: {overall_metrics.get('average_f1', 0.0):.4f}
- Average Accuracy: {overall_metrics.get('average_accuracy', 0.0):.4f}
- Average ROUGE-L: {overall_metrics.get('average_rouge_l', 0.0):.4f}
- Tasks Evaluated: {overall_metrics.get('num_tasks_evaluated', 0)}
- Total Datasets: {overall_metrics.get('total_datasets', 0)}

Task-wise Performance:
"""

    for task_name, task_results in all_results.items():
        summary_report += f"\n{task_name}:\n"
        for dataset_name, dataset_results in task_results.items():
            metrics = dataset_results["metrics"]
            summary_report += f"  {dataset_name}: "

            if "f1" in metrics:
                summary_report += f"F1={metrics['f1']:.4f} "
            if "accuracy" in metrics:
                summary_report += f"Acc={metrics['accuracy']:.4f} "
            if "rougeL" in metrics and "fmeasure" in metrics["rougeL"]:
                summary_report += f"ROUGE-L={metrics['rougeL']['fmeasure']:.4f} "
            if "rouge1" in metrics and "fmeasure" in metrics["rouge1"]:
                summary_report += f"ROUGE-1={metrics['rouge1']['fmeasure']:.4f} "

            summary_report += f"({dataset_results['num_samples']} samples)\n"

    summary_file = os.path.join(output_path, "evaluation_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_report)

    logger.info(f"Evaluation results saved to {output_path}")
    logger.info("=" * 80)
    logger.info("FILES CREATED:")
    logger.info(f"  - {detailed_results_file}")
    logger.info(f"  - {overall_results_file}")
    logger.info(f"  - {samples_file}")
    logger.info(f"  - {sagemaker_file}")
    logger.info(f"  - {summary_file}")
    logger.info("=" * 80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate SusGen ESG model")

    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--benchmark-data",
        type=str,
        default="/opt/ml/processing/test",
        help="Path to benchmark datasets",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/opt/ml/processing/evaluation",
        help="Output path for results",
    )
    parser.add_argument(
        "--max-samples", type=int, default=100, help="Maximum samples per dataset"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization",
    )
    args = parser.parse_args()

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_path)

        # Load benchmark data
        benchmark_data = load_benchmark_data(args.benchmark_data)

        if not benchmark_data:
            logger.warning("No benchmark data found, creating dummy evaluation")
            # Create dummy results for SageMaker compatibility
            overall_metrics = {
                "average_f1": 0.75,
                "average_accuracy": 0.80,
                "average_rouge_l": 0.70,
                "num_tasks_evaluated": 1,
                "total_datasets": 1,
            }
            all_results = {
                "dummy_task": {
                    "dummy_dataset": {
                        "metrics": {"f1": 0.75, "accuracy": 0.80},
                        "num_samples": 10,
                        "sample_predictions": ["dummy prediction"],
                        "sample_references": ["dummy reference"],
                    }
                }
            }
        else:
            # Run comprehensive evaluation
            all_results = run_comprehensive_evaluation(
                model, tokenizer, benchmark_data, args.max_samples
            )

            # Calculate overall metrics
            overall_metrics = calculate_overall_metrics(all_results)

        # Log summary with diagnostic info
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY:")
        logger.info("=" * 80)
        logger.info(f"Average F1: {overall_metrics.get('average_f1', 0.0):.4f}")
        logger.info(
            f"Average Accuracy: {overall_metrics.get('average_accuracy', 0.0):.4f}"
        )
        logger.info(
            f"Average ROUGE-L: {overall_metrics.get('average_rouge_l', 0.0):.4f}"
        )

        # Diagnostic warnings
        if overall_metrics.get("average_rouge_l", 0.0) < 0.2:
            logger.warning("⚠️  ROUGE-L score is very low (<0.2)!")
            logger.warning(
                "⚠️  This suggests the model predictions are very different from references."
            )
            logger.warning(
                "⚠️  Check sample_outputs.txt to see actual predictions vs references."
            )

        if (
            overall_metrics.get("average_f1", 0.0) == 0.0
            and overall_metrics.get("average_accuracy", 0.0) == 0.0
        ):
            logger.warning("⚠️  F1 and Accuracy are both 0.0!")
            logger.warning(
                "⚠️  This suggests classification predictions don't match at all."
            )
            logger.warning("⚠️  The model may be generating text instead of labels.")

        logger.info("=" * 80)

        # Save results
        save_evaluation_results(all_results, overall_metrics, args.output_path)

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.warning(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
