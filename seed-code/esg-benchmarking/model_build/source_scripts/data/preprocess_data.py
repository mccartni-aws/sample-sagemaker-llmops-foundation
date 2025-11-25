#!/usr/bin/env python3
"""
Data Preprocessing Script for ESG/Financial NLP Tasks

This script preprocesses the SusGen-30k dataset from HuggingFace for training,
handling the instruction-following format and preparing data splits for SageMaker training.

Dataset: WHATX/SusGen-30k (HuggingFace)
Format: JSON with instruction, input, output fields
No authentication required - public dataset
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
        "pandas": "pandas>=1.5.0",
        "scikit-learn": "scikit-learn>=1.2.0",
        "numpy": "numpy>=1.24.0",
        "datasets": "datasets>=2.14.0",
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
                    logger.warning(f"Invalid package specification: {package_spec}")
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

                # Security: Only import from predefined allowed packages
                if import_name not in [
                    "pandas",
                    "sklearn",
                    "numpy",
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
from sklearn.model_selection import train_test_split
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mistral_formal(sample):
    """
    Format sample using Mistral instruction template.

    Args:
        sample (dict): Sample with instruction, input, output keys

    Returns:
        str: Formatted prompt string
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")

    if input_text:
        prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output_text}</s>"
    else:
        prompt = f"<s>[INST] {instruction} [/INST] {output_text}</s>"

    return prompt


def llama3_formal(sample):
    """
    Format sample using Llama3 instruction template.

    Args:
        sample (dict): Sample with instruction, input, output keys

    Returns:
        str: Formatted prompt string
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")

    if input_text:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"
    else:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"

    return prompt


def load_dataset_from_huggingface(dataset_id="WHATX/SusGen-30k"):
    """
    Load dataset from HuggingFace.

    Args:
        dataset_id (str): HuggingFace dataset ID (default: WHATX/SusGen-30k)

    Returns:
        pd.DataFrame: Loaded dataset
    """
    logger.info(f"Loading dataset from HuggingFace: {dataset_id}")

    try:
        from datasets import load_dataset

        # Load dataset from HuggingFace (no token required - public dataset)
        dataset = load_dataset(dataset_id, split="train")

        # Convert to pandas DataFrame
        df = dataset.to_pandas()

        logger.info(f"Loaded {len(df)} samples from HuggingFace dataset {dataset_id}")

        # Validate required columns
        required_columns = ["instruction", "input", "output"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}")

        return df

    except Exception as e:
        logger.error(f"Failed to load dataset from HuggingFace: {str(e)}")
        raise


def preprocess_data(df, prompt_template="mistral_formal", max_length=512):
    """
    Preprocess the dataset by applying prompt templates and filtering.

    Args:
        df (pd.DataFrame): Input dataset
        prompt_template (str): Template to use ('mistral_formal' or 'llama3_formal')
        max_length (int): Maximum sequence length for filtering

    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    logger.info(f"Preprocessing data with template: {prompt_template}")

    # Select prompt template function
    if prompt_template == "mistral_formal":
        template_func = mistral_formal
    elif prompt_template == "llama3_formal":
        template_func = llama3_formal
    else:
        raise ValueError(f"Unknown prompt template: {prompt_template}")

    # Apply prompt template
    processed_samples = []

    for idx, row in df.iterrows():
        try:
            sample = {
                "instruction": row["instruction"],
                "input": row.get("input", ""),
                "output": row["output"],
            }

            # Format using template
            formatted_text = template_func(sample)

            # Basic length filtering (approximate token count)
            estimated_tokens = len(formatted_text.split())

            if estimated_tokens <= max_length:
                processed_samples.append(
                    {
                        "formatted_text": formatted_text,
                        "instruction": row["instruction"],
                        "input": row.get("input", ""),
                        "output": row["output"],
                        "estimated_tokens": estimated_tokens,
                    }
                )
            else:
                logger.debug(
                    f"Skipping sample {idx} due to length: {estimated_tokens} tokens"
                )

        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {str(e)}")
            continue

    processed_df = pd.DataFrame(processed_samples)
    logger.info(f"Processed {len(processed_df)} samples (filtered from {len(df)})")

    return processed_df


def create_data_splits(
    df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42
):
    """
    Split dataset into train, validation, and test sets.

    Args:
        df (pd.DataFrame): Input dataset
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
        random_state (int): Random seed

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger.info("Creating data splits")

    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    # First split: separate train from temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df, test_size=temp_ratio, random_state=random_state, shuffle=True
    )

    # Second split: separate val from test
    val_ratio_adjusted = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_state,
        shuffle=True,
    )

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    return train_df, val_df, test_df


def save_processed_data(train_df, val_df, test_df, output_dir):
    """
    Save processed datasets to output directory.

    Args:
        train_df (pd.DataFrame): Training dataset
        val_df (pd.DataFrame): Validation dataset
        test_df (pd.DataFrame): Test dataset
        output_dir (str): Output directory path
    """
    logger.info(f"Saving processed data to: {output_dir}")

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save as CSV files (SageMaker compatible)
    train_file = os.path.join(train_dir, "train.csv")
    val_file = os.path.join(val_dir, "validation.csv")
    test_file = os.path.join(test_dir, "test.csv")

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    logger.info(f"Training data saved to: {train_file}")
    logger.info(f"Validation data saved to: {val_file}")
    logger.info(f"Test data saved to: {test_file}")

    # Save as JSON files for compatibility
    train_json = os.path.join(train_dir, "train.json")
    val_json = os.path.join(val_dir, "validation.json")
    test_json = os.path.join(test_dir, "test.json")

    train_df.to_json(train_json, orient="records", indent=2)
    val_df.to_json(val_json, orient="records", indent=2)
    test_df.to_json(test_json, orient="records", indent=2)

    # Save preprocessing metadata
    metadata = {
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "total_samples": len(train_df) + len(val_df) + len(test_df),
        "columns": list(train_df.columns),
        "files": {
            "train_csv": train_file,
            "validation_csv": val_file,
            "test_csv": test_file,
            "train_json": train_json,
            "validation_json": val_json,
            "test_json": test_json,
        },
    }

    metadata_file = os.path.join(output_dir, "preprocessing_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Preprocessing metadata saved to: {metadata_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Preprocess SusGen-30k dataset from HuggingFace for ESG/Financial NLP training"
    )

    parser.add_argument(
        "--dataset-id",
        type=str,
        default="WHATX/SusGen-30k",
        help="HuggingFace dataset ID (default: WHATX/SusGen-30k)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ml/processing",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="mistral_formal",
        choices=["mistral_formal", "llama3_formal"],
        help="Prompt template to use",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for filtering",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    try:
        # Load data from HuggingFace
        df = load_dataset_from_huggingface(args.dataset_id)

        # Preprocess data
        processed_df = preprocess_data(
            df, prompt_template=args.prompt_template, max_length=args.max_length
        )

        # Create splits
        train_df, val_df, test_df = create_data_splits(
            processed_df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state,
        )

        # Save processed data
        save_processed_data(train_df, val_df, test_df, args.output_dir)

        logger.info("Data preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
