#!/usr/bin/env python3
"""
Combined SusGen Data Download and Preprocessing Script

This script downloads the SusGen-30k dataset and preprocesses it for training,
handling the instruction-following format and preparing data splits for SageMaker training.
"""

import argparse
import logging

from data_loader import download_susgen_data
from data_processor import preprocess_data, create_data_splits, save_processed_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download and preprocess SusGen dataset for training"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ml/processing",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="HuggingFace token for authentication",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="WHATX/SusGen-30k",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
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

    # Download data
    df = download_susgen_data(args.dataset_name, args.hf_token, args.output_dir)

    # Preprocess data
    processed_df = preprocess_data(df, max_length=args.max_length)

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

    logger.info("Data download and preprocessing completed successfully!")


if __name__ == "__main__":
    main()
