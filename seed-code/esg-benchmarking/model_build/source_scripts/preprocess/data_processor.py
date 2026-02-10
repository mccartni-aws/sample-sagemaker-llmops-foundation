import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(df, max_length=512):
    """
    Preprocess dataset - only basic cleaning, NO formatting.

    Args:
        df (pd.DataFrame): Input dataset with instruction, input, output
        max_length (int): Maximum token estimate for filtering

    Returns:
        pd.DataFrame: Cleaned dataset with instruction, input, output only
    """
    processed_samples = []

    for idx, row in df.iterrows():
        try:
            instruction = str(row.get("instruction", "")).strip()
            input_text = str(row.get("input", "")).strip()
            output_text = str(row.get("output", "")).strip()

            # Skip empty samples
            if not instruction or not output_text:
                continue

            # Rough token estimate for filtering
            estimated_tokens = (
                len(instruction.split())
                + len(input_text.split())
                + len(output_text.split())
            )

            if estimated_tokens <= max_length:
                processed_samples.append(
                    {
                        "instruction": instruction,
                        "input": input_text,
                        "output": output_text,
                    }
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
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Preprocessing metadata saved to: {metadata_file}")
