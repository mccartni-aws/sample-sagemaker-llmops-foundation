import logging
import pandas as pd
from datasets import Dataset
import os

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}</s>"""


def formatting_prompts_func(examples):
    """Format examples into prompt template with EOS token."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for instruction, input_text, output in zip(instructions, inputs, outputs):
        if not input_text or pd.isna(input_text):
            input_text = ""
        text = PROMPT_TEMPLATE.format(instruction, input_text, output)
        texts.append(text)

    return {"text": texts}


def load_csv_dataset(file_path, sample_fraction=None, seed=3407):
    """Load CSV and convert to HF dataset."""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    if sample_fraction and 0 < sample_fraction < 1.0:
        n_keep = max(1, int(len(df) * sample_fraction))
        df = df.sample(n=n_keep, random_state=seed).reset_index(drop=True)
        logger.info(f"Sampled {len(df)} rows ({sample_fraction * 100:.1f}%)")

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def load_datasets(
    train_data_dir, val_data_dir, train_fraction=None, eval_fraction=None, seed=3407
):
    """Load training and validation datasets."""
    train_file = os.path.join(train_data_dir, "train.csv")
    logger.info(f"Loading training data from: {train_file}")
    train_dataset = load_csv_dataset(
        train_file, sample_fraction=train_fraction, seed=seed
    )

    eval_dataset = None
    val_file = os.path.join(val_data_dir, "validation.csv")
    if os.path.exists(val_file):
        logger.info(f"Loading validation data from: {val_file}")
        eval_dataset = load_csv_dataset(
            val_file, sample_fraction=eval_fraction, seed=seed
        )

    return train_dataset, eval_dataset
