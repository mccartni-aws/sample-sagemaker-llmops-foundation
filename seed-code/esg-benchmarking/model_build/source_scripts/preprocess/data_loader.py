import logging
from huggingface_hub import login
from datasets import load_dataset
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_susgen_data(dataset_name, hf_token, output_dir):
    """
    Download SusGen dataset from HuggingFace.

    Args:
        dataset_name (str): Name of the dataset on HuggingFace
        hf_token (str): HuggingFace token for authentication
        output_dir (str): Output directory for downloaded data

    Returns:
        pd.DataFrame: Downloaded dataset
    """
    logger.info(f"Downloading dataset: {dataset_name}")

    # Login to HuggingFace if token provided
    if hf_token:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace")

    try:
        # Download dataset
        dataset = load_dataset(dataset_name, split="train")
        logger.info(f"Downloaded {len(dataset)} samples")

        # Convert to pandas DataFrame
        df = dataset.to_pandas()

        # Save raw dataset
        raw_file = os.path.join(output_dir, "susgen_30k_full.json")
        df.to_json(raw_file, orient="records", indent=2)
        logger.info(f"Saved raw dataset to: {raw_file}")

        return df

    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        raise
