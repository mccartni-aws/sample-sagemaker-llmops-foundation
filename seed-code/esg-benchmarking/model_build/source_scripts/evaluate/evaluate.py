"""
ESG Model Evaluation Script
Evaluates trained model using ROUGE scores on test dataset.
"""

import argparse
import logging
import warnings

from model_utils import extract_model, load_model
from eval_utils import (
    load_test_dataset,
    generate_predictions,
    calculate_rouge,
    save_evaluation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ESG model")
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-data", type=str, default="/opt/ml/processing/test")
    parser.add_argument(
        "--output-path", type=str, default="/opt/ml/processing/evaluation"
    )
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--load-in-4bit", type=lambda x: str(x).lower() == "true", default=True
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    import torch

    args = parse_args()

    model_dir = extract_model(args.model_path)
    model, tokenizer = load_model(model_dir, args.max_seq_length, args.load_in_4bit)
    test_data = load_test_dataset(args.test_data)

    logger.info(f"Loaded {len(test_data)} test samples")

    # Auto-reduce samples on CPU for faster evaluation
    num_batches = args.num_batches
    if not torch.cuda.is_available():
        num_batches = max(1, args.num_batches // 4)
        logger.info(
            f"🖥️  CPU detected: reducing batches from {args.num_batches} to {num_batches} for faster evaluation"
        )

    predictions, references = generate_predictions(
        model,
        tokenizer,
        test_data,
        num_batches,
        args.batch_size,
        args.max_seq_length,
        args.max_new_tokens,
    )

    rouge_scores = calculate_rouge(predictions, references)
    save_evaluation(rouge_scores, args.output_path)

    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
