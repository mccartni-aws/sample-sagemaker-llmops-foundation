"""Evaluation utilities for data loading, inference, and metrics."""

import json
import logging
import os
import pandas as pd
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def load_test_dataset(test_dir):
    """Load test dataset from CSV or JSON."""
    logger.info(f"Loading test data from: {test_dir}")

    test_csv = os.path.join(test_dir, "test.csv")
    test_json = os.path.join(test_dir, "test.json")

    if os.path.exists(test_csv):
        return pd.read_csv(test_csv).to_dict("records")
    elif os.path.exists(test_json):
        with open(test_json, "r") as f:
            return json.load(f)
    raise FileNotFoundError(f"No test.csv or test.json found in {test_dir}")


def generate_predictions(
    model, tokenizer, test_data, num_batches, batch_size, max_seq_length, max_new_tokens
):
    """Generate predictions on test data."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Generating predictions on {device.upper()} for {num_batches} batches of {batch_size}"
    )

    total_samples = num_batches * batch_size
    test_subset = test_data[:total_samples]
    predictions, references = [], []

    for i in range(0, len(test_subset), batch_size):
        batch = test_subset[i : i + batch_size]
        batch_prompts = []

        for sample in batch:
            references.append(sample.get("output", ""))
            prompt = PROMPT_TEMPLATE.format(
                sample.get("instruction", ""), sample.get("input", ""), ""
            )
            batch_prompts.append(prompt)

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            predictions.append(generated.strip())

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i // batch_size + 1}/{num_batches} batches")

    logger.info(f"Generated {len(predictions)} predictions")
    return predictions, references


def calculate_rouge(predictions, references):
    """Calculate ROUGE scores."""
    logger.info("Calculating ROUGE scores")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        if pred and ref:
            result = scorer.score(str(ref), str(pred))
            scores["rouge1"].append(result["rouge1"].fmeasure)
            scores["rouge2"].append(result["rouge2"].fmeasure)
            scores["rougeL"].append(result["rougeL"].fmeasure)

    avg_scores = {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}

    logger.info(f"ROUGE-1: {avg_scores['rouge1']:.4f}")
    logger.info(f"ROUGE-2: {avg_scores['rouge2']:.4f}")
    logger.info(f"ROUGE-L: {avg_scores['rougeL']:.4f}")

    return avg_scores


def save_evaluation(rouge_scores, output_path):
    """Save evaluation.json with ROUGE scores."""
    os.makedirs(output_path, exist_ok=True)

    evaluation = {"rouge_metrics": {k: {"value": v} for k, v in rouge_scores.items()}}

    eval_file = os.path.join(output_path, "evaluation.json")
    with open(eval_file, "w") as f:
        json.dump(evaluation, f, indent=2)

    logger.info(f"Saved evaluation results to {eval_file}")
