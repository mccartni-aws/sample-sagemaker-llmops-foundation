"""Model utilities for evaluation."""

import logging
import os
import tarfile
import tempfile
import torch

logger = logging.getLogger(__name__)


def extract_model(model_path):
    """Extract model.tar.gz if needed."""
    logger.info(f"Loading model from: {model_path}")

    if os.path.isfile(model_path) and model_path.endswith(".tar.gz"):
        temp_dir = tempfile.mkdtemp()
        with tarfile.open(model_path, "r:gz") as tar:
            tar.extractall(temp_dir)
        return temp_dir
    elif os.path.isdir(model_path):
        tar_path = os.path.join(model_path, "model.tar.gz")
        if os.path.exists(tar_path):
            temp_dir = tempfile.mkdtemp()
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(temp_dir)
            return temp_dir
    return model_path


def load_model(model_dir, max_seq_length, load_in_4bit):
    """Load model for inference with GPU/CPU auto-detection."""
    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"

    # Disable 4-bit quantization on CPU (not supported)
    if not has_gpu and load_in_4bit:
        logger.warning("⚠️ CPU detected: disabling 4-bit quantization")
        load_in_4bit = False

    logger.info(f"Loading model on {device.upper()} with FastLanguageModel")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:
        logger.warning(f"⚠️ Unsloth failed ({e}), falling back to transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if has_gpu else torch.float32,
            device_map="auto" if has_gpu else None,
        )
        if not has_gpu:
            model = model.to("cpu")
        model.eval()

    return model, tokenizer
