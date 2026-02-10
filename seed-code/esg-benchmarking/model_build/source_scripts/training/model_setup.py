import logging
from dataclasses import dataclass
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name: str = "unsloth/mistral-7b-v0.3"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0


def setup_model(config: ModelConfig):
    """Load and configure model with LoRA."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer
