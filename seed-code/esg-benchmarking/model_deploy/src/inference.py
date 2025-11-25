# inference.py (DJL Python)
import os, json, logging, traceback, tarfile, tempfile
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import boto3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from djl_python import Input, Output

LOGGER = logging.getLogger("djl-inference")
logging.basicConfig(level=logging.INFO)

# Cache per worker
STATE: Dict[str, Any] = {}


# ------------------------------- Helpers -------------------------------------
def _bool_env(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _build_prompt(messages):
    """Very simple chat-to-prompt formatting."""
    sys_msg = next(
        (m.get("content", "") for m in messages if m.get("role") == "system"), ""
    )
    user_msg = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"), ""
    )
    return f"{sys_msg}\n\nUser: {user_msg}\nAssistant:"


def _adapter_stage_dir() -> str:
    env_dir = os.environ.get("ADAPTER_STAGE_DIR")
    if env_dir:
        return env_dir
    return tempfile.mkdtemp(prefix="djl_adapters_")


def _download_s3_to_file(s3_uri: str, dst_path: str) -> None:
    u = urlparse(s3_uri)
    if u.scheme != "s3" or not u.netloc or not u.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket = u.netloc
    key = u.path.lstrip("/")
    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION"),
    )
    LOGGER.info(f"Downloading adapters from s3://{bucket}/{key} -> {dst_path}")
    s3.download_file(bucket, key, dst_path)


def _extract_tarball(tar_path: str, dst_dir: str) -> None:
    LOGGER.info(f"Extracting {tar_path} -> {dst_dir}")
    with tarfile.open(tar_path, "r:*") as tf:
        # Security fix: Validate tar members before extraction to prevent path traversal attacks
        def is_safe_path(path: str, base_path: str) -> bool:
            """Check if the path is safe for extraction (no path traversal)."""
            return (
                os.path.commonpath(
                    [os.path.realpath(os.path.join(base_path, path)), base_path]
                )
                == base_path
            )

        def safe_extract(members, path):
            """Safely extract tar members, filtering out dangerous paths."""
            for member in members:
                if member.isfile() or member.isdir():
                    # Check for path traversal attempts
                    if not is_safe_path(member.name, path):
                        LOGGER.warning(
                            f"Skipping potentially dangerous path: {member.name}"
                        )
                        continue
                    # Check for absolute paths
                    if os.path.isabs(member.name):
                        LOGGER.warning(f"Skipping absolute path: {member.name}")
                        continue
                    # Normalize the path
                    member.name = os.path.normpath(member.name)
                    yield member

        tf.extractall(dst_dir, members=safe_extract(tf, dst_dir))


def _materialize_adapters_if_needed(model_dir: str) -> str:
    """
    If ADAPTER_S3_URI points to a tar/tgz, pull & extract into a writable stage dir.
    Returns the stage dir (even if nothing was downloaded), for the caller to search.
    """
    stage_dir = _adapter_stage_dir()
    os.makedirs(stage_dir, exist_ok=True)  # /tmp is writable

    s3_uri = os.environ.get("ADAPTER_S3_URI", "").strip()
    if not s3_uri:
        LOGGER.info("ADAPTER_S3_URI not set; skipping adapter download.")
        return stage_dir

    done_flag = os.path.join(stage_dir, ".done")
    if os.path.exists(done_flag):
        LOGGER.info("Adapters already materialized in stage dir; skipping re-download.")
        return stage_dir

    with tempfile.TemporaryDirectory() as td:
        tmp_tar = os.path.join(td, "adapters.tar.gz")
        _download_s3_to_file(s3_uri, tmp_tar)
        _extract_tarball(tmp_tar, stage_dir)

    # Record where we found adapters (for debug)
    found = False
    for root, _dirs, files in os.walk(stage_dir):
        if "adapter_model.safetensors" in files:
            LOGGER.info(f"Adapter found at: {root}")
            found = True
    if not found:
        LOGGER.warning("No adapter_model.safetensors discovered after extraction.")

    with open(done_flag, "w", encoding="utf-8") as f:
        f.write("ok\n")

    return stage_dir


def _find_lora_root(*roots: str) -> Optional[str]:
    """
    Search one or more roots for a directory containing adapter_model.safetensors
    (optionally also adapter_config.json). Pick the shallowest match.
    """
    need = {"adapter_model.safetensors"}
    prefer = {"adapter_model.safetensors", "adapter_config.json"}

    best = None
    best_depth = 10**9
    for base in roots:
        for root, _dirs, files in os.walk(base):
            files_set = set(files)
            if need.issubset(files_set):
                depth = root.count(os.sep)
                if depth < best_depth:
                    best, best_depth = root, depth
                if prefer.issubset(files_set) and depth <= best_depth:
                    best, best_depth = root, depth
    return best


def _get_content_type(inputs: Input) -> str:
    # DJL can put headers under different casings/keys; normalize & fallback
    for k in ("Content-Type", "content-type", "content_type"):
        v = inputs.get_property(k)
        if v:
            return str(v).lower()
    return ""


def load_model(properties: Dict[str, Any]) -> None:
    try:
        model_dir = os.environ.get("DJL_MODEL_DIR", "/opt/ml/model")
        base_model_id = os.environ.get(
            "BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"
        )
        load_in_4bit = _bool_env("LOAD_IN_4BIT", False)

        LOGGER.info(
            f"DJL model_dir={model_dir} BASE_MODEL={base_model_id} LOAD_IN_4BIT={load_in_4bit}"
        )

        bnb_cfg = None
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)
                LOGGER.info("Using BitsAndBytes 4-bit quantization")
            except Exception:
                LOGGER.warning(
                    "BitsAndBytes not available; proceeding without 4-bit",
                    exc_info=True,
                )
                bnb_cfg = None

        local_base = os.path.join(model_dir, "base")
        if os.path.exists(os.path.join(local_base, "config.json")):
            LOGGER.info("Loading base model from local artifacts (/base)")
            tok = AutoTokenizer.from_pretrained(
                local_base, use_fast=True, local_files_only=True, trust_remote_code=True
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                local_base,
                torch_dtype=torch.float16,
                local_files_only=True,
                quantization_config=bnb_cfg,
                trust_remote_code=True,
            )
        else:
            LOGGER.info("Loading base model from hub (egress required)")
            tok = AutoTokenizer.from_pretrained(
                base_model_id, use_fast=True, trust_remote_code=True
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                quantization_config=bnb_cfg,
                trust_remote_code=True,
            )

        # Download/extract adapters into writable staging dir (e.g., /tmp/djl_adapters)
        stage_dir = _materialize_adapters_if_needed(model_dir)

        # Search both the original model_dir (if adapters were bundled) and the stage dir
        lora_root = _find_lora_root(model_dir, stage_dir)
        if lora_root:
            LOGGER.info("Found LoRA under: %s ; attaching with PEFT", lora_root)
            import peft

            LOGGER.info(peft.__version__)
            from peft import PeftModel

            mdl = PeftModel.from_pretrained(mdl, lora_root)
        else:
            LOGGER.info("No LoRA adapters found; serving base model only")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device).eval()

        STATE["model"] = mdl
        STATE["tokenizer"] = tok
        STATE["device"] = device

        LOGGER.info(f"Model loaded on device: {device}")
    except Exception:
        LOGGER.exception("load_model failed")
        raise


def handle(inputs: Input, _ctx=None) -> Output:
    """
    Accepts:
      - application/json: {"messages":[...], "parameters": {...}}
      - text/plain: raw text => {"messages":[{"role":"user","content": "..."}]}
    Returns JSON: {"generated_text": "..."} or {"error": "..."} with status_code set.
    """
    out = Output()
    try:
        if not STATE:
            load_model({})

        mdl = STATE["model"]
        tok = STATE["tokenizer"]
        device = STATE["device"]

        ct = _get_content_type(inputs)
        raw = inputs.get_as_bytes()
        if raw is None:
            raw = b""

        # Parse payload
        if "json" in ct:
            if not raw:
                out.status_code = 400
                out.add_as_json({"error": "Empty JSON body"})
                return out
            try:
                payload = json.loads(raw.decode("utf-8", errors="ignore"))
            except Exception as e:
                out.status_code = 400
                out.add_as_json({"error": f"Invalid JSON: {e}"})
                return out

        elif "text" in ct or not ct:
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                out.status_code = 400
                out.add_as_json({"error": "Empty text body"})
                return out
            payload = {"messages": [{"role": "user", "content": text}]}

        else:
            # Try JSON as a last resort
            try:
                payload = json.loads(raw.decode("utf-8", errors="ignore"))
            except Exception:
                out.status_code = 415
                out.add_as_json(
                    {
                        "error": "Unsupported media type. Use application/json or text/plain."
                    }
                )
                return out

        messages = payload.get("messages") or []
        params = payload.get("parameters") or {}
        max_new_tokens = int(params.get("max_new_tokens", 256))
        temperature = float(params.get("temperature", 0.7))
        top_p = float(params.get("top_p", 0.95))

        prompt = _build_prompt(messages)
        tokenized = tok(prompt, return_tensors="pt").to(device)

        # Optional streamer (does nothing if transformers lacks it)
        try:
            from transformers import TextStreamer

            streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        except Exception:
            streamer = None

        with torch.no_grad():
            out_ids = mdl.generate(
                **tokenized,
                do_sample=(temperature > 0),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tok.eos_token_id,
                streamer=streamer,
            )

        text = tok.decode(out_ids[0], skip_special_tokens=True)
        if "Assistant:" in text:
            text = text.split("Assistant:", 1)[-1].strip()

        out.add_as_json({"generated_text": text})
        return out

    except Exception as e:
        LOGGER.error("Handler error: %s\n%s", e, traceback.format_exc())
        out.status_code = 500
        out.add_as_json({"error": str(e)})
        return out
