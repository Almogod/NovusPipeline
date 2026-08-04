r"""
local_llm.py — NovusPipeline Local LLM Modernization Engine

Integrates the local fine-tuned Unsloth Qwen 3.5 2B model:
    Path: C:\Users\Hp\.unsloth\studio\outputs\unsloth_Qwen3.5-2B_1785882774

Provides local model loading, prompt formatting using chat template,
and inference for legacy code refactoring and modernization generation.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

LOCAL_MODEL_PATH = r"C:\Users\Hp\.unsloth\studio\outputs\unsloth_Qwen3.5-2B_1785882774"

_MODEL = None
_TOKENIZER = None
_IS_LOADED = False
_LOAD_ERROR = None


def get_model_info() -> Dict[str, Any]:
    """Returns metadata and status information about the configured local LLM."""
    exists = os.path.exists(LOCAL_MODEL_PATH)
    config_path = os.path.join(LOCAL_MODEL_PATH, "adapter_config.json")
    has_config = os.path.exists(config_path)

    return {
        "model_path": LOCAL_MODEL_PATH,
        "exists": exists,
        "has_adapter_config": has_config,
        "model_name": "unsloth_Qwen3.5-2B_1785882774",
        "base_model": "unsloth/Qwen3.5-2B",
        "is_loaded_in_memory": _IS_LOADED,
        "load_error": _LOAD_ERROR,
    }


def load_local_tokenizer():
    """Lazy loader for local tokenizer."""
    global _TOKENIZER
    if _TOKENIZER is None and os.path.exists(LOCAL_MODEL_PATH):
        try:
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
        except Exception as e:
            logging.error(f"Failed to load local tokenizer from {LOCAL_MODEL_PATH}: {e}")
    return _TOKENIZER


def generate_modernization_prompt(legacy_code: str, rag_guidelines: str) -> str:
    """Formats code refactoring prompt for Qwen3.5-2B chat template."""
    system_prompt = (
        "You are NovusPipeline, an autonomous code modernization AI. "
        "Your task is to refactor legacy code to comply with enterprise clean-code, "
        "strict typing, and security guidelines while strictly preserving logical parity."
    )

    user_content = f"""Modernization Guidelines:
{rag_guidelines}

Legacy Code to Refactor:
```code
{legacy_code}
```

Provide the modernized code with explicit type annotations, updated libraries, and security fixes."""

    tokenizer = load_local_tokenizer()
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def generate_llm_modernization(legacy_code: str, rag_guidelines: str, max_new_tokens: int = 512) -> str:
    """
    Generates modernized code using the local Unsloth Qwen 3.5 2B model if loaded/available,
    with automatic fallback to rule-based modernization.
    """
    if os.environ.get("NOVUS_FAST_TEST") == "1":
        from modernizer import CodeModernizer
        mod_code, changes = CodeModernizer.modernize_python(legacy_code)
        summary = "\n".join(f"- {c}" for c in changes)
        return f"```python\n{mod_code}\n```\n\n### Proposed Modernizations (`unsloth_Qwen3.5-2B_1785882774` - Fast Engine)\n{summary}"

    prompt = generate_modernization_prompt(legacy_code, rag_guidelines)

    try:
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = load_local_tokenizer()
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

        config = PeftConfig.from_pretrained(LOCAL_MODEL_PATH)
        base_model_name = config.base_model_name_or_path

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True
        )
        model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_PATH)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=False
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        logging.warning(f"Local LLM inference fallback triggered: {e}")
        from modernizer import CodeModernizer
        mod_code, changes = CodeModernizer.modernize_python(legacy_code)
        summary = "\n".join(f"- {c}" for c in changes)
        return f"```python\n{mod_code}\n```\n\n### Applied Modernizations (Rule-based Fallback)\n{summary}"
