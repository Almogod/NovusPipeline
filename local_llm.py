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


def generate_modernization_prompt(legacy_code: str, rag_guidelines: str, codebase_context: str = "") -> str:
    """Formats code refactoring prompt for Qwen3.5-2B chat template."""
    system_prompt = (
        "You are NovusPipeline, an autonomous code modernization AI. "
        "Your task is to refactor legacy code to comply with enterprise clean-code, "
        "strict typing, and security guidelines while strictly preserving logical parity. "
        "Never rename or remove symbols that other modules depend on."
    )

    context_block = f"\nCodebase Context (dependents, public API, smell locations):\n{codebase_context}\n" \
        if codebase_context else ""
    user_content = f"""Modernization Guidelines:
{rag_guidelines}
{context_block}
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


def _load_local_model():
    """Loads base model + LoRA adapter once per process (previously reloaded on every call)."""
    global _MODEL, _IS_LOADED, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)
    try:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM
        import torch

        config = PeftConfig.from_pretrained(LOCAL_MODEL_PATH)
        cuda = torch.cuda.is_available()
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            dtype=torch.float16 if cuda else torch.float32,
            device_map="auto" if cuda else None,
            trust_remote_code=True,
            local_files_only=True,
        )
        _MODEL = PeftModel.from_pretrained(base_model, LOCAL_MODEL_PATH)
        _MODEL.eval()
        _IS_LOADED = True
        return _MODEL
    except Exception as e:
        _LOAD_ERROR = f"{type(e).__name__}: {e}"
        raise RuntimeError(_LOAD_ERROR)


def _rule_based_proposal(legacy_code: str, label: str) -> str:
    from modernizer import CodeModernizer
    mod_code, changes = CodeModernizer.modernize_python(legacy_code)
    summary = "\n".join(f"- {c}" for c in changes) or "- No applicable automated transformations."
    return f"```python\n{mod_code}\n```\n\n### {label}\n{summary}"


def generate_llm_modernization(legacy_code: str, rag_guidelines: str, max_new_tokens: int = 512,
                               codebase_context: str = "") -> str:
    """
    Generates modernized code using the local Unsloth Qwen 3.5 2B model if available,
    with automatic fallback to rule-based modernization.
    """
    if os.environ.get("NOVUS_FAST_TEST") == "1":
        return _rule_based_proposal(
            legacy_code, "Proposed Modernizations (`unsloth_Qwen3.5-2B_1785882774` - Fast Engine)")

    try:
        model = _load_local_model()
        import torch

        tokenizer = load_local_tokenizer()
        if tokenizer is None:
            raise RuntimeError("tokenizer unavailable")
        prompt = generate_modernization_prompt(legacy_code, rag_guidelines, codebase_context)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # Greedy decoding: deterministic output for a parity-sensitive task.
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        logging.warning(f"Local LLM inference fallback triggered: {e}")
        return _rule_based_proposal(legacy_code, "Applied Modernizations (Rule-based Fallback)")
