"""
llm_worker.py — NovusPipeline local LLM worker process

Hosts a local LLM (NOVUS_LLM_BASE_MODEL, default unsloth/Qwen3.5-2B, plus an
optional LoRA adapter at NOVUS_LLM_PATH) in a separate process with its own
(GPU-capable) environment, so the MCP server stays light and a CUDA
out-of-memory error cannot take it down. Model files come from the Hugging Face
cache, downloading on first use unless NOVUS_LLM_OFFLINE=1. This script imports
nothing from the package, so it runs under any interpreter with torch +
transformers (+ peft for adapters). `local_llm.py` talks to it with one JSON
object per line:

  request  {"id": 1, "op": "status"}
           {"id": 2, "op": "generate", "messages": [...], "max_new_tokens": 512, "adapter": true}
  response {"id": 2, "ok": true, "text": "...", "stats": {...}}
           {"id": 2, "ok": false, "error": "..."}

Only JSON responses are written to stdout; all logging goes to stderr.
"""

import json
import os
import sys
import time
import traceback

ADAPTER_PATH = os.environ.get("NOVUS_LLM_PATH", "")
BASE_MODEL = os.environ.get("NOVUS_LLM_BASE_MODEL", "unsloth/Qwen3.5-2B")
USE_ADAPTER = bool(ADAPTER_PATH) and os.environ.get("NOVUS_LLM_ADAPTER", "1") != "0"
OFFLINE = os.environ.get("NOVUS_LLM_OFFLINE") == "1"

_state = {"model": None, "tokenizer": None, "device": None, "load_seconds": None, "load_error": None,
          "base_model": None, "model_class": None, "adapter_tensors": None, "adapter_verified": False,
          "merged": False}


def log(msg: str) -> None:
    print(f"[llm_worker] {msg}", file=sys.stderr, flush=True)


def load_adapter_weights(model) -> str:
    """
    Loads the LoRA weights explicitly and verifies them. The adapter was trained on the
    multimodal Qwen3_5ForConditionalGeneration, which names text layers
    `model.language_model.layers.*`; the text-only model names them `model.layers.*`.
    PEFT matches modules by regex (so LoRA layers are created) but then finds no weights
    under the new names and silently leaves them at their zero init — the "fine-tuned"
    model is then just the base model. Remap, load, and refuse to continue on any mismatch.
    """
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    saved = load_file(os.path.join(ADAPTER_PATH, "adapter_model.safetensors"))
    module_names = {n for n, _ in model.named_modules()}

    def remap(key: str) -> str:
        if ".language_model." in key:
            candidate = key.replace(".language_model.", ".", 1)
            if candidate.rsplit(".lora_", 1)[0] in module_names:
                return candidate
        return key

    result = set_peft_model_state_dict(model, {remap(k): v for k, v in saved.items()})
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} adapter tensors did not match the model, e.g. {unexpected[0]}")
    zero_b = [n for n, p in model.named_parameters() if "lora_B" in n and float(p.detach().float().abs().sum()) == 0.0]
    if zero_b:
        raise RuntimeError(f"{len(zero_b)} LoRA B matrices are still zero after loading (adapter not applied)")
    return f"{len(saved)}/{len(saved)}"


def load() -> None:
    if _state["model"] is not None or _state["load_error"] is not None:
        return
    start = time.perf_counter()
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer

        base_name = BASE_MODEL
        if USE_ADAPTER:
            from peft import PeftConfig
            base_name = PeftConfig.from_pretrained(ADAPTER_PATH).base_model_name_or_path
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
        # Qwen3.5 checkpoints are multimodal; the text-only causal LM class skips the vision
        # tower. Other base models load through AutoModelForCausalLM.
        model_cls = (getattr(transformers, "Qwen3_5ForCausalLM", None) if "qwen3.5" in base_name.lower() else None) \
            or transformers.AutoModelForCausalLM
        base = model_cls.from_pretrained(base_name, dtype=dtype, device_map="cuda" if cuda else None,
                                         local_files_only=OFFLINE)
        if not USE_ADAPTER:
            # Base model only (no adapter configured, or NOVUS_LLM_ADAPTER=0).
            model, adapter_tensors, merged = base, "disabled", True
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, ADAPTER_PATH)
            adapter_tensors = load_adapter_weights(model)
            # Merging folds LoRA into the base weights: no per-token adapter overhead, but
            # adapter on/off comparisons (eval_llm.py) need it unmerged (NOVUS_LLM_MERGE=0).
            merged = os.environ.get("NOVUS_LLM_MERGE", "1") == "1"
            if merged:
                model = model.merge_and_unload()
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH if USE_ADAPTER else base_name, local_files_only=OFFLINE)
        _state.update(model=model, tokenizer=tokenizer, device=str(next(model.parameters()).device),
                      base_model=base_name, model_class=model_cls.__name__,
                      adapter_tensors=adapter_tensors, adapter_verified=adapter_tensors != "disabled", merged=merged,
                      load_seconds=round(time.perf_counter() - start, 1))
        log(f"loaded {model_cls.__name__} + adapter ({adapter_tensors} tensors verified, merged={merged}) "
            f"on {_state['device']} ({dtype}) in {_state['load_seconds']}s")
    except Exception as e:
        _state["load_error"] = f"{type(e).__name__}: {e}"
        log("load failed:\n" + traceback.format_exc())


def status() -> dict:
    info = {k: v for k, v in _state.items() if k not in ("model", "tokenizer")}
    info["loaded"] = _state["model"] is not None
    info["adapter_path"] = ADAPTER_PATH or None
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    except Exception as e:  # torch missing in this environment
        info["cuda"] = f"unavailable: {e}"
    return info


def generate(messages: list, max_new_tokens: int = 512, adapter: bool = True) -> dict:
    load()
    if _state["model"] is None:
        raise RuntimeError(f"model not loaded: {_state['load_error']}")
    import contextlib
    import torch

    model, tokenizer = _state["model"], _state["tokenizer"]
    if not adapter and _state["merged"]:
        raise RuntimeError("adapter is merged into the base weights; start the worker with NOVUS_LLM_MERGE=0 "
                           "to compare against the base model")
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
    except TypeError:  # template without a thinking switch
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    disable = model.disable_adapter() if not adapter else contextlib.nullcontext()
    with torch.no_grad(), disable:
        # Greedy decoding: deterministic output for a parity-sensitive rewrite task.
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    seconds = time.perf_counter() - start
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    stats = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "new_tokens": int(new_tokens.shape[0]),
             "seconds": round(seconds, 2), "tokens_per_second": round(new_tokens.shape[0] / seconds, 1) if seconds else None,
             "adapter": adapter}
    if torch.cuda.is_available():
        stats["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return {"text": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), "stats": stats}


def main() -> None:
    # stdout is the protocol channel: keep stray prints from libraries off it.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    log(f"ready (adapter {ADAPTER_PATH})")
    for line in sys.stdin:
        if not line.strip():
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            if req["op"] == "status":
                resp = {"id": req_id, "ok": True, "status": status()}
            elif req["op"] == "load":
                load()
                resp = {"id": req_id, "ok": _state["model"] is not None, "status": status(),
                        "error": _state["load_error"]}
            elif req["op"] == "generate":
                resp = {"id": req_id, "ok": True, **generate(req["messages"], int(req.get("max_new_tokens", 512)),
                                                             bool(req.get("adapter", True)))}
            elif req["op"] == "shutdown":
                protocol_out.write(json.dumps({"id": req_id, "ok": True}) + "\n")
                protocol_out.flush()
                return
            else:
                resp = {"id": req_id, "ok": False, "error": f"unknown op {req['op']!r}"}
        except Exception as e:
            log(traceback.format_exc())
            resp = {"id": req_id, "ok": False, "error": f"{type(e).__name__}: {e}"}
        protocol_out.write(json.dumps(resp) + "\n")
        protocol_out.flush()


if __name__ == "__main__":
    main()
