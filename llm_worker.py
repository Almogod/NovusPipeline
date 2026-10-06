"""
llm_worker.py — NovusPipeline local LLM worker process

Hosts the fine-tuned Unsloth Qwen3.5-2B LoRA adapter (on its base model) in a
separate process with its own (GPU-capable) environment, so the MCP server
stays light and a CUDA out-of-memory error cannot take it down. `local_llm.py`
starts this script with NOVUS_LLM_PYTHON (default: ./.venv-llm) and talks to it
with one JSON object per line:

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

ADAPTER_PATH = os.environ.get(
    "NOVUS_LLM_PATH", r"C:\Users\Hp\.unsloth\studio\outputs\unsloth_Qwen3.5-2B_1785882774")

_state = {"model": None, "tokenizer": None, "device": None, "load_seconds": None, "load_error": None,
          "base_model": None, "model_class": None}


def log(msg: str) -> None:
    print(f"[llm_worker] {msg}", file=sys.stderr, flush=True)


def load() -> None:
    if _state["model"] is not None or _state["load_error"] is not None:
        return
    start = time.perf_counter()
    try:
        import torch
        import transformers
        from peft import PeftConfig, PeftModel
        from transformers import AutoTokenizer

        cfg = PeftConfig.from_pretrained(ADAPTER_PATH)
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
        # The adapter was trained on the multimodal Qwen3_5ForConditionalGeneration; prefer the
        # text-only causal LM class when transformers provides one (no vision tower in memory).
        model_cls = getattr(transformers, "Qwen3_5ForCausalLM", None) or getattr(transformers, "AutoModelForImageTextToText")
        base = model_cls.from_pretrained(cfg.base_model_name_or_path, dtype=dtype,
                                         device_map="cuda" if cuda else None, local_files_only=True)
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        _state.update(model=model, tokenizer=tokenizer, device=str(next(model.parameters()).device),
                      base_model=cfg.base_model_name_or_path, model_class=model_cls.__name__,
                      load_seconds=round(time.perf_counter() - start, 1))
        log(f"loaded {model_cls.__name__} + adapter on {_state['device']} ({dtype}) in {_state['load_seconds']}s")
    except Exception as e:
        _state["load_error"] = f"{type(e).__name__}: {e}"
        log("load failed:\n" + traceback.format_exc())


def status() -> dict:
    info = {k: v for k, v in _state.items() if k not in ("model", "tokenizer")}
    info["loaded"] = _state["model"] is not None
    info["adapter_path"] = ADAPTER_PATH
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
