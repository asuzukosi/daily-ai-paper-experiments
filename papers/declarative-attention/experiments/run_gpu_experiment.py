#!/usr/bin/env python3
"""GPU experiment driver for Declarative Attention smoke / longctx runs.

Reads a YAML config, builds a magic-chunked prompt for Qwen2.5-7B-Instruct,
generates (transformers or vLLM if available), parses DA tags, and logs
protocol adherence + projected attended-token savings under artifacts/.

HARD GATE: if CUDA is unavailable, exit with a clear RunPod message.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from da.chunking import build_synthetic_long_context  # noqa: E402
from da.controller import DAController  # noqa: E402
from da.metrics import account_decode, summarize_accounting  # noqa: E402
from da.prompting import (  # noqa: E402
    DA_INSTRUCTION_BLOCK,
    DA_SYSTEM_PROMPT,
    build_da_prompt,
    scaffold_segments_from_prompt_parts,
)


def _load_yaml_fallback(path: Path) -> Dict[str, Any]:
    """Minimal YAML subset loader (top-level keys + nested resources map)."""
    data: Dict[str, Any] = {}
    resources: Dict[str, Any] = {}
    in_resources = False
    def _strip_comment(s: str) -> str:
        # naive: drop unquoted # comments
        if "#" not in s:
            return s
        out = []
        in_q = False
        qch = ""
        for ch in s:
            if ch in ("\"", "'") and not in_q:
                in_q = True
                qch = ch
                out.append(ch)
            elif in_q and ch == qch:
                in_q = False
                out.append(ch)
            elif ch == "#" and not in_q:
                break
            else:
                out.append(ch)
        return "".join(out).rstrip()

    for raw in path.read_text().splitlines():
        raw = _strip_comment(raw)
        if raw.strip().startswith("#") or not raw.strip():
            continue
        if raw.startswith("resources:"):
            in_resources = True
            continue
        if in_resources:
            if raw.startswith(" ") or raw.startswith("\t"):
                k, _, v = raw.strip().partition(":")
                resources[k.strip()] = v.strip().strip('"').strip("'")
                continue
            in_resources = False
        if ":" in raw and not raw.startswith(" "):
            k, _, v = raw.partition(":")
            val: Any = v.strip().strip('"').strip("'")
            if isinstance(val, str) and val.replace(".", "", 1).isdigit():
                val = float(val) if "." in val else int(val)
            data[k.strip()] = val
    if resources:
        data["resources"] = resources
    return data


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_yaml_fallback(path)
    with path.open() as f:
        return yaml.safe_load(f)


def _require_cuda() -> None:
    try:
        import torch
    except ImportError:
        print(
            "ERROR: PyTorch is not installed. On RunPod, run runpod/01_setup_env.sh first.\n"
            "This experiment requires a CUDA GPU. Use experiments/run_cpu_demo.sh for the "
            "stdlib protocol simulator on CPU.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available on this machine.\n"
            "Declarative Attention smoke/longctx experiments need a GPU pod.\n"
            "  • Local CPU protocol demo:  bash experiments/run_cpu_demo.sh\n"
            "  • On RunPod:               bash runpod/01_setup_env.sh && bash runpod/03_run_smoke.sh\n"
            "Never silently pretending a GPU run succeeded.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _generate_transformers(
    model_id: str,
    prompt: str,
    *,
    max_new_tokens: int,
    max_model_len: int,
    temperature: float,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id} with transformers (device_map=auto)…")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tok, "apply_chat_template"):
        input_ids = tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
    else:
        input_ids = tok(prompt, return_tensors="pt").input_ids
    # Truncate if over budget (rough safety).
    if input_ids.shape[-1] > max_model_len - max_new_tokens:
        input_ids = input_ids[:, -(max_model_len - max_new_tokens) :]
    input_ids = input_ids.to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.time() - t0
    gen = out[0, input_ids.shape[-1] :]
    text = tok.decode(gen, skip_special_tokens=True)
    print(f"Generated {len(gen)} tokens in {elapsed:.1f}s")
    return text


def _generate_vllm(
    model_id: str,
    prompt: str,
    *,
    max_new_tokens: int,
    max_model_len: int,
    temperature: float,
) -> str:
    from vllm import LLM, SamplingParams

    print(f"Loading {model_id} with vLLM (max_model_len={max_model_len})…")
    llm = LLM(
        model=model_id,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="auto",
    )
    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    # Prefer chat template if available via tokenizer.
    tok = llm.get_tokenizer()
    if hasattr(tok, "apply_chat_template"):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = prompt
    t0 = time.time()
    outputs = llm.generate([rendered], params)
    elapsed = time.time() - t0
    text = outputs[0].outputs[0].text
    print(f"vLLM finished in {elapsed:.1f}s ({len(text.split())} approx words)")
    return text


def run(cfg: Dict[str, Any], artifacts_dir: Path) -> Dict[str, Any]:
    model_id = cfg.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
    chunk_size = int(cfg.get("chunk_size", 400))
    n_chunks = int(cfg.get("n_chunks", 12))
    max_new_tokens = int(cfg.get("max_new_tokens", 512))
    max_model_len = int(cfg.get("max_model_len", 8192))
    temperature = float(cfg.get("temperature", 0.0))
    backend = cfg.get("backend", "auto")  # auto | transformers | vllm
    prompt_path = cfg.get("prompt_path")

    if prompt_path:
        text = Path(prompt_path).read_text()
        # Split provided file into chunks by word budget.
        from da.chunking import partition_into_magic_chunks

        chunks = partition_into_magic_chunks(text, chunk_size=chunk_size)
        question = cfg.get("question", "Summarize the key facts using DA tags.")
    else:
        _, chunks, question = build_synthetic_long_context(
            n_chunks=n_chunks,
            words_per_chunk=chunk_size,
        )

    prompt = build_da_prompt(chunks, question)
    scaffold = scaffold_segments_from_prompt_parts(
        DA_SYSTEM_PROMPT, question, DA_INSTRUCTION_BLOCK
    )

    # Choose backend.
    use_vllm = False
    if backend == "vllm":
        use_vllm = True
    elif backend == "transformers":
        use_vllm = False
    else:
        try:
            import vllm  # noqa: F401

            use_vllm = True
        except ImportError:
            use_vllm = False

    if use_vllm:
        generation = _generate_vllm(
            model_id,
            prompt,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
        )
        backend_used = "vllm"
    else:
        generation = _generate_transformers(
            model_id,
            prompt,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
        )
        backend_used = "transformers"

    steps, attended_total, vanilla_total, prompt_tok = account_decode(
        generation, scaffold, chunks
    )
    summary = summarize_accounting(steps, attended_total, vanilla_total)

    # Protocol adherence: did we see focus/local tags with valid ids?
    ctrl = DAController()
    ctrl.feed_text(generation)
    valid_ids = {c.chunk_id for c in chunks if c.chunk_id is not None}
    focus_refs = set()
    from da.modes import OPEN_TAG_RE

    for m in OPEN_TAG_RE.finditer(generation):
        if m.group(1).lower() == "focus" and m.group(2):
            for part in m.group(2).split(","):
                part = part.strip()
                if part.isdigit():
                    focus_refs.add(int(part))
    invalid_refs = sorted(focus_refs - valid_ids)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "backend": backend_used,
        "config": cfg,
        "prompt_tokens_approx": prompt_tok,
        "n_chunks": len(chunks),
        "generation": generation,
        "summary": summary,
        "focus_refs": sorted(focus_refs),
        "invalid_focus_refs": invalid_refs,
        "protocol_adherence": len(invalid_refs) == 0 and summary.get("protocol_valid", False),
        "note": (
            "Attended-token figures are projected from the DA controller "
            "(full attention may still run under the hood without vLLM DA hooks)."
        ),
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = artifacts_dir / f"{cfg.get('name', 'run')}_{stamp}.jsonl"
    with out_path.open("w") as f:
        f.write(json.dumps(record) + "\n")
    metrics_path = artifacts_dir / f"{cfg.get('name', 'run')}_{stamp}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {metrics_path}")
    print(json.dumps(summary, indent=2))
    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML experiment config",
    )
    p.add_argument(
        "--artifacts",
        type=Path,
        default=_ROOT / "artifacts",
        help="Output directory for JSONL / metrics",
    )
    p.add_argument(
        "--dry-config-check",
        action="store_true",
        help="Load and print config only (no CUDA / no model load).",
    )
    args = p.parse_args()
    cfg = _load_yaml(args.config)
    cfg.setdefault("name", args.config.stem)

    print("Declarative Attention experiment", flush=True)
    for k, v in cfg.items():
        print(f"  {k}: {v}", flush=True)

    if args.dry_config_check:
        print("\nDry config check OK — no GPU work started.")
        return 0

    _require_cuda()
    run(cfg, args.artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
