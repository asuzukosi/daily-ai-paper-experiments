#!/usr/bin/env python3
"""GPU experiment driver for WikiSkill mini LiveMath evolution.

Loads Qwen2.5-7B-Instruct (or config model_id) via transformers and runs
the WikiSkill loop for K iterations on tasks/mini_livemath.

HARD GATE: if CUDA is unavailable, exit with a clear RunPod message.
Never silently pretend a GPU run succeeded.

Smoke mode (--smoke): verify imports/config; optionally skip full model
weights download when WIKISKILL_SMOKE_SKIP_LOAD=1 (tokenizer-only check).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wikiskill.loop import LoopConfig, WikiSkillLoop, load_jsonl  # noqa: E402
from wikiskill.runtime import TransformersBackend  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except ImportError:
        pass
    data: Dict[str, Any] = {}
    resources: Dict[str, Any] = {}
    in_resources = False

    def _strip_comment(s: str) -> str:
        out, in_q, qch = [], False, ""
        for ch in s:
            if ch in ("\"", "'") and not in_q:
                in_q, qch = True, ch
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
        if not raw.strip():
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
            if isinstance(val, str) and val.lower() in ("true", "false"):
                val = val.lower() == "true"
            elif isinstance(val, str) and val.replace(".", "", 1).isdigit():
                val = float(val) if "." in val else int(val)
            data[k.strip()] = val
    if resources:
        data["resources"] = resources
    return data


def _require_cuda() -> None:
    try:
        import torch
    except ImportError:
        print(
            "[wikiskill] ERROR: torch not installed. On RunPod: bash runpod/01_setup_env.sh",
            file=sys.stderr,
        )
        sys.exit(2)
    if not torch.cuda.is_available():
        print(
            "[wikiskill] ERROR: CUDA unavailable. This GPU driver hard-gates without CUDA.\n"
            "  - CPU path: bash experiments/run_cpu_demo.sh\n"
            "  - GPU path: use a RunPod GPU pod (see runpod/00_create_pod.example.sh)",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        f"[wikiskill] CUDA OK — devices={torch.cuda.device_count()} "
        f"name={torch.cuda.get_device_name(0)}"
    )


def run_smoke(cfg: Dict[str, Any]) -> int:
    _require_cuda()
    model_id = str(cfg.get("model_id", "Qwen/Qwen2.5-7B-Instruct"))
    skip = os.environ.get("WIKISKILL_SMOKE_SKIP_LOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ) or bool(cfg.get("smoke_skip_model_load"))
    print(f"[wikiskill] smoke model_id={model_id} skip_full_load={skip}")
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"[wikiskill] ERROR: transformers missing: {e}", file=sys.stderr)
        sys.exit(2)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"[wikiskill] tokenizer OK vocab={len(tok)}")
    if skip:
        print("[wikiskill] smoke SKIPPED full model load (WIKISKILL_SMOKE_SKIP_LOAD).")
        print("[wikiskill] Full weights (~15GB bf16) download on first mini_evolve run.")
        return 0
    backend = TransformersBackend(
        model_id=model_id,
        max_new_tokens=int(cfg.get("max_new_tokens", 64)),
        temperature=float(cfg.get("temperature", 0.2)),
        torch_dtype=str(cfg.get("torch_dtype", "bfloat16")),
    )
    out = backend.generate(
        system="You are a concise math assistant.",
        user="What is 2+2? End with Final answer: <n>.",
        role="inference",
    )
    print("[wikiskill] sample generation:\n", out[:500])
    print("[wikiskill] smoke OK")
    return 0


def run_evolve(cfg: Dict[str, Any]) -> int:
    _require_cuda()
    model_id = str(cfg.get("model_id", "Qwen/Qwen2.5-7B-Instruct"))
    task_dir = _ROOT / str(cfg.get("task_dir", "tasks/mini_livemath"))
    out = _ROOT / "artifacts" / str(cfg.get("artifacts_subdir", "mini_evolve_gpu"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    backend = TransformersBackend(
        model_id=model_id,
        max_new_tokens=int(cfg.get("max_new_tokens", 512)),
        temperature=float(cfg.get("temperature", 0.2)),
        torch_dtype=str(cfg.get("torch_dtype", "bfloat16")),
    )
    loop = WikiSkillLoop(
        workspace=out,
        llm=backend,
        train=load_jsonl(task_dir / "train.jsonl"),
        val=load_jsonl(task_dir / "val.jsonl"),
        test=load_jsonl(task_dir / "test.jsonl"),
        config=LoopConfig(
            K=int(cfg.get("K", 3)),
            eval_test_each_iter=bool(cfg.get("eval_test_each_iter", True)),
        ),
    )
    result = loop.run()
    print(json.dumps(result.metrics.to_dict(), indent=2))
    print(f"[wikiskill] artifacts: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="WikiSkill GPU experiment driver")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="CUDA/import/tokenizer smoke only")
    args = ap.parse_args()
    cfg = _load_yaml(args.config)
    print(f"[wikiskill] config={args.config} name={cfg.get('name')}")
    if args.smoke or str(cfg.get("name", "")).startswith("smoke"):
        return run_smoke(cfg)
    return run_evolve(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
